"""
src/retrieval/routing.py

Lightweight Query Router and Unified Context Retriever for SysML v2 GraphRAG.
Classifies incoming queries to bypass deep PPR/path-stitching when unneeded and
orchestrates end-to-end context retrieval.
"""

import time
from typing import Dict, Any, List, Optional
import networkx as nx

# Imports from adjacent retrieval modules
from candidate.retrieval.grounding import ground_terms_via_embeddings
from candidate.retrieval.query_identity import check_query_identity
from candidate.retrieval.composition import recognize_typed_composition, resolve_composition_identity, retrieve_typed_composition
from candidate.retrieval.composition import recognize_scoped_inventory, resolve_inventory_identity, retrieve_scoped_inventory
from candidate.retrieval.inventory_scope import confirm_inventory_scope, propose_inventory_clarification
from candidate.retrieval.endpoint_focus import recognize_endpoint_focus, resolve_endpoint_focus
from candidate.retrieval.allocation import recognize_allocation, retrieve_allocation
from candidate.retrieval.action_order import recognize_action_order, retrieve_action_order
from candidate.retrieval.traversal import SubgraphPruner, stringify_subgraph_markdown_tree
from candidate.retrieval.traceability import recognize_traceability, retrieve_traceability


class QueryRouter:
    """
    Adaptive query router and execution orchestrator for GraphRAG retrieval pipeline.
    
    Routes:
      - FAST_REFUSAL: Low anchor similarity / no anchors found. Skips LLM & graph traversal.
      - IDENTITY_UNVERIFIED: Explicit name lacks an exact projected match.
      - CLARIFICATION_REQUIRED: Explicit name has multiple projected identities.
      - GROUNDED_INVENTORY_CLARIFICATION: An unresolved owner has cited candidates.
      - TYPED_COMPOSITION: Bounded root/usage/type/direct-parts evidence lookup.
      - COMPOSITION_UNVERIFIED: Required declaration/type chain is incomplete.
      - SCOPED_INVENTORY: Verified scope's direct part usages and declared types.
      - INVENTORY_UNVERIFIED: Inventory membership cannot be established.
      - FOCUSED_ENDPOINTS: Traverse explicit scoped endpoint targets and their connectors.
      - ENDPOINT_FOCUS_UNVERIFIED: An explicit occurrence lacks a complete connector record.
      - DIRECT_LOOKUP: Single entity / 1-hop query. Bypasses deep PPR & path-stitching.
      - DEEP_GRAPHRAG: Multi-entity / multi-hop query. Runs full Strategy C pipeline.
    """

    def __init__(
        self,
        min_grounding_threshold: float = 0.25,
        multi_entity_threshold: int = 2,
        default_top_percentile: float = 0.35,
        default_max_path_len: int = 4,
        pruner=None,
        *,
        grounded_inventory_clarification: bool = False,
    ):
        self.min_grounding_threshold = min_grounding_threshold
        self.multi_entity_threshold = multi_entity_threshold
        self.default_top_percentile = default_top_percentile
        self.default_max_path_len = default_max_path_len
        self.pruner = pruner if pruner is not None else SubgraphPruner()
        self.grounded_inventory_clarification = grounded_inventory_clarification

    def confirm_inventory(self, graph, pending, embedder, precomputed_embeddings, *, accepted, usage_id=None,
                          precomputed_node_text_sha256=None):
        """Resume an inventory from an explicit choice and a rechecked graph path."""
        if not self.grounded_inventory_clarification:
            return {'status': 'disabled', 'scope_id': None}
        confirmation = confirm_inventory_scope(graph, pending, accepted=accepted, usage_id=usage_id)
        if confirmation['status'] != 'confirmed':
            return confirmation
        candidate = confirmation['candidate']
        query = f"Which parts are listed directly in `{candidate['owner_name']}`'s `{candidate['usage_name']}`?"
        retrieval = self.retrieve_context(
            query, graph, embedder, precomputed_embeddings,
            precomputed_node_text_sha256=precomputed_node_text_sha256)
        return {**confirmation, 'retrieval': retrieval}

    def route_query(
        self,
        query_text: str,
        anchors: List[str],
        max_similarity_score: float
    ) -> Dict[str, Any]:
        """
        Determines the optimal execution route based on grounding metrics.
        """
        # Route 1: Retrieval could not establish sufficiently grounded evidence.
        if len(anchors) == 0 or max_similarity_score < self.min_grounding_threshold:
            return {
                "route": "FAST_REFUSAL",
                "reason": f"Max similarity ({max_similarity_score:.2f}) < threshold ({self.min_grounding_threshold}).",
                "execute_llm": False,
                "default_response": (
                    "I cannot establish sufficiently grounded evidence for this question "
                    "in the current retrieval. This does not establish absence from the "
                    "full SysML model. Please provide a qualified identifier or scope.")
            }

        # Route 2: Direct 1-Hop Lookup
        elif len(anchors) < self.multi_entity_threshold:
            return {
                "route": "DIRECT_LOOKUP",
                "reason": f"Single anchor detected ({len(anchors)} anchor): outgoing neighborhood plus named-interface endpoint records.",
                "execute_llm": True,
                "default_response": None
            }

        # Route 3: Deep GraphRAG (Multi-Hop / Cross-Package)
        else:
            return {
                "route": "DEEP_GRAPHRAG",
                "reason": f"Multiple anchor entities detected ({len(anchors)} anchors). Deep PPR & path stitching required.",
                "execute_llm": True,
                "default_response": None
            }

    def retrieve_context(
        self,
        query_text: str,
        graph: nx.Graph,
        embedder: Any,
        precomputed_embeddings: Any,
        top_percentile: Optional[float] = None,
        max_path_len: Optional[int] = None,
        precomputed_node_text_sha256: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Unified entry point: Grounds query terms, routes execution, prunes the AST graph,
        and returns serialized Markdown context alongside benchmarking metadata.
        """
        start_time = time.time()
        p = top_percentile if top_percentile is not None else self.default_top_percentile
        l = max_path_len if max_path_len is not None else self.default_max_path_len

        # 1. Compute Max Cosine Similarity Score
        import sentence_transformers.util as st_util
        query_emb = embedder.encode(query_text, convert_to_tensor=True)
        sim_scores = st_util.cos_sim(query_emb, precomputed_embeddings)[0]
        max_sim = float(sim_scores.max().item())

        # 2. Ground Anchor Entities
        anchors = ground_terms_via_embeddings(
            G=graph,
            query_text=query_text,
            embedder=embedder,
            max_delta=0.12,
            min_similarity=0.35,
            precomputed_node_embs=precomputed_embeddings,
            precomputed_node_text_sha256=precomputed_node_text_sha256
        )

        anchor_list = sorted(anchors)

        # 3. Decision Routing
        routing_info = self.route_query(
            query_text=query_text,
            anchors=anchor_list,
            max_similarity_score=max_sim
        )
        composition_request = recognize_typed_composition(query_text)
        inventory_request = recognize_scoped_inventory(
            query_text, allow_possessive=self.grounded_inventory_clarification)
        endpoint_request = recognize_endpoint_focus(query_text)
        allocation_request = recognize_allocation(query_text)
        action_order_request = recognize_action_order(query_text)
        # Complete recognized requests specify a direct-member role and its root.
        # Resolve them jointly instead of treating the member as a global name.
        clarification = None
        if inventory_request and inventory_request.get('possessive_scope'):
            clarification = propose_inventory_clarification(graph, inventory_request['root_reference'])
            resolution = clarification['resolution']
            root = resolution['scope_id']
            identity = {'version': 'possessive-inventory-identity-v1',
                        'status': resolution['status'], 'root_id': root,
                        'explicit_scope': graph.nodes[root]['qualified_name'] if root is not None else None,
                        'mentions': []}
        else:
            identity = (resolve_inventory_identity(graph, inventory_request) if inventory_request
                        else resolve_composition_identity(graph, composition_request) if composition_request
                        else resolve_endpoint_focus(graph, endpoint_request) if endpoint_request
                        else {'status': 'unchecked', 'mentions': [], 'explicit_scope': None} if allocation_request or action_order_request
                        else check_query_identity(graph, query_text))
        if clarification and identity['status'] != 'matched':
            needs_confirmation = clarification['status'] == 'confirmation_required'
            response = clarification.get('response') or (
                'Please specify a unique owner and member usage with a verifiable declaration and type.')
            route = ('GROUNDED_INVENTORY_CLARIFICATION' if needs_confirmation
                     else 'CLARIFICATION_REQUIRED' if identity['status'] == 'ambiguous'
                     else 'INVENTORY_UNVERIFIED')
            routing_info = {'route': route, 'reason': response, 'execute_llm': False,
                            'default_response': response}
        elif identity['status'] in {'unverified', 'ambiguous'}:
            uncertain = [row for row in identity['mentions'] if row['status'] != 'matched']
            if identity['status'] == 'unverified':
                names = ', '.join(row['mention'] for row in uncertain if row['status'] == 'unverified')
                response = (f"I cannot verify an exact match for {names} in the current graph projection. "
                            "Please check the identifier or provide its qualified scope/source. "
                            "This does not establish absence from the full SysML model.")
                route = 'IDENTITY_UNVERIFIED'
            else:
                options = '; '.join(row['mention'] + ': ' + ', '.join(
                    candidate['kind'] + ' ' + candidate['qualified_name'] for candidate in row['candidates'])
                    for row in uncertain)
                response = f"Please clarify the intended scoped element: {options}."
                route = 'CLARIFICATION_REQUIRED'
            routing_info = {'route': route, 'reason': response,
                            'execute_llm': False, 'default_response': response}
        intent = recognize_traceability(query_text)
        # Explicit requirement IDs are exact evidence, independent of a dense
        # model's ability to embed an opaque code. The identity guard above
        # resolves scope/ambiguity first; the measured similarity is unchanged.
        if (identity['status'] == 'matched' and not inventory_request and not composition_request
                and not endpoint_request and intent not in {'satisfaction', 'satisfaction_allocation'}):
            exact_id_nodes = {node for mention in identity['mentions'] for node in mention['node_ids']
                              if graph.nodes[node].get('requirement_id') and (
                                  mention['mention'] == graph.nodes[node]['requirement_id']
                                  or mention['mention'].endswith('::'+graph.nodes[node]['requirement_id']))}
            if exact_id_nodes:
                explicit_nodes = {node for mention in identity['mentions'] for node in mention['node_ids']
                                  if graph.nodes[node].get('qualified_name') != identity.get('explicit_scope')}
                anchor_list = sorted(exact_id_nodes | explicit_nodes)
                routing_info = {'route': 'DIRECT_LOOKUP' if len(anchor_list) == 1 else 'DEEP_GRAPHRAG',
                                'execute_llm': True, 'default_response': None,
                                'reason': 'Exact scoped requirement ID match; dense similarity is not the identity gate.'}
        composition = None
        if composition_request and routing_info['execute_llm']:
            composition_graph, composition = retrieve_typed_composition(graph, composition_request, identity=identity)
            complete = composition['status'] == 'complete_projected_chain'
            response = None if complete else (
                'I cannot verify the requested composition chain in the current graph. '
                + composition['reason'] + ' This does not establish absence from the full SysML model.')
            routing_info = {'route': 'TYPED_COMPOSITION' if complete else 'COMPOSITION_UNVERIFIED',
                            'execute_llm': complete, 'reason': composition['reason'],
                            'default_response': response}
        inventory, inventory_records, inventory_graph = None, [], None
        if inventory_request and identity['status']=='matched':
            inventory_graph,inventory_records,inventory=retrieve_scoped_inventory(graph,inventory_request,identity)
            available=bool(inventory_records)
            routing_info={'route':'SCOPED_INVENTORY' if available else 'INVENTORY_UNVERIFIED',
                          'execute_llm':available,'default_response':None if available else inventory['reason'],
                          'reason':inventory['reason']}
            anchor_list=[identity['root_id']]
        traceability, trace_records, trace_graph = None, [], None
        if ((allocation_request or action_order_request or intent) and not inventory_request and not composition_request
                and not endpoint_request and identity['status'] not in {'unverified', 'ambiguous'}):
            if allocation_request:
                trace_graph, trace_records, traceability = retrieve_allocation(graph, allocation_request)
            elif action_order_request:
                trace_graph, trace_records, traceability = retrieve_action_order(graph, action_order_request)
            else:
                scores = {node: float(sim_scores[index]) for index, node in enumerate(graph.nodes())}
                trace_graph, trace_records, traceability = retrieve_traceability(
                    graph, query_text, identity, scores, intent, max_hops=l)
            status = traceability['status']
            if status == 'paths_found':
                anchor_list = traceability['seeds']
                routing_info = {'route': 'TYPED_TRACEABILITY', 'execute_llm': True, 'default_response': None,
                                'reason': 'Bounded modeled traceability paths; complete path records are retained for evidence packing.'}
            else:
                ambiguous = status == 'ambiguous'
                response = ('Please clarify the intended scoped declaration: '+', '.join(
                    graph.nodes[n].get('qualified_name', str(n)) for n in traceability['grounding']['candidates'])
                    if ambiguous else 'I cannot establish a supported traceability path in the current graph projection. '
                    'This does not establish absence from the complete model.')
                routing_info = {'route': 'CLARIFICATION_REQUIRED' if ambiguous else 'TRACEABILITY_UNVERIFIED',
                                'execute_llm': False, 'default_response': response, 'reason': response}
        route_name = routing_info["route"]

        # Expand traversal seeds only after the primary-grounding refusal gate.
        # Keep candidate discovery distinct from primary anchors and port semantics.
        discovery = None
        retrieval_seeds = []
        eligible_graph = graph
        endpoint_focus = ({'version': 'explicit-endpoint-traversal-v1', 'request': endpoint_request,
                           'status': 'not_executed', 'endpoint_anchors': identity.get('endpoint_anchors', [])}
                          if endpoint_request else None)
        if routing_info['execute_llm'] and route_name not in {'TYPED_COMPOSITION', 'TYPED_TRACEABILITY','SCOPED_INVENTORY'}:
            traversal_anchors = identity['endpoint_anchors'] if endpoint_request else anchor_list
            discovery = self.pruner.discover_interface_connectors(graph, traversal_anchors)
            if endpoint_request:
                unmatched = sorted(set(traversal_anchors) - {match['anchor'] for match in discovery['matches']})
                endpoint_focus['unmatched_occurrences'] = unmatched
                complete = bool(traversal_anchors) and not unmatched
                endpoint_focus['status'] = 'focused' if complete else 'unverified'
                if complete:
                    allowed_interfaces = set(discovery['connectors'])
                    allowed_occurrences = {target for connector in allowed_interfaces
                                           for _, target, edge in graph.out_edges(connector, data=True)
                                           if edge.get('relationship') in {'source_endpoint', 'target_endpoint'}}
                    excluded = {node for node, data in graph.nodes(data=True)
                                if (data.get('kind') == 'interface' and node not in allowed_interfaces)
                                or (data.get('kind') == 'port_binding' and node not in allowed_occurrences)}
                    eligible_graph = graph.subgraph(set(graph) - excluded).copy()
                    endpoint_focus['eligible_interfaces'] = sorted(allowed_interfaces)
                    endpoint_focus['excluded_interface_count'] = sum(graph.nodes[node].get('kind') == 'interface' for node in excluded)
                response = None if complete else (
                    'I cannot verify complete named-interface records for every requested endpoint occurrence '
                    'in the current graph projection. This does not establish absence from the full SysML model.')
                routing_info = {'route': 'FOCUSED_ENDPOINTS' if complete else 'ENDPOINT_FOCUS_UNVERIFIED',
                                'execute_llm': complete, 'default_response': response,
                                'reason': 'Traversal focused on every explicitly named scoped endpoint.' if complete else response}
                route_name = routing_info['route']
            if routing_info['execute_llm']:
                retrieval_seeds = sorted(set(traversal_anchors).union(discovery['connectors']))

        # 4. Context Extraction Based on Route
        endpoint_completion = None
        selected = graph.subgraph([]).copy()
        if not routing_info['execute_llm']:
            context_md = ""
            subgraph_nodes = 0
            subgraph_edges = 0

        elif route_name == 'SCOPED_INVENTORY':
            selected=inventory_graph
            retrieval_seeds=[identity['root_id']]
            context_md=stringify_subgraph_markdown_tree(selected)
            subgraph_nodes=selected.number_of_nodes()
            subgraph_edges=selected.number_of_edges()

        elif route_name == 'TYPED_TRACEABILITY':
            selected = trace_graph
            retrieval_seeds = traceability['seeds']
            context_md = stringify_subgraph_markdown_tree(selected)
            subgraph_nodes = selected.number_of_nodes()
            subgraph_edges = selected.number_of_edges()

        elif route_name == 'TYPED_COMPOSITION':
            selected = composition_graph
            context_md = stringify_subgraph_markdown_tree(composition_graph)
            subgraph_nodes = composition_graph.number_of_nodes()
            subgraph_edges = composition_graph.number_of_edges()

        elif route_name == "DIRECT_LOOKUP":
            selected_nodes = set(retrieval_seeds)
            for anchor_node in anchor_list:
                if graph.has_node(anchor_node):
                    selected_nodes.update(graph.neighbors(anchor_node))
            subgraph = graph.subgraph(selected_nodes)
            subgraph = self.pruner.preserve_interface_endpoints(subgraph, graph)
            selected = subgraph
            endpoint_completion = subgraph.graph.get('interface_endpoint_completion')
            context_md = stringify_subgraph_markdown_tree(subgraph)
            subgraph_nodes = subgraph.number_of_nodes()
            subgraph_edges = subgraph.number_of_edges()

        else:  # DEEP_GRAPHRAG
            subgraph_ppr = self.pruner.prune_with_ppr_decay(
                G=eligible_graph, anchor_nodes=retrieval_seeds, top_percentile=p
            )
            subgraph_final = self.pruner.prune_with_path_stitching(
                G=subgraph_ppr, anchor_nodes=retrieval_seeds, max_path_len=l,
                original_graph=eligible_graph
            )
            endpoint_completion = subgraph_final.graph.get('interface_endpoint_completion')
            selected = subgraph_final
            context_md = stringify_subgraph_markdown_tree(subgraph_final)
            subgraph_nodes = subgraph_final.number_of_nodes()
            subgraph_edges = subgraph_final.number_of_edges()

        elapsed_ms = (time.time() - start_time) * 1000

        return {
            "route": route_name,
            "context_md": context_md,
            "execute_llm": routing_info["execute_llm"],
            "default_response": routing_info["default_response"],
            "selected_evidence": {
                "records": trace_records+inventory_records,
                "nodes": [{"id": node, **data} for node, data in selected.nodes(data=True)],
                "edges": [{"source": source, "target": target, **data}
                          for source, target, data in selected.edges(data=True)]},
            "metadata": {
                "max_similarity": round(max_sim, 4),
                "num_anchors": len(anchor_list),
                "anchors": anchor_list,
                "retrieval_seeds": retrieval_seeds,
                "interface_discovery": discovery,
                "subgraph_nodes": subgraph_nodes,
                "subgraph_edges": subgraph_edges,
                "context_chars": len(context_md),
                "retrieval_time_ms": round(elapsed_ms, 2),
                "reason": routing_info["reason"],
                "interface_endpoint_completion": endpoint_completion,
                "query_identity": identity,
                "composition_request": composition_request,
                "typed_composition": composition,
                "typed_traceability": traceability,
                "scoped_inventory": inventory,
                **({"inventory_clarification": clarification} if clarification is not None else {}),
                "endpoint_focus": endpoint_focus
            }
        }

# =====================================================================
# Intent-Aware Routing & Faithful Refusal Module Level Helpers
# =====================================================================

FAITHFUL_REFUSAL_KEYWORD = "NO INFORMATION AVAILABLE IN CONTEXT"

FAITHFUL_SYSTEM_PROMPT = (
    "You are an expert Model-Based Systems Engineering (MBSE) assistant analyzing SysML v2 architectures.\n"
    "Synthesize a clear technical answer to the user query using the provided model context.\n"
    f"Note: If the provided context is completely irrelevant or contains zero facts regarding the query, "
    f"respond ONLY with '{FAITHFUL_REFUSAL_KEYWORD}'."
)


def check_faithful_refusal(text: str) -> bool:
    if not text:
        return True
    clean_text = text.strip().upper()
    exact_keyword = FAITHFUL_REFUSAL_KEYWORD.upper()
    first_line = clean_text.split('\n')[0].strip()
    return clean_text == exact_keyword or first_line == exact_keyword


def get_intent_hyperparameters(category: str) -> Dict[str, Any]:
    """
    Returns tuned Strategy C hyperparameters based on query intent/category.
    """
    cat_lower = str(category).lower()
    if any(k in cat_lower for k in ["requirement", "traceability", "behavioral", "flow"]):
        return {
            "t0": 0.18,                   # Lowered from 0.25 to capture multi-hop requirement bindings
            "similarity_floor": 0.18,
            "decay_rate": 0.88,
            "max_anchors": 20,            # Increased from 15 to expand anchor coverage
            "top_percentile": 0.35,
            "category_type": "Requirement/Behavioral Flow",
        }
    else:  # Architectural / Structural Containment
        return {
            "t0": 0.30,
            "similarity_floor": 0.30,
            "decay_rate": 0.85,
            "max_anchors": 12,
            "top_percentile": 0.35,
            "category_type": "Architectural Containment",
        }


def format_faithful_prompt(query_text: str, context_str: str) -> str:
    """
    Formats the prompt with explicit faithful refusal instructions and graph context.
    """
    ctx = context_str.strip() if context_str and context_str.strip() else "No matching graph elements found in model context."
    return (
        f"{FAITHFUL_SYSTEM_PROMPT}\n\n"
        f"--- GRAPH CONTEXT ---\n"
        f"{ctx}\n"
        f"---------------------\n\n"
        f"USER QUERY: {query_text}\n\n"
        f"ANSWER:"
    )
