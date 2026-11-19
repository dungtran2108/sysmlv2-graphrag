"""
src/retrieval/traversal.py

Graph Traversal, PPR Decay Pruning, Path-Stitching, and Tree Serialization
for SysML v2 GraphRAG.
"""

from typing import List, Set, Dict, Any, Optional
import logging
import networkx as nx
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


def _provenance(data):
    if data.get('file') and data.get('line') is not None:
        return f" [source: {data['file']}:{data['line']}]"
    return ''


def stringify_subgraph_triples(subgraph: nx.DiGraph) -> str:
    """Converts a NetworkX subgraph into clean RDF-style triples for LLM prompts."""
    if subgraph.number_of_edges() == 0 and subgraph.number_of_nodes() == 0:
        return "No matching multi-hop graph paths found."

    doc_lines = ["--- Extracted Multi-Hop Graph Traversal Paths ---"]

    # List Subgraph Nodes
    for node_id, data in sorted(subgraph.nodes(data=True), key=lambda row: str(row[0])):
        name = data.get("qualified_name", data.get("name", str(node_id)))
        node_type = data.get("type", data.get("kind", "Element"))
        doc_lines.append(f"Node: [{name}] (Type: {node_type})" + _provenance(data))
        for detail in data.get('evidence_details', []):
            doc_lines.append('  Evidence: ' + detail)

    doc_lines.append("\nRelationships:")
    # List Traversed Edges
    for u, v, edge_data in sorted(subgraph.edges(data=True), key=lambda row: (str(row[0]), str(row[1]))):
        u_name = subgraph.nodes[u].get("qualified_name", subgraph.nodes[u].get("name", str(u)))
        v_name = subgraph.nodes[v].get("qualified_name", subgraph.nodes[v].get("name", str(v)))
        relation = edge_data.get(
            "relation", edge_data.get("relationship", "connected_to")
        )
        doc_lines.append(f"  - ({u_name}) --[{relation}]--> ({v_name})")
        if edge_data.get('source_statement'):
            doc_lines.append('    Literal assertion: ' + edge_data['source_statement'] + _provenance(edge_data))
        if edge_data.get('endpoint_alias'):
            doc_lines.append(f"    Endpoint alias: {edge_data['endpoint_alias']} ::> {edge_data['endpoint_path']}")

    return "\n".join(doc_lines)


def stringify_subgraph_markdown_tree(subgraph: nx.DiGraph) -> str:
    """Serialize directed relationships, including cross-links and cyclic components.

    Revisited targets are explicit references. Size and token savings must be
    measured for each context rather than assumed from the format.
    """
    if subgraph.number_of_edges() == 0 and subgraph.number_of_nodes() == 0:
        return "No matching multi-hop graph paths found."

    # Identify root nodes (nodes with 0 in-degree or root definitions)
    in_degrees = dict(subgraph.in_degree())
    roots = sorted((node for node, deg in in_degrees.items() if deg == 0), key=str)

    # Fallback to first node if graph is cyclic or has no obvious root
    if not roots:
        roots = sorted(subgraph.nodes(), key=str)[:1]

    doc_lines = ["--- Extracted System Architecture Hierarchy ---"]
    visited_nodes: Set[str] = set()

    def build_tree(node_id: str, depth: int = 0):
        if node_id not in subgraph.nodes:
            return

        data = subgraph.nodes[node_id]
        name = data.get("qualified_name", data.get("name", str(node_id)))
        kind = data.get("type", data.get("kind", "Element"))

        indent = "  " * depth
        prefix = "└── " if depth > 0 else ""
        if node_id in visited_nodes:
            doc_lines.append(f"{indent}{prefix}Reference: [{kind}] {name} (id: {node_id})")
            return
        visited_nodes.add(node_id)
        doc_lines.append(f"{indent}{prefix}[{kind}] {name}")
        if _provenance(data):
            doc_lines.append(f"{indent}    Source:" + _provenance(data))
        for detail in data.get('evidence_details', []):
            doc_lines.append(f"{indent}    Evidence: " + detail)

        # Traverse outgoing directed edges to build tree branches
        for _, successor, edge_data in sorted(subgraph.out_edges(node_id, data=True), key=lambda row: str(row[1])):
            rel = edge_data.get(
                "relation", edge_data.get("relationship", "owns")
            )
            sub_indent = "  " * (depth + 1)
            doc_lines.append(f"{sub_indent}├── ({rel})")
            if edge_data.get('source_statement'):
                doc_lines.append(f"{sub_indent}    Literal assertion: " + edge_data['source_statement'] + _provenance(edge_data))
            if edge_data.get('endpoint_alias'):
                doc_lines.append(f"{sub_indent}    Endpoint alias: {edge_data['endpoint_alias']} ::> {edge_data['endpoint_path']}")
            build_tree(successor, depth + 2)

    # Build trees starting from identified roots
    for root in roots:
        build_tree(root)

    # Traverse disconnected components too; listing their nodes loses their edges.
    for node in sorted(subgraph.nodes(), key=str):
        if node not in visited_nodes:
            doc_lines.append("\nAdditional Component:")
            build_tree(node)

    return "\n".join(doc_lines)


class MultiHopGraphTraversal:
    """Helper utilities for basic keyword matching and n-hop graph exploration."""

    def __init__(self, graph: nx.DiGraph):
        self.graph = graph

    def find_nodes_by_keyword(self, keywords: List[str]) -> List[str]:
        """Finds node IDs whose 'name' or 'type' matches any keyword (case-insensitive)."""
        matching_nodes = []
        for node_id, data in self.graph.nodes(data=True):
            node_name = str(data.get("name", "")).lower()
            node_type = str(data.get("type", "")).lower()

            for kw in keywords:
                kw_lower = kw.lower()
                if kw_lower in node_name or kw_lower in node_type:
                    matching_nodes.append(node_id)
                    break
        return matching_nodes

    def get_nhop_subgraph(
        self, start_node_ids: List[str], n_hops: int = 3
    ) -> nx.DiGraph:
        """Performs a BFS starting from target anchor nodes up to N hops (bidirectional)."""
        visited_nodes: Set[str] = set(start_node_ids)
        current_frontier: Set[str] = set(start_node_ids)

        for _ in range(n_hops):
            next_frontier: Set[str] = set()
            for node in current_frontier:
                if node in self.graph:
                    neighbors = set(self.graph.successors(node)).union(
                        set(self.graph.predecessors(node))
                    )
                    next_frontier.update(neighbors - visited_nodes)
            visited_nodes.update(next_frontier)
            current_frontier = next_frontier

        return self.graph.subgraph(visited_nodes).copy()

    def find_explicit_paths(
        self,
        source_nodes: List[str],
        target_nodes: List[str],
        max_len: int = 4,
    ) -> List[List[str]]:
        """Finds all simple structural paths between source and target concepts up to max_len."""
        undirected_g = self.graph.to_undirected()
        paths = []

        for src in source_nodes:
            for tgt in target_nodes:
                if src in undirected_g and tgt in undirected_g and src != tgt:
                    try:
                        found = nx.all_simple_paths(
                            undirected_g, source=src, target=tgt, cutoff=max_len
                        )
                        paths.extend(list(found))
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        continue
        return paths


class SubgraphPruner:
    """Strategy C Subgraph Pruning Engine using Personalized PageRank (PPR) and Path-Stitching."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", embedder=None):
        self.embedder = embedder if embedder is not None else SentenceTransformer(model_name)

    def discover_interface_connectors(self, graph: nx.DiGraph, anchors: List[str]) -> Dict[str, Any]:
        """Find explicit named interfaces by incidence or scoped literal-path matching.

        Usage-prefix matches are retrieval candidates, never resolved-port links.
        Do not cross namespaces, follow type inheritance or guess ambiguous roots.
        """
        records = {}
        usages = {}
        for node, data in graph.nodes(data=True):
            if (data.get('declared') and data.get('kind') in {'part', 'port'}
                    and isinstance(data.get('scope'), str) and isinstance(data.get('name'), str)):
                usages.setdefault((data.get('scope'), data.get('name')), []).append(node)
            if (data.get('kind') != 'interface'
                    or data.get('connection_parse_status') != 'binary_endpoints_preserved'
                    or not isinstance(data.get('scope'), str)):
                continue
            roles = {'source_endpoint': [], 'target_endpoint': []}
            for _, target, edge in graph.out_edges(node, data=True):
                role = edge.get('relationship', edge.get('relation'))
                if role in roles:
                    roles[role].append((target, edge))
            if not all(len(edges) == 1 for edges in roles.values()):
                continue
            edges = [edges[0] for edges in roles.values()]
            if any(not isinstance(edge.get('endpoint_path'), str) or not edge['endpoint_path']
                   or graph.nodes[target].get('endpoint_path') != edge['endpoint_path']
                   or graph.nodes[target].get('scope') != data.get('scope')
                   for target, edge in edges):
                continue
            records[node] = edges
        matches, ambiguous = [], []
        for anchor in sorted(set(anchors)):
            if anchor not in graph:
                continue
            data = graph.nodes[anchor]
            is_usage = (data.get('declared') and data.get('kind') in {'part', 'port'}
                        and isinstance(data.get('scope'), str) and isinstance(data.get('name'), str))
            if is_usage and len(usages.get((data.get('scope'), data.get('name')), [])) != 1:
                ambiguous.append(anchor)
                continue
            for connector, endpoints in records.items():
                for target, edge in endpoints:
                    rule = None
                    if target == anchor:
                        rule = 'explicit_endpoint_role'
                    elif (is_usage and data.get('scope') == graph.nodes[connector].get('scope')
                          and (edge['endpoint_path'] == data['name']
                               or edge['endpoint_path'].startswith(data['name'] + '.'))):
                        rule = 'scoped_literal_usage_prefix'
                    if rule:
                        matches.append({'anchor': anchor, 'connector': connector,
                                        'endpoint_occurrence': target, 'rule': rule,
                                        'written_path': edge['endpoint_path'],
                                        'role': edge.get('relationship', edge.get('relation')),
                                        'endpoint_identity_status': 'syntactic_endpoint_only'})
        return {'version': 'scoped-literal-interface-discovery-v1',
                'connectors': sorted({match['connector'] for match in matches}),
                'matches': matches, 'ambiguous_usage_anchors': ambiguous,
                'semantic_port_resolution': False}

    def preserve_interface_endpoints(
        self, selected: nx.DiGraph, eligible_graph: nx.DiGraph
    ) -> nx.DiGraph:
        """Complete selected binary interfaces and verified written flow endpoints.

        Only explicit source/target role edges are restored. This neither resolves
        literal paths to port instances nor discovers interfaces absent from the
        selected graph. The caller controls the eligible source graph.
        """
        result = selected.copy()
        before_nodes, before_edges = len(result), result.number_of_edges()
        completed, incomplete = [], []
        completed_flows, incomplete_flows = [], []
        for node in list(selected.nodes()):
            selected_data = selected.nodes[node]
            flow = selected_data.get('written_flow_endpoints') is True
            if selected_data.get('kind') != 'interface' and not flow:
                continue
            completed_records, incomplete_records = (completed_flows, incomplete_flows) if flow else (completed, incomplete)
            if node not in eligible_graph:
                incomplete_records.append(node)
                continue
            evidence = eligible_graph.nodes[node]
            eligible_kind = evidence.get('written_flow_endpoints') is True if flow else evidence.get('kind') == 'interface'
            if not eligible_kind or evidence.get('connection_parse_status') != 'binary_endpoints_preserved':
                if selected_data.get('connection_parse_status') == 'binary_endpoints_preserved':
                    incomplete_records.append(node)
                continue
            roles = {'source_endpoint': [], 'target_endpoint': []}
            for _, target, data in eligible_graph.out_edges(node, data=True):
                role = data.get('relationship', data.get('relation'))
                if role in roles:
                    roles[role].append((target, data))
            if not all(len(edges) == 1 for edges in roles.values()):
                incomplete_records.append(node)
                continue
            # A role label alone does not certify a complete literal-path record.
            # Reject a damaged/stale eligible record instead of reporting success.
            if any(not isinstance(data.get('endpoint_path'), str)
                   or not data['endpoint_path']
                   or eligible_graph.nodes[target].get('endpoint_path') != data['endpoint_path']
                   for edges in roles.values() for target, data in edges):
                incomplete_records.append(node)
                continue
            for edges in roles.values():
                target, data = edges[0]
                result.add_node(target, **dict(eligible_graph.nodes[target]))
                result.add_edge(node, target, **dict(data))
            completed_records.append(node)
        result.graph['interface_endpoint_completion'] = {
            'completed_interfaces': completed, 'incomplete_interfaces': incomplete,
            'added_nodes': len(result) - before_nodes,
            'added_edges': result.number_of_edges() - before_edges,
        }
        if completed_flows or incomplete_flows:
            result.graph['interface_endpoint_completion'].update(completed_flows=completed_flows,
                                                                 incomplete_flows=incomplete_flows)
        return result

    def prune_with_ppr_decay(
        self,
        G: nx.DiGraph,
        anchor_nodes: List[str],
        alpha: float = 0.85,
        top_percentile: float = 0.35,
        min_ppr_score: float = 1e-4,
    ) -> nx.DiGraph:
        """Personalized PageRank (PPR) seeded from anchor nodes for dynamic decay.
        
        Default top_percentile = 0.35 established by empirical Pareto frontier sweep.
        """
        if G.number_of_nodes() == 0:
            return G.copy()

        valid_anchors = set(anchor_nodes).intersection(G.nodes())
        if not valid_anchors:
            return G.subgraph([]).copy()

        personalization = {node: 1.0 / len(valid_anchors) for node in valid_anchors}

        try:
            ppr_scores = nx.pagerank(
                G, alpha=alpha, personalization=personalization, weight="weight"
            )
        except Exception as e:
            logger.warning(f"PageRank computation failed ({e}); retaining seed anchors.")
            return G.subgraph(valid_anchors).copy()

        sorted_nodes = sorted(
            ppr_scores.items(), key=lambda item: item[1], reverse=True
        )
        cutoff_count = max(1, int(len(sorted_nodes) * top_percentile))

        retained_nodes = {
            node
            for idx, (node, score) in enumerate(sorted_nodes)
            if idx < cutoff_count or score >= min_ppr_score
        }
        retained_nodes.update(valid_anchors)

        return G.subgraph(retained_nodes).copy()

    def prune_with_path_stitching(
        self,
        G: nx.DiGraph,
        anchor_nodes: List[str],
        max_path_len: int = 4,
        original_graph: Optional[nx.DiGraph] = None,
    ) -> nx.DiGraph:
        """Relational Path Stitching: Connects pairs of grounded seed anchors via simple paths.
        
        Complete endpoint roles of surviving named interfaces after path selection.
        Supply the original eligible graph to restore roles removed by prior PPR;
        without it, completion is limited to evidence still present in G.
        """
        if G.number_of_nodes() == 0:
            return G.copy()

        valid_anchors = list(set(anchor_nodes).intersection(G.nodes()))
        
        # With fewer than two anchors, preserve PPR context and complete selected records.
        if len(valid_anchors) < 2:
            return self.preserve_interface_endpoints(
                G, original_graph if original_graph is not None else G)

        retained_nodes = set(valid_anchors)
        G_undirected = G.to_undirected()

        for i in range(len(valid_anchors)):
            for j in range(i + 1, len(valid_anchors)):
                src, tgt = valid_anchors[i], valid_anchors[j]
                if nx.has_path(G_undirected, src, tgt):
                    try:
                        paths = nx.all_simple_paths(
                            G_undirected,
                            source=src,
                            target=tgt,
                            cutoff=max_path_len,
                        )
                        for path in paths:
                            retained_nodes.update(path)
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        continue

        # Retain original nodes from G that lie on the stitched paths
        final_nodes = set(G.nodes()).intersection(retained_nodes)
        return self.preserve_interface_endpoints(
            G.subgraph(final_nodes), original_graph if original_graph is not None else G)

    def prune_with_hop_decay(
        self,
        G: nx.DiGraph,
        query: str = "",
        anchor_nodes: Any = None,
        base_threshold: float = 0.35,
        decay_factor: float = 0.85,
        decay_rate: float = 0.85,
        **kwargs: Any
    ) -> nx.DiGraph:
        """Backward-compatible delegation from legacy hop decay to SOTA PPR decay."""
        anchors = list(anchor_nodes) if anchor_nodes is not None else []
        alpha = decay_rate if decay_rate != 0.85 else decay_factor
        return self.prune_with_ppr_decay(
            G=G,
            anchor_nodes=anchors,
            alpha=alpha,
            top_percentile=base_threshold
        )
