"""Direct selection, frozen routing fallback and post-admission rendering."""
from collections import Counter
import hashlib
import re
from time import perf_counter

from sysml_graphrag import tree_audit
from sysml_graphrag.evaluation.budget import ChatTokenCounter, COMPOSITION_PACKING, messages_for
from sysml_graphrag.evaluation.comparators import EvidenceIndex, graph_units
from sysml_graphrag.evaluation.pilot_conditions import condition, rank_evidence
from sysml_graphrag.retrieval.composition import composition_packing_edges
from sysml_graphrag.retrieval.node_text import node_grounding_text, node_text_digest
from sysml_graphrag.retrieval.query_identity import check_query_identity
from sysml_graphrag.retrieval.routing import QueryRouter
from sysml_graphrag.retrieval.traversal import SubgraphPruner, stringify_subgraph_markdown_tree

METHOD, CAP = 'graph_routed', 4096
CONFIG = {
    'max_evidence_candidates': 100,
    'packing_policy': COMPOSITION_PACKING,
    'generator': {'context_window': 8192, 'output_reserve': 512},
}
ROLES = {'source_endpoint', 'target_endpoint'}
CONNECTOR_KINDS = {'connection', 'interface'}


def written_roles(graph, connector):
    node = graph.nodes[connector]
    # Anonymous written connections have no declared name. Their compiler
    # source span and directed endpoint witnesses establish their identity.
    if (node.get('kind') not in CONNECTOR_KINDS
            or node.get('connection_parse_status') != 'binary_endpoints_preserved'
            or not node.get('file') or node.get('line') is None):
        return []
    return [(a, b) for a, b, edge in graph.out_edges(connector, data=True)
            if edge['relationship'] in ROLES and edge.get('endpoint_path')
            and edge['endpoint_path'] == graph.nodes[b].get('endpoint_path')
            and graph.nodes[b].get('scope') == node.get('scope')]


def locality(question, graph):
    """Classify only from wording and the compiler projection."""
    identity = check_query_identity(graph, question)
    def result(reason, root=None, renderer='triples'):
        return dict(selection='direct_scoped' if root else 'frozen_routing',
                    reason=reason, root=root, renderer=renderer, query_identity=identity)

    if identity['status'] != 'matched':
        return result(f"Frozen identity guard returned {identity['status']}.")
    if re.search(r'\b(?:parent|sibling|inherited|across|following|allocat\w*|satisf\w*|requirement\w*)\b',
                 question, re.I):
        return result('Wording requests a cross-declaration traversal or modeled traceability.')
    roots = {node for mention in identity['mentions'] for node in mention['node_ids']
             if graph.nodes[node].get('declared') and graph.nodes[node]['kind'].endswith(' def')}
    explicit = identity['explicit_scope']
    if explicit:
        roots = {node for node in roots if graph.nodes[node]['qualified_name'] == explicit}
    if len(roots) != 1:
        return result('Wording and projected identities do not establish one declaration.')
    root, = roots
    children = {b for _, b, edge in graph.out_edges(root, data=True) if edge['relationship'] == 'owns'}
    connectors = {child: written_roles(graph, child) for child in children}
    connectors = {child: roles for child, roles in connectors.items() if roles}
    references = {node for mention in identity['mentions'] for node in mention['node_ids']} - {root}
    for target in references:
        # A dotted endpoint is local when its written connector is owned here.
        # A reference to a distinct declaration needs the frozen traversal.
        if target not in children and not any(b == target for roles in connectors.values() for _, b in roles):
            return result('An explicit reference lacks a verified ownership or written connector in the declaration.')
    asks_connector = bool(re.search(r'\b(?:connections?|connectors?|endpoints?|interface type)\b', question, re.I))
    asks_members = bool(re.search(r'\b(?:direct(?:ly)?|declared types?|part usages?|owned)\b', question, re.I))
    if asks_connector:
        eligible = set(connectors).intersection(references) if references else set(connectors)
        if references and not eligible:
            eligible = {node for node, roles in connectors.items() if any(b in references for _, b in roles)}
        verified = {node for node in eligible if len(connectors[node]) == 2
                    and {graph.edges[e]['relationship'] for e in connectors[node]} == ROLES}
        if not verified:
            return result('The projection does not verify locally owned connector roles for the request.')
        if re.search(r'\b(?:trace|through)\b', question, re.I):
            # Prove a local written chain by usage-prefix joins, without treating the
            # prefixes as resolved ports. Unclear chains defer to the frozen route.
            paths = {}
            for node in verified:
                roles = {graph.edges[a, b]['relationship']: graph.edges[a, b]['endpoint_path']
                         for a, b in connectors[node]}
                paths[node] = tuple(roles[role].split('.')[0].casefold()
                                    for role in ('source_endpoint', 'target_endpoint'))
            words = set(re.findall(r'[a-z0-9]+', question.casefold()))
            named_usages = {graph.nodes[node]['name'].casefold() for node in children
                            if graph.nodes[node].get('declared') and graph.nodes[node]['kind'] == 'part'}
            chain_usages = words.intersection(named_usages)
            chains = {(a, b) for a, (source, bridge) in paths.items()
                      for b, (join, target) in paths.items() if a != b and bridge == join
                      and len({source, bridge, target}) == 3 and {source, bridge, target} <= chain_usages}
            if len(chains) != 1 or not re.search(r'\btwo written connections\b', question, re.I):
                return result('Traversal wording does not establish one chain of written local connectors.')
            decision = result('Two written connectors form a unique local chain by usage prefix. '
                              'The multi-step trace uses triples.', root)
            decision['verified_connector_ids'] = list(next(iter(chains)))
            return decision
        decision = result('Projected ownership verifies local written connector roles.', root,
                          'triples' if asks_members else 'markdown_tree')
        if asks_members:
            decision['reason'] += ' The wording also requests a member inventory or declared types.'
        decision['verified_connector_ids'] = sorted(verified)
        return decision
    if asks_members and any(graph.nodes[node].get('declared') and graph.nodes[node]['kind'] == 'part'
                            for node in children):
        return result('Wording requests direct members or typing of one verified declaration.', root, 'markdown_tree')
    return result('Wording does not establish a local ownership, typing or ordinary connector request.')


def direct(question, decision, graph, pruner, encoder):
    start = perf_counter()
    root = decision['root']
    edges = set()
    for _, child, owns in graph.out_edges(root, data=True):
        if owns['relationship'] != 'owns':
            continue
        node = graph.nodes[child]
        edges.add((root, child))
        edges.update(written_roles(graph, child))
        if not node.get('declared') or (owns.get('file'), owns.get('line')) != (node.get('file'), node.get('line')):
            continue
        edges.update((a, b) for a, b, edge in graph.out_edges(child, data=True)
                     if edge['relationship'] == 'typed_by' and edge.get('file') and edge.get('line')
                     and (edge['file'], edge['line']) == (owns.get('file'), owns.get('line'))
                     and edge.get('resolution_status') not in {'unresolved', 'ambiguous'})
    selected = graph.edge_subgraph(sorted(edges)).copy()
    selected.add_node(root, **graph.nodes[root])
    selected = pruner.preserve_interface_endpoints(selected, graph)
    evidence = dict(records=[], nodes=[dict(id=n, **d) for n, d in selected.nodes(data=True)],
                    edges=[dict(source=a, target=b, **d) for a, b, d in selected.edges(data=True)])
    index = EvidenceIndex(graph_units(evidence), encoder)
    ranked = index.rank(question)
    return dict(ranked=ranked, router_result=dict(route='DIRECT_SCOPED', execute_llm=True,
                default_response=None, selected_evidence=evidence,
                metadata=dict(query_identity=decision['query_identity'], root=root,
                              interface_endpoint_completion=selected.graph.get('interface_endpoint_completion'))),
                priority_edges=composition_packing_edges(graph, question, decision['query_identity']),
                ranking_embedding_diagnostics=index.embedding_diagnostics, retrieval_seconds=perf_counter()-start)


def tree_prompt(row, graph, counter):
    """Render the frozen pack's exact records as one directed Markdown tree."""
    packed = row['packed']
    units = row['packed_units']
    edges = {tuple(e) for unit in units for e in unit['edges']}
    subgraph = graph.edge_subgraph(edges).copy()
    for unit in units:
        if not unit['edges']:
            assert unit['id'].startswith('node:'), unit['id']
            node = unit['id'].removeprefix('node:')
            subgraph.add_node(node, **graph.nodes[node])
        assert 'record' not in unit, unit['id']
    tree = stringify_subgraph_markdown_tree(subgraph) if subgraph else ''
    _, decoded, nodes = tree_audit.parse_tree(tree, subgraph)
    assert Counter((e['source'], e['role'], e['target']) for e in decoded) == Counter(
        (a, d.get('relation', d['relationship']), b) for a, b, d in subgraph.edges(data=True))
    locators = set(tree_audit.SOURCE.findall(tree))
    assert locators == set(tree_audit.SOURCE.findall(packed['evidence']))
    for item in decoded:
        edge = subgraph.edges[item['source'], item['target']]
        metadata = ' '.join(item['edge_metadata'].split())
        if statement := edge.get('source_statement'):
            assert ' '.join(statement.split()) in metadata
            assert tree_audit.locator(edge) in tree_audit.SOURCE.findall(metadata)
        if alias := edge.get('endpoint_alias'):
            assert f"Endpoint alias: {alias} ::> {edge['endpoint_path']}" in metadata
    row['packer_output'] = packed
    messages = messages_for(row['query'], tree)
    row['packed'] = dict(packed, evidence=tree, messages=messages,
                         evidence_tokens=counter.text(tree), input_tokens=counter.messages(messages))
    return dict(edge_positions=decoded, node_positions=nodes, locators=sorted(locators))


def fit(row, counter):
    packed = row['packed']
    routed = row['router_result']
    rendered = counter.tokenizer.apply_chat_template(packed['messages'], tokenize=False, add_generation_prompt=True)
    assert packed['messages'] == messages_for(row['query'], packed['evidence'])
    assert counter.text(rendered) == counter.messages(packed['messages']) == packed['input_tokens']
    assert counter.text(packed['evidence']) == packed['evidence_tokens'] <= CAP
    assert packed['input_tokens'] + packed['output_reserve'] <= packed['context_window']
    return dict(pass_fit=True, evidence_tokens=packed['evidence_tokens'],
                input_tokens_if_sent=packed['input_tokens'],
                input_tokens_to_send=packed['input_tokens'] if routed['execute_llm'] else 0,
                reserved_output=packed['output_reserve'], context=packed['context_window'],
                rendered_prompt_sha256=hashlib.sha256(rendered.encode('utf-8')).hexdigest(),
                fixed_response=None if routed['execute_llm'] else routed['default_response'])


class Candidate:
    """Prepare independent question conditions on one ordered projection."""

    def __init__(self, graph, encoder, tokenizer):
        self.graph = graph
        self.encoder = encoder
        self.counter = ChatTokenCounter(tokenizer)
        self.node_digest = node_text_digest(graph)
        self.vectors = None

    def prepare(self, question, *, query_id='query', model='projection', partition='unscored',
                query_family_id=None):
        graph = self.graph
        decision = locality(question, graph)
        pruner = SubgraphPruner(embedder=self.encoder)
        if decision['root']:
            selection = direct(question, decision, graph, pruner, self.encoder)
        else:
            if self.vectors is None:
                texts = [node_grounding_text(n, data) for n, data in graph.nodes(data=True)]
                self.vectors = self.encoder.encode(texts, normalize_embeddings=True, convert_to_tensor=True)
            router = QueryRouter(pruner=pruner, grounded_inventory_clarification=False)
            selection = rank_evidence(question, METHOD, graph, self.vectors, self.encoder, router, {},
                                      self.node_digest)
        query = dict(query=question, query_id=query_id, query_family_id=query_family_id or query_id,
                     model=model, partition=partition)
        row = condition(query, METHOD, selection, CAP, self.counter, CONFIG)
        identity = row['router_result']['metadata']['query_identity']
        if decision['query_identity'] != identity:
            raise ValueError('Frozen identity guard and candidate scope disagree.')
        row.update(arm='candidate', candidate_decision=decision, renderer=decision['renderer'])
        if row['renderer'] == 'markdown_tree':
            row['tree_equivalence'] = tree_prompt(row, graph, self.counter)
        row.update(prompt_fit=fit(row, self.counter),
                   generation_planned=row['router_result']['execute_llm'])
        return row
