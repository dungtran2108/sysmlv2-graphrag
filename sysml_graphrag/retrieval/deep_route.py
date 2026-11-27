"""Focus supported deep traversals and keep binary relationships whole."""
from itertools import chain
import re

from sysml_graphrag.evaluation.comparators import EvidenceIndex, graph_units
from sysml_graphrag.io import graph_document
from sysml_graphrag.retrieval.composition import retrieve_typed_composition
from sysml_graphrag.retrieval.satisfaction import allocation_endpoints, allocation_witnesses, targets
from sysml_graphrag.retrieval.traceability import normalized_label
from sysml_graphrag.retrieval.traversal import stringify_subgraph_triples


def composition_edges(graph, query):
    match = re.fullmatch(
        r"What (?:declared )?type is the ([\w:-]+) part in ([\w:-]+)\? "
        r"What (?:[\w -]+ )?parts does that type declare, and what are their types\?",
        query, re.IGNORECASE)
    if not match:
        return set()
    usage, root = match.groups()
    selected, metadata = retrieve_typed_composition(
        graph, {'root_reference': root, 'usage_reference': usage})
    if metadata['status'] != 'complete_projected_chain':
        return set()
    edges = set(selected.edges())
    for part in metadata['part_ids']:
        types = targets(graph, part, 'typed_by')
        if len(types) == 1:
            edges.add((part, types[0]))
    return edges


def satisfaction_edges(graph, query, identity):
    if not re.search(r"\bwhich (?:written )?action path satisfies\b", query, re.IGNORECASE):
        return set()
    if identity['status'] in {'ambiguous', 'unverified'}:
        return set()
    ask_allocation = bool(re.search(r'\ballocation target\b', query, re.I))
    text = ' ' + normalized_label(query) + ' '
    requirements = [n for n, d in graph.nodes(data=True)
                    if d.get('kind') == 'requirement' and d.get('declared')
                    and len(normalized_label(d.get('name', '')).split()) >= 2
                    and ' ' + normalized_label(d['name']) + ' ' in text]
    if len(requirements) != 1:
        return set()
    requirement = requirements[0]
    edges = set()
    for assertion in graph.predecessors(requirement):
        data = graph.nodes[assertion]
        if data.get('compiler_class') != 'SatisfyRequirementUsage' or data.get('is_negated') is not False:
            continue
        relation = graph[assertion][requirement]
        if relation['relationship'] not in {'satisfied_requirement', 'references_declaration'}:
            continue
        subjects = targets(graph, assertion, 'assertion_subject')
        references = targets(graph, assertion, 'assertion_requirement')
        if len(subjects) != 1 or len(references) != 1:
            continue
        subject, reference = subjects[0], references[0]
        if reference not in targets(graph, subject, 'satisfies'):
            continue
        if relation['relationship'] == 'references_declaration':
            records = [(assertion, record, requirement) for record in targets(graph, assertion, 'relation_record')
                       if graph.nodes[record].get('represented_relationship') == 'satisfied_requirement'
                       and requirement in targets(graph, record, 'relation_target')]
            if len(records) != 1:
                continue
            a, record, r = records[0]
            edges.update([(a, record), (record, r)])
        edges.update([(assertion, requirement), (assertion, subject), (assertion, reference), (subject, reference)])
        if ask_allocation:
            edges.update(chain.from_iterable(allocation_witnesses(graph, assertion, subject)))
    if edges and re.search(r'\bwhat type is declared for requirement\b', query, re.I):
        types = targets(graph, requirement, 'typed_by')
        if len(types) == 1:
            edges.add((requirement, types[0]))
    return edges


def wiring_edges(graph, query):
    match = re.fullmatch(
        r'How is the ([\w -]+) signal wired through ([\w:-]+)\? '
        r'Give the written source and target for the connection inside the harness '
        r'and the (?:\w+ )?connections at (?:robot|system|parent) level[?.]?', query, re.I)
    if not match:
        return set()
    signal, name = match.groups()
    roots = [n for n, d in graph.nodes(data=True) if d.get('declared')
             and d.get('kind') == 'part def'
             and (d.get('name') == name or d.get('qualified_name') == name)]
    if len(roots) != 1:
        return set()
    root = roots[0]
    label = normalized_label(signal)

    def connections(owner):
        for node in targets(graph, owner, 'owns'):
            if graph.nodes[node].get('kind') != 'connection':
                continue
            endpoints = allocation_endpoints(graph, node)
            if endpoints:
                paths = [graph.nodes[end].get('endpoint_path', '') for end in endpoints]
                yield node, endpoints, paths

    internal_ports = set()
    edges = set()
    for node, endpoints, paths in connections(root):
        if all('.' not in path and (' ' + normalized_label(path)).endswith(' ' + label) for path in paths):
            internal_ports.update(paths)
            edges.update([(root, node), *((node, end) for end in endpoints)])
    if not internal_ports:
        return set()
    for usage, _, relation in graph.in_edges(root, data=True):
        member = graph.nodes[usage]
        if relation.get('relationship') != 'typed_by' or member.get('kind') != 'part' or not member.get('declared'):
            continue
        if targets(graph, usage, 'typed_by') != [root]:
            continue
        names = {f"{member['name']}.{port}" for port in internal_ports}
        for owner, _, ownership in graph.in_edges(usage, data=True):
            if ownership.get('relationship') != 'owns' or graph.nodes[owner].get('qualified_name') != member.get('scope'):
                continue
            siblings = [s for s in targets(graph, owner, 'owns')
                        if graph.nodes[s].get('declared') and graph.nodes[s].get('name') == member['name']]
            if siblings != [usage]:
                continue
            for node, endpoints, paths in connections(owner):
                if names.intersection(paths):
                    edges.update([(owner, usage), (usage, root), (owner, node), *((node, end) for end in endpoints)])
    return edges


def focus(graph, query, identity):
    for name, edges in (
        ('verified type traversal', composition_edges(graph, query)),
        ('verified satisfaction traversal', satisfaction_edges(graph, query, identity)),
        ('verified harness traversal', wiring_edges(graph, query)),
    ):
        if edges:
            return name, graph.edge_subgraph(edges).copy()
    return None, None


def relationship_groups(graph, selected):
    """Complete each surviving binary relation using the full projection."""
    groups = {}
    for node in selected:
        data = graph.nodes[node]
        if data.get('kind') not in {'connection', 'interface', 'AllocationUsage'} or not data.get('file'):
            continue
        endpoints = allocation_endpoints(graph, node)
        if not endpoints:
            continue
        owners = [u for u, _, e in graph.in_edges(node, data=True)
                  if e.get('relationship') == 'owns' and graph.nodes[u].get('qualified_name') == data.get('scope')]
        if len(owners) == 1:
            groups[node] = [(owners[0], node), *((node, end) for end in endpoints)]
    return groups


def complete_relationships(graph, selected):
    groups = relationship_groups(graph, selected)
    edges = set(selected.edges()).union(chain.from_iterable(groups.values()))
    expanded = graph.edge_subgraph(edges).copy()
    expanded.add_nodes_from((n, graph.nodes[n]) for n in selected if n not in expanded)
    return expanded, groups


def group_units(graph, units, groups):
    group_edges = set(chain.from_iterable(groups.values()))
    result = [u for u in units if not group_edges.intersection(map(tuple, u['edges']))]
    for node, edges in groups.items():
        result.append({'id': f'whole:{node}', 'edges': edges,
                       'text': stringify_subgraph_triples(graph.edge_subgraph(edges))})
    return result


def select_deep_evidence(graph, question, selection, encoder):
    route = selection['router_result']
    if not route['execute_llm'] or route['route'] != 'DEEP_GRAPHRAG':
        return selection
    branch, selected = focus(graph, question, route['metadata']['query_identity'])
    if selected is None:
        evidence = route['selected_evidence']
        selected = graph.edge_subgraph((edge['source'], edge['target']) for edge in evidence['edges']).copy()
        selected.add_nodes_from((node['id'], graph.nodes[node['id']])
                                for node in evidence['nodes'] if node['id'] not in selected)
    expanded, groups = complete_relationships(graph, selected)
    if branch is None and not groups:
        return selection
    evidence = graph_document(expanded)
    index = EvidenceIndex(group_units(expanded, graph_units(evidence), groups), encoder)
    return {**selection, 'ranked': index.rank(question), 'priority_edges': [],
            'ranking_embedding_diagnostics': index.embedding_diagnostics, 'retrieval_seconds': None,
            'router_result': {**route, 'selected_evidence': evidence,
                              'route': 'PROTOTYPE_FOCUSED' if branch else 'PROTOTYPE_GROUPED_FALLBACK',
                              'context_md': None}}
