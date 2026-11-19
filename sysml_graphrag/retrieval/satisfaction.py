"""Retrieve written satisfy assertions and compiler-witnessed allocation joins."""
import hashlib
import re

from sysml_graphrag.retrieval.query_identity import IDENTIFIER

REJECTED = {'unresolved', 'ambiguous', 'unsupported_reference_structure',
            'unsupported_target_structure'}


def targets(graph, node, relationship):
    return [v for _, v, edge in graph.out_edges(node, data=True)
            if edge.get('relationship') == relationship and edge.get('resolution_status') not in REJECTED]


def allocation_endpoints(graph, allocation):
    sources = targets(graph, allocation, 'source_endpoint')
    destinations = targets(graph, allocation, 'target_endpoint')
    if len(sources) == len(destinations) == 1:
        return sources[0], destinations[0]
    return None


def allocation_witnesses(graph, assertion, subject):
    # Keep the written path relative to its typed usage. The same action
    # declaration can be used by several systems and does not identify a scope.
    path = graph.nodes[subject].get('name', '').split('.')
    if len(path) < 2 or not all(re.fullmatch(IDENTIFIER, part) for part in path):
        return []
    owners = [u for u, _, edge in graph.in_edges(assertion, data=True)
              if edge.get('relationship') == 'owns'
              and graph.nodes[u].get('qualified_name') == graph.nodes[assertion].get('scope')]
    if len(owners) != 1:
        return []
    owner = owners[0]
    roots = [v for v in targets(graph, owner, 'owns') if graph.nodes[v].get('declared')
             and graph.nodes[v].get('name') == path[0]]
    features = targets(graph, assertion, 'satisfying_feature')
    if len(roots) != 1 or len(features) != 1:
        return []
    root, feature = roots[0], features[0]
    definitions = targets(graph, root, 'typed_by')
    if len(definitions) != 1 or not graph.nodes[definitions[0]].get('declared'):
        return []
    definition = definitions[0]
    feature_keys = set(graph.nodes[feature].get('compiler_witnesses', []))
    allocations = []
    for alloc in targets(graph, definition, 'owns'):
        if graph.nodes[alloc].get('compiler_class') != 'AllocationUsage':
            continue
        endpoints = allocation_endpoints(graph, alloc)
        if endpoints is None:
            continue
        source, destination = endpoints
        data = graph.nodes[source]
        chain = data.get('declaration_chain', [])
        if (data.get('endpoint_path') != '.'.join(path[1:])
                or [ref.get('name') for ref in chain] != path[1:]
                or any(ref.get('proxy') or not ref.get('source_key') for ref in chain)
                or chain[-1]['source_key'] not in feature_keys
                or not graph.nodes[destination].get('endpoint_path')):
            continue
        # The relative source path starts with a declaration directly owned
        # by the definition reached through the asserted system usage.
        if not any(chain[0]['source_key'] in graph.nodes[v].get('compiler_witnesses', [])
                   for v in targets(graph, definition, 'owns')):
            continue
        allocations.append([(owner, root), (root, definition), (definition, alloc),
                            (assertion, feature), (alloc, source), (alloc, destination)])
    return allocations


def retrieve_satisfaction(graph, seeds, intent, max_paths):
    empty = graph.subgraph([]).copy()
    records = []
    if len(seeds) != 1:
        return empty, records, {'status': 'ambiguous'}
    requirement = seeds[0]
    assertions = sorted(u for u, _, edge in graph.in_edges(requirement, data=True)
                        if edge.get('relationship') == 'satisfied_requirement'
                        and edge.get('resolution_status') not in REJECTED
                        and graph.nodes[u].get('compiler_class') == 'SatisfyRequirementUsage'
                        and graph.nodes[u].get('is_negated') is False)
    unmatched = []
    for assertion in assertions:
        subjects = targets(graph, assertion, 'assertion_subject')
        references = targets(graph, assertion, 'assertion_requirement')
        if len(subjects) != 1 or len(references) != 1:
            unmatched.append(assertion)
            continue
        subject, reference = subjects[0], references[0]
        if reference not in targets(graph, subject, 'satisfies'):
            unmatched.append(assertion)
            continue
        base = [(assertion, requirement), (assertion, subject), (assertion, reference), (subject, reference)]
        extensions = allocation_witnesses(graph, assertion, subject) if intent == 'satisfaction_allocation' else [[]]
        if not extensions:
            unmatched.append(assertion)
            continue
        if len(records) + len(extensions) > max_paths:
            return empty, [], {'status': 'no_supported_paths', 'limits_reached': True}
        for extension in extensions:
            edges = sorted(set(base + extension))
            key = hashlib.sha256(repr((intent, edges)).encode()).hexdigest()[:24]
            records.append({'id': key, 'kind': 'traceability_path', 'intent': intent,
                            'seed': requirement, 'edges': [list(edge) for edge in edges]})
    metadata = {'status': 'paths_found' if records else 'no_supported_paths',
                'unmatched_assertions': unmatched}
    if not records:
        return empty, records, metadata
    for record in records:
        record['other_assertions_unmatched'] = bool(unmatched)
    edges = {tuple(edge) for record in records for edge in record['edges']}
    return graph.edge_subgraph(edges).copy(), records, metadata
