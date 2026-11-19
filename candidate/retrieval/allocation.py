"""Scoped lookup of written allocation statements in either direction."""
import re

from candidate.retrieval.query_identity import IDENTIFIER
from candidate.retrieval.satisfaction import REJECTED, allocation_endpoints
from candidate.retrieval.traceability import normalized_label


_PATH = IDENTIFIER + r'(?:\.' + IDENTIFIER + r')*'
_SCOPE = IDENTIFIER + r'(?:::' + IDENTIFIER + r')*'
_ROLE = r'(?:actions?|behaviou?rs?|parts?|components?|(?:firmware )?tasks?|elements?)'
_QUESTION = r'(?:which|what) (?:' + _ROLE + r' )?(?:is|are) '
_FORWARD = re.compile(_QUESTION + r'(?P<endpoint>.+?) (?:explicitly )?allocated to', re.I)
_REVERSE = re.compile(_QUESTION + r'(?:explicitly )?allocated to (?P<endpoint>.+)', re.I)
_COMPOUND = re.compile(r'\b(?:and|or|except|including|excluding|with|not|never)\b', re.I)


def recognize_allocation(query):
    """Recognize a complete single-endpoint allocation question."""
    text = ' '.join(query.strip().rstrip('?.').split())
    parts = re.split(r' in ', text, flags=re.I)
    scope = None
    if len(parts) > 1:
        text, qualifier = ' in '.join(parts[:-1]), parts[-1]
        # A model description frames the question on the already selected corpus.
        if not (re.fullmatch(r'the (?:[\w-]+ )*model', qualifier, re.I) and not _COMPOUND.search(qualifier)):
            scope = qualifier[1:-1] if qualifier.startswith('`') and qualifier.endswith('`') else qualifier
            if not re.fullmatch(_SCOPE, scope):
                return None
    for direction, pattern in (('forward', _FORWARD), ('reverse', _REVERSE)):
        match = pattern.fullmatch(text)
        if not match:
            continue
        endpoint = re.sub(r'^the ', '', match['endpoint'], flags=re.I)
        if endpoint.startswith('`') and endpoint.endswith('`'):
            endpoint = endpoint[1:-1]
        elif ' ' in endpoint:
            endpoint = re.sub(r' ' + _ROLE + r'$', '', endpoint, flags=re.I)
        if (not re.fullmatch(_PATH + r'|[A-Za-z_][\w]*(?:[ -][\w]+)*', endpoint)
                or _COMPOUND.search(endpoint)):
            return None
        return {'direction': direction, 'endpoint': endpoint, 'scope': scope}
    return None


def retrieve_allocation(graph, request, *, max_records=32):
    """Match written endpoints and keep all supported matches scoped and cited."""
    role = 'source_endpoint' if request['direction'] == 'forward' else 'target_endpoint'
    endpoint, scope = request['endpoint'], request['scope']
    exact_path = '.' in endpoint
    name = normalized_label(endpoint)
    matches, incomplete = [], False
    for node, data in graph.nodes(data=True):
        if data.get('compiler_class') != 'AllocationUsage' or not data.get('scope'):
            continue
        if scope and data['scope'] != scope and not data['scope'].endswith('::' + scope):
            continue
        ends_by_role = {'source_endpoint': [], 'target_endpoint': []}
        for _, target, edge in graph.out_edges(node, data=True):
            relationship = edge.get('relationship')
            if relationship in ends_by_role:
                ends_by_role[relationship].append(graph.nodes[target].get('endpoint_path') or '')
        queried_ends = ends_by_role[role]
        if not any(path == endpoint if exact_path else normalized_label(path.rsplit('.', 1)[-1])
                   == name for path in queried_ends):
            continue
        ends = allocation_endpoints(graph, node)
        if (ends is None or any(len(paths) != 1 or not re.fullmatch(_PATH, paths[0])
                                for paths in ends_by_role.values())
                or data.get('resolution_status') in REJECTED or not data.get('file')
                or type(data.get('line')) is not int or data['line'] < 1):
            incomplete = True
            continue
        matches.append((node, *ends))

    meta = {'intent': 'standalone_allocation', 'request': request, 'seeds': [],
            'status': 'no_supported_paths', 'limits_reached': len(matches) > max_records}
    if not matches or incomplete or meta['limits_reached']:
        return graph.subgraph([]).copy(), [], meta
    matches.sort(key=lambda row: (graph.nodes[row[0]]['file'], graph.nodes[row[0]]['line'], row[0]))
    edges = [(node, end) for node, source, target in matches for end in (source, target)]
    meta.update(status='paths_found', seeds=[node for node, _, _ in matches])
    record = {'id': 'standalone-allocation', 'kind': 'allocation_inventory',
              'allocations': [list(row) for row in matches], 'edges': [list(edge) for edge in edges]}
    return graph.edge_subgraph(edges).copy(), [record], meta
