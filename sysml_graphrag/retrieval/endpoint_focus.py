"""Explicit scoped endpoint targets for a bounded connection-review request.

Literal occurrences are traversal targets, not resolved individual-port identity.
Recognition covers complete request forms only; other questions keep their policy.
"""
import re

from sysml_graphrag.retrieval.query_identity import IDENTIFIER

PATH = IDENTIFIER + r'(?:\.' + IDENTIFIER + r')+'
QUOTED_PATH = r'(?:`' + PATH + r'`|' + PATH + r')'
TARGETS = r'(?P<targets>' + QUOTED_PATH + r'(?:\s+and\s+' + QUOTED_PATH + r')*)'
NAME = IDENTIFIER + r'(?:::' + IDENTIFIER + r')*'
SCOPE = r'(?P<quote>`?)(?P<scope>' + NAME + r')(?P=quote)'
PATTERNS = [
    r'If ' + TARGETS + r' changes in ' + SCOPE
    + r', which explicitly connected (?:hull )?endpoint paths and interfaces should be reviewed',
    r'In ' + SCOPE + r', changing the ' + TARGETS
    + r' contract should prompt review of which connected (?:hull )?ports and named interfaces',
    r'Which (?:named )?interfaces explicitly connect to ' + TARGETS + r' in ' + SCOPE,
]


def recognize_endpoint_focus(query):
    for pattern in PATTERNS:
        match = re.fullmatch(pattern + r'[?.]?\s*', query.strip(), flags=re.IGNORECASE)
        if match:
            targets = [value.strip('`') for value in re.split(r'\s+and\s+', match['targets'], flags=re.IGNORECASE)]
            return {'intent': 'explicit_endpoint_connections', 'scope_reference': match['scope'],
                    'endpoint_paths': list(dict.fromkeys(targets))}
    return None


def resolve_endpoint_focus(graph, request):
    """Require one declared scope and literal occurrences for every named path."""
    reference = request['scope_reference']
    scopes = [node for node, data in graph.nodes(data=True)
              if data.get('declared') and data.get('kind') in {'package', 'part def', 'part'}
              and (data.get('qualified_name') == reference or data.get('qualified_name', '').endswith('::' + reference))]
    candidates = sorted({(graph.nodes[node]['kind'], graph.nodes[node]['qualified_name']) for node in scopes})
    status = 'matched' if len(scopes) == 1 else 'ambiguous' if scopes else 'unverified'
    identity = {'version': 'scoped-literal-endpoint-identity-v1', 'status': status,
                'explicit_scope': None, 'endpoint_anchors': [],
                'mentions': [{'mention': reference, 'status': status, 'node_ids': sorted(scopes),
                              'candidates': [{'kind': kind, 'qualified_name': name} for kind, name in candidates]}],
                'coverage': 'Named paths in one declared scope; syntactic occurrences only, not individual ports.'}
    if len(scopes) != 1:
        return identity
    scope = identity['explicit_scope'] = graph.nodes[scopes[0]]['qualified_name']
    anchors = set()
    for path in request['endpoint_paths']:
        occurrences = sorted(node for node, data in graph.nodes(data=True)
                             if data.get('kind') == 'port_binding'
                             and data.get('resolution_status') == 'syntactic_endpoint_only'
                             and data.get('scope') == scope and data.get('endpoint_path') == path)
        anchors.update(occurrences)
        status = 'matched' if occurrences else 'unverified'
        identity['mentions'].append({'mention': path, 'status': status, 'node_ids': occurrences,
                                     'candidates': [{'kind': 'port_binding', 'qualified_name': scope + '::' + path}]
                                     if occurrences else []})
    identity['status'] = 'matched' if all(row['status'] == 'matched' for row in identity['mentions']) else 'unverified'
    identity['endpoint_anchors'] = sorted(anchors)
    return identity
