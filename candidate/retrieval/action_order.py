"""Scoped lookup of explicit, immediate action successions."""
import re

from candidate.retrieval.query_identity import IDENTIFIER
from candidate.retrieval.traceability import normalized_label


_ARGUMENT = r'(?:`' + IDENTIFIER + r'(?:::' + IDENTIFIER + r')*`|[\w :]+)'
_QUESTION = re.compile(
    r'(?:which|what) actions? (?:'
    r'(?:is|are) (?:explicitly )?(?:ordered )?(?:immediately|directly) (?P<direction>after|before)|'
    r'comes? (?:immediately )?(?P<comes>after|before)|(?P<follows>follows?)) '
    r'(?P<action>' + _ARGUMENT + r') in (?P<scope>' + _ARGUMENT + r')', re.I)
_COMPOUND = re.compile(r'\b(?:and|or|except|including|excluding|with|not|never)\b', re.I)


def recognize_action_order(query):
    text = ' '.join(query.strip().rstrip('?.').split())
    match = _QUESTION.fullmatch(text)
    if not match:
        return None
    action, scope = (re.sub(r'^the ', '', match[key], flags=re.I).strip('` ')
                     for key in ('action', 'scope'))
    if any(_COMPOUND.search(value) or not normalized_label(value) for value in (action, scope)):
        return None
    direction = match['direction'] or match['comes'] or 'after'
    return {'action': action, 'scope': scope, 'direction': direction.lower()}


def retrieve_action_order(graph, request, *, max_records=32):
    """Return explicitly adjacent actions in the uniquely matched scope."""
    empty = graph.subgraph([]).copy()
    meta = {'intent': 'action_order', 'request': request, 'seeds': [], 'status': 'no_supported_paths'}
    scope_name = request['scope']
    scopes = []
    for node, data in graph.nodes(data=True):
        if not data.get('declared') or data.get('kind') not in {'action def', 'action'}:
            continue
        name = data.get('qualified_name') or ''
        if ('::' in scope_name and (name == scope_name or name.endswith('::' + scope_name))
                or '::' not in scope_name and normalized_label(name.rsplit('::', 1)[-1]) == normalized_label(scope_name)):
            scopes.append(node)
    if len(scopes) > 1:
        return empty, [], {**meta, 'status': 'ambiguous', 'grounding': {'candidates': sorted(scopes)}}
    if not scopes:
        return empty, [], meta
    owner = scopes[0]
    children = [child for _, child, edge in graph.out_edges(owner, data=True) if edge.get('relationship') == 'owns']
    reference = request['action']
    words = [word for word in normalized_label(reference).split() if word not in {'the', 'a', 'an'}]
    names = {' '.join(words)}
    if len(words) > 1 and len(words[0]) > 4 and words[0].endswith('ing'):
        names.add(' '.join([words[0][:-3], *words[1:]]))
    actions = [child for child in children if graph.nodes[child].get('kind') == 'action'
               and graph.nodes[child].get('declared') and (graph.nodes[child].get('qualified_name') == reference
               if '::' in reference else normalized_label(graph.nodes[child].get('name') or '') in names)]
    if len(actions) > 1:
        return empty, [], {**meta, 'status': 'ambiguous', 'grounding': {'candidates': sorted(actions)}}
    if not actions:
        return empty, [], meta

    action = actions[0]
    role = 'earlier_action' if request['direction'] == 'after' else 'later_action'
    witnesses = []
    for node, _, edge in graph.in_edges(action, data=True):
        if node in children and graph.nodes[node].get('kind') == 'action_succession' and edge.get('relationship') == role:
            witnesses.append(node)
    if not witnesses or len(witnesses) > max_records:
        return empty, [], {**meta, 'limits_reached': len(witnesses) > max_records}
    witnesses.sort(key=lambda node: (graph.nodes[node]['file'], graph.nodes[node]['line'], node))
    edges = []
    roles = {'earlier_action', 'later_action'}
    for node in witnesses:
        ends = [(target, edge['relationship']) for _, target, edge in graph.out_edges(node, data=True)
                if edge.get('relationship') in roles]
        if len(ends) != 2 or {role for _, role in ends} != roles or any(target not in children for target, _ in ends):
            return empty, [], meta
        edges.extend((node, target) for target, _ in ends)
    record = {'id': 'action-order', 'kind': 'action_order', 'successions': witnesses,
              'edges': [list(edge) for edge in edges]}
    return graph.edge_subgraph(edges).copy(), [record], {**meta, 'status': 'paths_found', 'seeds': [action]}
