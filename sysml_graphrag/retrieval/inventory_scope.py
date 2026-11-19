"""Resolve and clarify the declared owner of a direct-part inventory."""
import re

from sysml_graphrag.retrieval.traceability import normalized_label
from sysml_graphrag.retrieval.composition import resolve_inventory_identity

_POSSESSIVE = re.compile(r"['\u2019]s\s+", re.IGNORECASE)


def _names_match(data, name):
    qualified = data.get('qualified_name') or ''
    suffix = tuple(normalized_label(piece) for piece in qualified.split('::'))
    target = tuple(normalized_label(piece) for piece in name.split('::'))
    leaf = (normalized_label(data.get('name') or ''),)
    return all(target) and (target == leaf or target == suffix[-len(target):])


def _witness(graph, source, target):
    edge = graph[source][target]
    if edge.get('resolution_status') not in (None, 'compiler_resolved'):
        return None
    if not edge.get('file') or type(edge.get('line')) is not int or edge['line'] < 1:
        return None
    return {'source_id': source, 'target_id': target, 'relationship': edge['relationship'],
            'file': edge['file'], 'line': edge['line'], 'end_line': edge.get('end_line')}


def _resolve_scope(graph, name):
    identity = resolve_inventory_identity(graph, {'root_reference': name, 'selection': 'membership_only'})
    ids = identity['mentions'][0]['node_ids']
    candidates = sorted(
        ({'id': node, 'qualified_name': graph.nodes[node]['qualified_name']} for node in ids),
        key=lambda row: (row['qualified_name'], str(row['id'])))
    return {'status': identity['status'], 'scope_id': identity['root_id'], 'candidates': candidates}


def resolve_inventory_scope(graph, scope_text):
    """Resolve a bare definition or verify one direct owner → usage → type path."""
    parts = _POSSESSIVE.split(scope_text)
    if len(parts) == 1:
        return {**_resolve_scope(graph, scope_text), 'mode': 'definition_name'}
    result = {'mode': 'owner_and_member', 'status': 'unverified', 'scope_id': None,
              'candidates': [], 'reason': 'unsupported_possessive'}
    if len(parts) != 2 or not all(part.strip() for part in parts):
        return result
    owner, member = (part.strip("`' ") for part in parts)
    owner_identity = _resolve_scope(graph, owner)
    result.update(owner=owner_identity, owner_text=owner, member_text=member)
    if owner_identity['status'] != 'matched':
        return {**result, 'status': owner_identity['status'], 'reason': 'owner_' + owner_identity['status']}

    owner_id = owner_identity['scope_id']
    candidates = []
    incomplete = False
    for _, usage_id, ownership in graph.out_edges(owner_id, data=True):
        usage = graph.nodes[usage_id]
        if ownership.get('relationship') != 'owns' or usage.get('kind') != 'part' or not usage.get('declared'):
            continue
        types = [target for _, target, edge in graph.out_edges(usage_id, data=True)
                 if edge.get('relationship') == 'typed_by']
        if not (_names_match(usage, member) or any(_names_match(graph.nodes[t], member) for t in types)):
            continue
        if len(types) != 1:
            incomplete = True
            continue
        type_id = types[0]
        definition = graph.nodes[type_id]
        witnesses = [_witness(graph, owner_id, usage_id), _witness(graph, usage_id, type_id)]
        if (not definition.get('declared') or definition.get('kind') != 'part def'
                or not definition.get('qualified_name') or not all(witnesses)):
            incomplete = True
            continue
        candidates.append({'usage_id': usage_id, 'usage_name': usage['qualified_name'],
                           'type_id': type_id, 'type_name': definition['qualified_name'],
                           'witnesses': witnesses})
    result['candidates'] = sorted(candidates, key=lambda row: (row['usage_name'], str(row['usage_id'])))
    if len(candidates) > 1:
        return {**result, 'status': 'ambiguous', 'reason': 'member_ambiguous'}
    if incomplete or not candidates:
        return {**result, 'reason': 'member_relationship_unverified'}
    return {**result, 'status': 'matched', 'reason': 'verified_ownership_and_typing',
            'scope_id': candidates[0]['type_id']}


def _location(edge):
    start, end = edge['line'], edge['end_line']
    span = f'{start}-{end}' if isinstance(end, int) and end > start else str(start)
    return f"{edge['file']}:{span}"


def propose_inventory_clarification(graph, scope_text):
    """Suggest cited candidates while leaving the requested owner unresolved."""
    resolution = resolve_inventory_scope(graph, scope_text)
    if resolution.get('reason') != 'owner_unverified':
        return {'status': resolution['status'], 'resolution': resolution}
    member = _resolve_scope(graph, resolution['member_text'])
    candidates = []
    for target in member['candidates']:
        for usage, _, typing in graph.in_edges(target['id'], data=True):
            if typing.get('relationship') != 'typed_by':
                continue
            for owner, _, ownership in graph.in_edges(usage, data=True):
                if ownership.get('relationship') != 'owns':
                    continue
                owner_name = graph.nodes[owner].get('qualified_name')
                usage_name = graph.nodes[usage].get('qualified_name')
                if not owner_name or not usage_name:
                    continue
                verified = resolve_inventory_scope(graph, f"{owner_name}'s {usage_name}")
                if verified['scope_id'] != target['id']:
                    continue
                path = verified['candidates'][0]
                if path['usage_id'] != usage or verified['owner']['scope_id'] != owner:
                    continue
                candidates.append({**path, 'owner_id': owner, 'owner_name': owner_name})
    candidates.sort(key=lambda row: (row['owner_name'], row['usage_name']))
    subject = resolution['owner_text']

    def description(row):
        locations = ', '.join(dict.fromkeys(_location(edge) for edge in row['witnesses']))
        usage = row['usage_name'].rsplit('::', 1)[-1]
        return f"`{row['owner_name']}` declares `{usage} : {row['type_name']}` ({locations})"

    if len(candidates) == 1:
        row = candidates[0]
        response = (f'By "{subject}", do you mean `{row["owner_name"]}`? '
                    f'{description(row)}.')
    elif candidates:
        options = '\n'.join('- ' + description(row) for row in candidates)
        response = f'Which declaration matches the owner you mean by "{subject}"?\n{options}'
    else:
        return {'status': 'unverified', 'resolution': resolution,
                'response': f'Which model element do you mean by "{subject}"?'}
    return {'status': 'confirmation_required', 'resolution': resolution, 'scope_text': scope_text,
            'candidates': candidates, 'response': response}


def confirm_inventory_scope(graph, pending, *, accepted, usage_id=None):
    """Verify an explicit selection against the offered and current paths."""
    if pending['status'] != 'confirmation_required':
        return {'status': 'no_pending_confirmation', 'scope_id': None}
    if accepted is not True:
        return {'status': 'declined', 'scope_id': None}
    offered = pending['candidates']
    if usage_id is None:
        if len(offered) != 1:
            return {'status': 'selection_required', 'scope_id': None}
        usage_id = offered[0]['usage_id']
    match = [row for row in offered if row['usage_id'] == usage_id]
    if len(match) != 1:
        return {'status': 'selection_unverified', 'scope_id': None}
    current = propose_inventory_clarification(graph, pending['scope_text'])
    if current['status'] != 'confirmation_required' or current['candidates'] != offered:
        return {'status': 'candidate_changed', 'scope_id': None}
    return {'status': 'confirmed', 'scope_id': match[0]['type_id'], 'candidate': match[0]}
