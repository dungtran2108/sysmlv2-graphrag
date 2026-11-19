"""Bounded declared-part lookup through one explicitly selected usage's type.

Literal request recognition and directed graph evidence only; no instance,
inheritance or physical-composition inference. Dense anchors remain independent.
"""
import re
import hashlib

from sysml_graphrag.retrieval.query_identity import IDENTIFIER
from sysml_graphrag.retrieval.traceability import normalized_label

NAME = IDENTIFIER + r'(?:::' + IDENTIFIER + r')*'
USAGE = r'(?P<usage_quote>`?)(?P<usage>' + NAME + r')(?P=usage_quote)'
ROOT = r'(?P<root_quote>`?)(?P<root>' + NAME + r')(?P=root_quote)'
PATTERNS = [
    r'(?:Which|What) (?:stage )?parts are directly declared by the type of ' + USAGE + r' in ' + ROOT,
    r'What parts does the type of ' + USAGE + r' in ' + ROOT + r' directly declare',
    r'List the directly declared parts of ' + USAGE + r"(?:'s|’s) type in " + ROOT,
]
_MEMBERSHIP_HEAD = re.compile(
    r'(?:which|what(?: are)?|list|show(?: me)?) (?:the )?(?:declared )?'
    r'(?:parts|part usages)(?: are)?(?: listed| declared)?', re.IGNORECASE)
_DIRECT = re.compile(r'\bdirect(?:ly)?\b', re.IGNORECASE)
_SCOPE_SEPARATOR = re.compile(r'\s+(?:in|inside|of|under)\s+', re.IGNORECASE)
_COMPOUND_SCOPE = re.compile(r'\b(?:and|or|with|including|excluding|except)\b', re.IGNORECASE)
_INVENTORY_SCOPE = NAME + r'|[A-Za-z0-9_]+(?:[ -][A-Za-z0-9_]+)*'
_POSSESSIVE_SCOPE = re.compile(r"['\u2019]s\s+", re.IGNORECASE)


def recognize_scoped_inventory(query, *, allow_possessive=False):
    """Complete supported inventory requests; member names are query data."""
    patterns = [
        (r'In '+ROOT+r',? list the directly declared (?:parts|part usages|component usages) (?:with|and) their declared types', 'all_parts'),
        (r'List the directly declared (?:parts|part usages|component usages) (?:with|and) their declared types in '+ROOT, 'all_parts'),
        (r'In '+ROOT+r',? list the directly declared (?P<member>'+IDENTIFIER+r') and referenced (?:module|part|component) usages with their declared types', 'named_and_references'),
    ]
    for pattern, selection in patterns:
        match = re.fullmatch(pattern+r'[?.]?\s*', query.strip(), re.IGNORECASE)
        if match:
            return {'intent':'scoped_declared_inventory','root_reference':match['root'],
                    'selection':selection,'member_name':match.groupdict().get('member')}
    text = ' '.join(query.strip().rstrip('?.').split())
    if text.lower().startswith('in ') and ',' in text:
        scope, head = text[3:].split(',', 1)
    else:
        pair = _SCOPE_SEPARATOR.split(text, maxsplit=1)
        if len(pair) != 2:
            return None
        head, scope = pair
    if not _DIRECT.search(head) or not _MEMBERSHIP_HEAD.fullmatch(' '.join(_DIRECT.sub('', head).split())):
        return None
    scope = re.sub(r'^the\s+', '', scope.strip(), flags=re.IGNORECASE)
    if allow_possessive and not _COMPOUND_SCOPE.search(scope):
        segments = _POSSESSIVE_SCOPE.split(scope)
        if len(segments) == 2 and all(re.fullmatch(_INVENTORY_SCOPE, segment.strip('` '))
                                      for segment in segments):
            return {'intent': 'scoped_declared_inventory', 'root_reference': scope,
                    'selection': 'membership_only', 'possessive_scope': True}
    if scope.startswith('`') and scope.endswith('`'):
        scope = scope[1:-1]
    if (not re.fullmatch(_INVENTORY_SCOPE, scope)
            or _COMPOUND_SCOPE.search(scope)):
        return None
    return {'intent': 'scoped_declared_inventory', 'root_reference': scope,
            'selection': 'membership_only'}


def resolve_inventory_identity(graph, request):
    reference=request['root_reference']
    roots=sorted(n for n,d in graph.nodes(data=True) if d.get('declared') and d.get('kind')=='part def'
                 and (d.get('qualified_name')==reference or d.get('qualified_name','').endswith('::'+reference)))
    if request['selection'] == 'membership_only' and ('::' not in reference or not roots):
        target = tuple(normalized_label(segment) for segment in reference.split('::'))
        roots = sorted(n for n, d in graph.nodes(data=True)
                       if d.get('declared') and d.get('kind') == 'part def' and d.get('qualified_name')
                       and tuple(normalized_label(segment) for segment in
                                 d['qualified_name'].split('::')[-len(target):]) == target)
    status='matched' if len(roots)==1 else 'ambiguous' if roots else 'unverified'
    return {'version':'scoped-inventory-identity-v1','status':status,
            'explicit_scope':graph.nodes[roots[0]]['qualified_name'] if len(roots)==1 else None,
            'root_id':roots[0] if len(roots)==1 else None,
            'mentions':[{'mention':reference,'status':status,'node_ids':roots,
                         'candidates':[{'kind':graph.nodes[n]['kind'],'qualified_name':graph.nodes[n]['qualified_name']} for n in roots]}],
            'coverage':'Exact declared definition scope; direct projected part usages only.'}


def retrieve_scoped_inventory(graph, request, identity, *, max_members=64):
    if type(max_members) is not int or max_members<=0:
        raise ValueError('Inventory member limit must be a positive integer.')
    empty=graph.subgraph([]).copy()
    meta={'version':'scoped-inventory-v1','status':'unverified','root_id':identity.get('root_id'),
          'request':request,'max_members':max_members,'limits_reached':False,
          'unresolved_types':[],'semantics':'Direct projected declarations and declared types; no inherited or instance closure.'}
    def stop(reason):
        return empty,[],{**meta,'reason':reason}
    if identity['status']!='matched':
        return stop('The requested scope is not one verified declared definition.')
    root=identity['root_id']
    membership_only = request['selection'] == 'membership_only'
    if membership_only and any(e.get('relationship') == 'owns' and graph.nodes[v].get('kind') == 'part'
                               and e.get('resolution_status') in {'unresolved', 'ambiguous'}
                               for _, v, e in graph.out_edges(root, data=True)):
        return stop('Some direct part ownership in the graph projection is unresolved.')
    members=sorted(v for _,v,e in graph.out_edges(root,data=True)
                   if e.get('relationship')=='owns' and graph.nodes[v].get('declared') and graph.nodes[v].get('kind')=='part'
                   and e.get('resolution_status') not in {'unresolved','ambiguous'})
    if request['selection']=='named_and_references':
        named=[n for n in members if graph.nodes[n].get('name')==request['member_name']]
        if len(named)!=1:
            return stop('The named direct member is missing or ambiguous.')
        if any(type(graph.nodes[n].get('is_reference')) is not bool for n in members if n not in named):
            return stop('Reference/composite membership is not represented for every direct part.')
        members=sorted(set(named)|{n for n in members if graph.nodes[n].get('is_reference')})
    if not members:
        return stop('No matching direct declarations are represented; source completeness is unverified.')
    names=[graph.nodes[n].get('qualified_name') for n in members]
    if len(set(names))!=len(names):
        return stop('Duplicate direct declaration identities prevent an unambiguous inventory.')
    meta['represented_member_count']=len(members)
    meta['limits_reached']=len(members)>max_members
    if membership_only:
        if meta['limits_reached']:
            return stop('The direct part inventory exceeds the graph member limit.')
        edges = [(root, usage) for usage in members]
        for _, usage in edges:
            witness = graph[root][usage]
            if (not witness.get('file') or type(witness.get('line')) is not int
                    or witness['line'] < 1 or not graph.nodes[usage].get('name')):
                return stop('Some direct part declarations lack a name or source location in the graph projection.')
        meta.update(status='complete_projected_inventory', member_ids=members,
                    semantics='Direct projected part declarations; no inherited or instance closure.',
                    reason='Preserved all cited direct ownership records in the current graph projection.')
        record = {'kind': 'direct_inventory', 'id': f'direct-parts:{root}', 'root': root,
                  'edges': [list(edge) for edge in edges]}
        return graph.edge_subgraph(edges).copy(), [record], meta
    members=members[:max_members]
    records=[]
    for usage in members:
        edges=[(root,usage)]
        types=[v for _,v,e in graph.out_edges(usage,data=True) if e.get('relationship')=='typed_by']
        resolved=(len(types)==1 and graph.nodes[types[0]].get('declared')
                  and graph.nodes[types[0]].get('kind')=='part def'
                  and graph[usage][types[0]].get('resolution_status') not in {'unresolved','ambiguous'}
                  and sum(bool(d.get('declared')) and d.get('kind')=='part def' and d.get('qualified_name')==graph.nodes[types[0]].get('qualified_name')
                          for _,d in graph.nodes(data=True))==1)
        if resolved:
            edges.append((usage,types[0]))
        else:
            meta['unresolved_types'].append(usage)
        key=hashlib.sha256(repr((root,usage)).encode()).hexdigest()[:24]
        records.append({'kind':'inventory_member','id':key,'root':root,'usage':usage,
                        'edges':[list(e) for e in edges],'type_resolved':resolved,
                        'search_limits_reached':meta['limits_reached']})
    selected=graph.edge_subgraph({tuple(e) for r in records for e in r['edges']}).copy()
    meta.update(status='partial_projected_inventory' if meta['unresolved_types'] or meta['limits_reached'] else 'complete_projected_inventory',
                member_ids=members,reason='Preserved direct ownership and available unambiguous declared typing witnesses.')
    return selected,records,meta


def recognize_typed_composition(query):
    """Recognize only complete supported requests; never drop trailing constraints."""
    for pattern in PATTERNS:
        match = re.fullmatch(pattern + r'[?.]?\s*', query.strip(), flags=re.IGNORECASE)
        if match:
            return {'intent': 'typed_part_composition', 'root_reference': match['root'],
                    'usage_reference': match['usage']}
    return None


def resolve_composition_identity(graph, request):
    """Resolve the request's root and its direct member as one scoped identity.

    A recognized request explicitly names both roles, even when a name uses no
    internal capitals. Names in other namespaces or nested owners do not compete
    with a uniquely identified directly owned member. Duplicate direct declarations
    and ambiguous roots remain ambiguous.
    """
    def named(node, reference):
        name = graph.nodes[node].get('qualified_name', '')
        return name == reference or name.endswith('::' + reference)

    def mention(reference, nodes):
        candidates = sorted({(graph.nodes[node]['kind'], graph.nodes[node]['qualified_name']) for node in nodes})
        return {'mention': reference, 'status': 'matched' if len(nodes) == 1 else 'ambiguous' if nodes else 'unverified',
                'candidates': [{'kind': kind, 'qualified_name': name} for kind, name in candidates],
                'node_ids': sorted(nodes)}

    roots = [node for node, data in graph.nodes(data=True)
             if data.get('declared') and data.get('kind') == 'part def'
             and named(node, request['root_reference'])]
    root_mention = mention(request['root_reference'], roots)
    identity = {'version': 'direct-part-composition-identity-v1', 'status': root_mention['status'],
                'explicit_scope': None, 'root_id': None, 'usage_id': None, 'mentions': [root_mention],
                'coverage': 'Recognized composition request: exact declared root and directly owned part only.'}
    if len(roots) != 1:
        identity['reason'] = 'The requested root does not identify one declared part definition.'
        return identity
    root = roots[0]
    identity.update(root_id=root, explicit_scope=graph.nodes[root]['qualified_name'])
    usages = [target for _, target, edge in graph.out_edges(root, data=True)
              if edge.get('relationship') == 'owns' and graph.nodes[target].get('declared')
              and graph.nodes[target].get('kind') == 'part'
              and named(target, request['usage_reference'])]
    usage_mention = mention(request['usage_reference'], usages)
    identity['mentions'].append(usage_mention)
    identity['status'] = usage_mention['status']
    if len(usages) != 1:
        identity['reason'] = 'The root does not directly own one matching declared part usage.'
        return identity
    identity.update(usage_id=usages[0], reason='Verified one declared root and one directly owned matching part.')
    return identity


def retrieve_typed_composition(graph, request, *, identity=None):
    """Return a selected edge graph and a serializable record of the bounded path.

    Completion means the explicit projected chain was found, not compiler-level
    completeness. Zero represented children cannot certify source-level absence.
    """
    empty = graph.subgraph([]).copy()
    metadata = {'version': 'declared-typed-part-composition-v2', 'request': request,
                'relation_policy': ['owns', 'typed_by', 'owns'], 'max_hops': 3,
                'status': 'unverified', 'root_id': None, 'usage_id': None,
                'type_id': None, 'part_ids': [], 'evidence_edges': [],
                'semantics': 'Direct projected declarations only; no inherited or instance closure.'}

    def stop(reason):
        metadata['reason'] = reason
        return empty, metadata

    identity = resolve_composition_identity(graph, request) if identity is None else identity
    metadata.update(root_id=identity['root_id'], usage_id=identity['usage_id'])
    if identity['status'] != 'matched':
        return stop(identity['reason'])
    root, usage = identity['root_id'], identity['usage_id']
    types = [target for _, target, edge in graph.out_edges(usage, data=True)
             if edge.get('relationship') == 'typed_by']
    if len(types) != 1 or not graph.nodes[types[0]].get('declared') or graph.nodes[types[0]].get('kind') != 'part def':
        return stop('The selected usage lacks one verified declared part type.')
    definition = metadata['type_id'] = types[0]
    # Do not treat a damaged graph's link to one of several same-scope definitions
    # as uniquely resolved, even if it contains only one typed_by edge.
    same_types = [node for node, data in graph.nodes(data=True)
                  if data.get('declared') and data.get('kind') == 'part def'
                  and data.get('qualified_name') == graph.nodes[definition].get('qualified_name')]
    if len(same_types) != 1:
        return stop('The type has duplicate declared identities in the current projection.')
    parts = [target for _, target, edge in graph.out_edges(definition, data=True)
             if edge.get('relationship') == 'owns' and graph.nodes[target].get('declared')
             and graph.nodes[target].get('kind') == 'part']
    if not parts:
        return stop('No direct part declarations are represented for the selected type; source completeness is unverified.')
    names = [graph.nodes[part].get('qualified_name') for part in parts]
    if len(set(names)) != len(names):
        return stop('The selected type has duplicate direct part identities.')
    edges = [(root, usage), (usage, definition)] + [(definition, part) for part in sorted(parts)]
    selected = graph.edge_subgraph(edges).copy()
    metadata.update(status='complete_projected_chain', part_ids=sorted(parts),
                    evidence_edges=[{'source': source, 'target': target,
                                     'relationship': graph[source][target]['relationship']}
                                    for source, target in edges],
                    reason='Preserved the directed declaration/type chain and direct part declarations.')
    return selected, metadata


def composition_packing_edges(graph, query, identity):
    """Return the complete projected chain for a recognized, unambiguous request."""
    if identity['status'] in {'ambiguous', 'unverified'}:
        return []
    request = recognize_typed_composition(query)
    if request is None:
        match = re.fullmatch(
            r'What (?:design |declared )?type is used for (.+?),? and (?:which|what) '
            r'(?:stage )?parts does (?:that|the) (?:design|type) (?:directly )?declare[?.]?',
            query.strip(), re.IGNORECASE)
        if match is None:
            return []
        target = re.sub(r'^the\s+', '', match[1].strip('` '), flags=re.IGNORECASE)
        candidates = []
        for root, usage, edge in graph.edges(data=True):
            if edge.get('relationship') != 'owns':
                continue
            owner, member = graph.nodes[root], graph.nodes[usage]
            if (owner.get('kind') != 'part def' or member.get('kind') != 'part'
                    or not owner.get('declared') or not member.get('declared')):
                continue
            possessive = f"{owner.get('name', '')}'s {member.get('name', '')}"
            if (target == member.get('qualified_name')
                    or normalized_label(target) == normalized_label(possessive)):
                candidates.append({'root_reference': owner['qualified_name'],
                                   'usage_reference': member['qualified_name']})
        if len(candidates) != 1:
            return []
        request = candidates[0]
    _, metadata = retrieve_typed_composition(graph, request)
    return [(edge['source'], edge['target']) for edge in metadata['evidence_edges']]
