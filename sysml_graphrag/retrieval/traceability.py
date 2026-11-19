"""Bounded retrieval over modeled relationships, independent of corpus names.

Ownership is traversed only between a requirement and its constraint. Review
paths are candidate review obligations, not predicted failures or test results.
"""
from collections import defaultdict, deque
import hashlib
import re
import networkx as nx

VERSION = 'typed-traceability-v1'
RELATIONS = {'depends_on', 'owns', 'verifies', 'declares_verification'}


def role(data):
    kind, cls = data.get('kind'), data.get('compiler_class')
    if kind in {'requirement', 'requirement def'} or cls in {'RequirementUsage', 'RequirementDefinition'}:
        return 'requirement'
    if kind in {'constraint', 'constraint def', 'ConstraintUsage', 'ConstraintDefinition'} or cls in {'ConstraintUsage', 'ConstraintDefinition'}:
        return 'constraint'
    if kind in {'verification', 'verification def', 'VerificationCaseUsage', 'VerificationCaseDefinition'} or cls in {'VerificationCaseUsage', 'VerificationCaseDefinition'}:
        return 'verification'
    if kind == 'verification_record':
        return 'verification_record'
    if kind in {'attribute', 'feature', 'calc', 'calc def'} or cls in {'AttributeUsage', 'CalculationUsage', 'CalculationDefinition'}:
        return 'parameter'
    return None


def recognize_traceability(query):
    # Namespaces and quoted element labels are data, not intent words. A
    # package called Verification must not turn a value lookup into a test query.
    text = re.sub(r"\b[A-Za-z_]\w*(?:::(?:[A-Za-z_]\w*|'[^']*'))+(?:\.[A-Za-z_]\w*)*", ' ', query)
    text = re.sub(r"`[^`]*`|'[^']*'|\"[^\"]*\"", ' ', text).lower()
    if re.search(r'\b(ports?|interfaces?|connections?|endpoints?)\b', text) and not re.search(r'\b(requirements?|constraints?|verification|dependencies)\b', text):
        return None
    if re.search(r'\b(changes?|changing|changed|impact|impacts|affected|depend(?:s|ent)? on)\b', text):
        return 'change_review'
    if re.search(r'\b(constraint|predicate)s?\b', text) and re.search(r'\b(parameters?|inputs?|referenced|references|depends?)\b', text):
        return 'constraint_inputs'
    if re.search(r'\b(verif(?:y|ies|ication|ications)|verification cases?|test cases?|validation cases?)\b', text):
        return 'verification'
    if (re.search(r'\b(?:which|what)\s+(?:(?:model(?:ed|led)\s+)?'
                  r'(?:behaviou?r|actions?|features?|elements?|parts?|systems?)\b[^?]*?)?'
                  r'\b(?:satisfy|satisfies|satisfying)\b', text)
            and not re.search(r'\b(?:not|never)\b', text)):
        return 'satisfaction_allocation' if re.search(r'\ballocat\w*\b', text) else 'satisfaction'
    return None


def normalized_label(text):
    text = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', text)
    return ' '.join(re.findall(r'[a-z0-9]+', text.lower()))


def resolve_seeds(graph, query, identity, scores, intent, max_seeds=8):
    allowed = {'change_review': {'parameter', 'requirement', 'constraint'},
               'constraint_inputs': {'requirement', 'constraint'},
               'verification': {'requirement', 'verification'},
               'satisfaction': {'requirement'},
               'satisfaction_allocation': {'requirement'}}[intent]
    candidates = {n for n, d in graph.nodes(data=True) if role(d) in allowed
                  and not d.get('boundary') and d.get('resolution_status') != 'syntactic_endpoint_only'
                  and (d.get('declared') or role(d) == 'constraint')}
    explicit = {n for mention in identity.get('mentions', []) if mention['status'] == 'matched'
                for n in mention['node_ids'] if n in candidates
                and graph.nodes[n].get('qualified_name') != identity.get('explicit_scope')}
    if explicit:
        chosen, method = sorted(explicit), 'exact_projected_identity'
    else:
        scope = identity.get('explicit_scope')
        if scope:
            candidates = {n for n in candidates if graph.nodes[n].get('qualified_name', '').startswith(scope+'::')}
        # Exact multiword declaration labels are lexical identities, not aliases
        # inferred from endpoints. Keep repeated names ambiguous across scopes.
        text = ' '+normalized_label(query)+' '
        matches = []
        for n in candidates:
            label = normalized_label(graph.nodes[n].get('name', ''))
            if len(label.split()) < 2:
                continue
            for found in re.finditer(r'(?<!\w)'+re.escape(label)+r'(?!\w)', text):
                matches.append((found.start(), found.end(), label, n))
        matches = [m for m in matches if not any(a <= m[0] and m[1] <= b and (a, b) != m[:2]
                   for a, b, _, _ in matches)]
        if matches:
            grouped = defaultdict(set)
            for _, _, label, n in matches:
                grouped[label].add(n)
            ambiguous = sorted({n for nodes in grouped.values() if len(nodes) > 1 for n in nodes})
            if ambiguous:
                return [], {'status': 'ambiguous', 'method': 'exact_declared_label', 'candidates': ambiguous}
            chosen, method = sorted({m[3] for m in matches}), 'exact_declared_label'
        elif intent in {'satisfaction', 'satisfaction_allocation'}:
            chosen, method = [], 'exact_requirement_required'
        else:
            # Apply the existing similarity floor/delta within the semantic role
            # pool; an auxiliary occurrence cannot become the only seed here.
            best = max((scores.get(n, -1.) for n in candidates), default=-1.)
            cutoff = max(.35, best-.12)
            chosen = sorted((n for n in candidates if scores.get(n, -1.) >= cutoff),
                            key=lambda n: (-scores[n], str(n))) if best >= .35 else []
            method = 'dense_role_filtered'
    if len(chosen) > max_seeds:
        return [], {'status': 'too_many_candidates', 'method': method, 'candidates': chosen, 'max_seeds': max_seeds}
    return chosen, {'status': 'matched' if chosen else 'unverified', 'method': method, 'candidates': chosen}


def relation_index(graph):
    incoming, outgoing = defaultdict(list), defaultdict(list)
    for u, v, data in graph.edges(data=True):
        if data.get('resolution_status') in {'unresolved', 'unsupported_reference_structure', 'unsupported_target_structure'}:
            continue
        relation = data.get('relationship')
        if relation in RELATIONS:
            item = (u, v, relation, ((u, v),))
            outgoing[u].append(item); incoming[v].append(item)
        elif relation == 'relation_record':
            represented = graph.nodes[v].get('represented_relationship')
            if represented in RELATIONS:
                for _, target, edge in graph.out_edges(v, data=True):
                    if edge.get('relationship') == 'relation_target':
                        if edge.get('resolution_status') in {'unresolved', 'unsupported_reference_structure', 'unsupported_target_structure'}:
                            continue
                        item = (u, target, represented, ((u, v), (v, target)))
                        outgoing[u].append(item); incoming[target].append(item)
    return incoming, outgoing


def retrieve_traceability(graph, query, identity, scores, intent, *, max_hops=4, max_paths=32, max_expansions=256):
    if any(type(value) is not int or value <= 0 for value in [max_hops, max_paths, max_expansions]):
        raise ValueError('Traceability search limits must be positive integers.')
    seeds, grounding = resolve_seeds(graph, query, identity, scores, intent)
    metadata = {'version': VERSION, 'intent': intent, 'grounding': grounding, 'seeds': seeds,
                'max_hops': max_hops, 'max_paths': max_paths, 'max_expansions': max_expansions,
                'semantics': 'Modeled declaration references and candidate review paths; no execution or predicted failures'}
    empty = graph.subgraph([]).copy()
    if not seeds:
        return empty, [], {**metadata, 'status': grounding['status']}
    if intent in {'satisfaction', 'satisfaction_allocation'}:
        from sysml_graphrag.retrieval.satisfaction import retrieve_satisfaction
        selected, records, result = retrieve_satisfaction(graph, seeds, intent, max_paths)
        return selected, records, {**metadata, **result,
                                   'retained_paths': len(records),
                                   'semantics': 'Written positive satisfy assertions and scoped allocation witnesses.'}
    incoming, outgoing = relation_index(graph)

    def next_steps(node, seed):
        current, initial = role(graph.nodes[node]), role(graph.nodes[seed])
        result = []
        if intent == 'change_review':
            result += [(u, witness) for u, _, rel, witness in incoming[node] if rel == 'depends_on']
            if current == 'constraint':
                result += [(u, witness) for u, _, rel, witness in incoming[node]
                           if rel == 'owns' and role(graph.nodes[u]) == 'requirement']
            if current == 'requirement':
                result += verification_predecessors(node)
            if current == 'verification_record':
                result += [(u, witness) for u, _, rel, witness in incoming[node] if rel == 'declares_verification']
        elif intent == 'constraint_inputs':
            if current == 'requirement':
                result += [(v, witness) for _, v, rel, witness in outgoing[node]
                           if rel == 'owns' and role(graph.nodes[v]) == 'constraint']
            elif current == 'constraint':
                result += [(v, witness) for _, v, rel, witness in outgoing[node] if rel == 'depends_on']
        elif initial == 'requirement':
            if current == 'requirement':
                result += verification_predecessors(node)
            elif current == 'verification_record':
                result += [(u, witness) for u, _, rel, witness in incoming[node] if rel == 'declares_verification']
        elif initial == 'verification':
            if current == 'verification':
                direct = [(v, witness) for _, v, rel, witness in outgoing[node] if rel == 'verifies' and role(graph.nodes[v]) == 'requirement']
                result += direct or [(v, witness) for _, v, rel, witness in outgoing[node] if rel == 'declares_verification']
            elif current == 'verification_record':
                result += [(v, witness) for _, v, rel, witness in outgoing[node] if rel == 'verifies' and role(graph.nodes[v]) == 'requirement']
        return sorted(result, key=lambda r: (str(r[0]), r[1]))

    def verification_predecessors(node):
        direct = [(u, witness) for u, _, rel, witness in incoming[node] if rel == 'verifies' and role(graph.nodes[u]) == 'verification']
        return direct or [(u, witness) for u, _, rel, witness in incoming[node] if rel == 'verifies' and role(graph.nodes[u]) == 'verification_record']

    pending = deque((seed, (seed,), (), 0) for seed in seeds)
    candidates, expansions, cutoff = [], 0, False
    while pending and expansions < max_expansions:
        seed, nodes, edges, hops = pending.popleft()
        expansions += 1
        node = nodes[-1]
        options = [(target, witness) for target, witness in next_steps(node, seed) if target not in nodes]
        terminal = ((intent == 'change_review' and role(graph.nodes[node]) == 'verification')
                    or (intent == 'verification' and hops and role(graph.nodes[node]) in {'verification', 'requirement'})
                    or (intent == 'constraint_inputs' and hops and role(graph.nodes[node]) not in {'requirement', 'constraint'}))
        if terminal or not options or hops >= max_hops:
            if edges:
                candidates.append((seed, nodes, edges, hops))
            cutoff |= bool(options and hops >= max_hops and not terminal)
            continue
        for target, witness in options:
            pending.append((seed, nodes+(target,), edges+witness, hops+1))
    # Prefer paths reaching the requested semantic role over incidental leaves.
    order = {'verification': 0, 'requirement': 1, 'constraint': 2}
    candidates.sort(key=lambda p: (order.get(role(graph.nodes[p[1][-1]]), 3), -p[3], p[1]))
    records, unique = [], set()
    for seed, nodes, edges, hops in candidates:
        canonical = tuple(sorted(set(edges)))
        if canonical in unique:
            continue
        unique.add(canonical)
        key = hashlib.sha256(repr((intent, seed, canonical)).encode()).hexdigest()[:24]
        records.append({'id': key, 'kind': 'traceability_path', 'intent': intent, 'seed': seed,
                        'target': nodes[-1], 'logical_hops': hops, 'edges': [list(e) for e in canonical]})
    limited = bool(pending) or cutoff or len(records) > max_paths
    records = records[:max_paths]
    for record in records:
        record['search_limits_reached'] = limited
    edges = {tuple(e) for record in records for e in record['edges']}
    selected = graph.edge_subgraph(edges).copy() if edges else empty
    metadata.update(status='paths_found' if records else 'no_supported_paths', expansions=expansions,
                    limits_reached=limited, candidate_paths=len(unique), retained_paths=len(records))
    return selected, records, metadata
