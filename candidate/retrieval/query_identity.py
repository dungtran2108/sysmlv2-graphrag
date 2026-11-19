"""Conservative checks for explicit identifiers in the supported source projection.

This is a literal identity guard, not a natural-language entity linker or a SysML
name resolver. Matching endpoint occurrences does not resolve individual ports.
"""
import re

IDENTIFIER = r'[A-Za-z_][A-Za-z0-9_]*'
REFERENCE = IDENTIFIER + r'(?:(?:::|\.)' + IDENTIFIER + r')*'
ID_COMPONENT = r'[A-Za-z_][A-Za-z0-9_-]*'
TOKENS = re.compile(r'(?<![A-Za-z0-9_-])(' + ID_COMPONENT + r'(?:(?:::|\.)' + ID_COMPONENT + r')*)(?![A-Za-z0-9_-])')


def explicit_identifiers(query, known_requirement_ids=()):
    """Recognize qualified/dotted, backticked and internally capitalized names.

    Ordinary capitalized prose and lowercase natural-language terms are excluded.
    Quoted SysML names, aliases and other language syntax are outside this subset.
    """
    quoted = set(re.findall(r'`(' + REFERENCE + r')`', query))
    known = [(m.start(), m.end(), short_id) for short_id in set(known_requirement_ids)
             for m in re.finditer(r'(?<![A-Za-z0-9_.:-])'+re.escape(short_id)+r'(?![A-Za-z0-9_.:-])', query)]
    terms = []
    for match in TOKENS.finditer(query):
        term = match.group(1)
        if any(start <= match.start() and match.end() <= end for start, end, _ in known):
            continue
        if term in {'e.g', 'i.e'}:
            continue
        if term in {'SysML', 'SysMLv2', 'GraphRAG', 'v2'} and term not in quoted:
            continue
        if (term in quoted or '::' in term or '.' in term or ('-' in term and re.search(r'\d', term))
                or re.search(r'[a-z0-9_][A-Z]|[A-Za-z]_[A-Za-z]|[A-Za-z][0-9]', term)):
            terms.append((match.start(), term))
    ordered = sorted(terms + [(start, term) for start, _, term in known])
    return list(dict.fromkeys(term for _, term in ordered))


def check_query_identity(graph, query):
    """Return exact projected identities, preserving ambiguity and missing support.

    A unique explicit ``in <scope>`` narrows the other identifiers by containment
    namespace. Without that scope, qualified names match only on :: boundaries.
    Multiple references may intentionally name different scopes, so do not impose
    a common inferred scope. Repeated endpoint occurrences share a literal identity.
    """
    records = []
    for node, data in graph.nodes(data=True):
        if not data.get('qualified_name'):
            continue
        if data.get('declared') or data.get('resolution_status') == 'syntactic_endpoint_only':
            records.append({'node_id': node, 'qualified_name': data['qualified_name'],
                            'name': data.get('endpoint_path', data.get('name')),
                            'kind': data.get('kind', ''), 'scope': data.get('scope'),
                            'requirement_id': data.get('requirement_id'),
                            'declared': bool(data.get('declared'))})

    def matches(term, scope=None):
        found = [r for r in records if (r['qualified_name'] == term
                 or r['qualified_name'].endswith('::' + term)
                 or (r['requirement_id'] and (term == r['requirement_id'] or (
                     '::'+r['qualified_name'].rsplit('::', 1)[0]+'::'+r['requirement_id']).endswith('::'+term))))
                 and (scope is None or r['qualified_name'].startswith(scope + '::'))]
        # A package with the same leaf name is a namespace, not another element
        # definition. Qualified package references still remain available as scopes.
        if '::' not in term and '.' not in term and any(r['kind'] != 'package' for r in found):
            found = [r for r in found if r['kind'] != 'package']
        return found

    terms = explicit_identifiers(query, [r['requirement_id'] for r in records if r['requirement_id']])
    scopes = []
    for match in re.finditer(r'\bin\s+(' + REFERENCE + r')', query, re.IGNORECASE):
        if match.group(1) not in terms:
            continue
        candidates = matches(match.group(1))
        candidates = [r for r in candidates if r['kind'] == 'package' or r['kind'].endswith(' def')]
        if len(candidates) == 1:
            scopes.append((match.group(1), candidates[0]['qualified_name']))
    scope = scopes[0][1] if len(scopes) == 1 else None
    rows = []
    for term in terms:
        # Explicitly qualified references stand on their own, allowing comparisons
        # across scopes. Only unqualified names/paths use the explicit scope.
        active_scope = scope if scope and term != scopes[0][0] and '::' not in term else None
        found = matches(term, active_scope)
        identities = sorted({(r['kind'], r['qualified_name']) for r in found})
        duplicate = any(sum(r['declared'] and (r['kind'], r['qualified_name']) == identity
                            for r in found) > 1 for identity in identities)
        status = ('unverified' if not found else 'ambiguous'
                  if len(identities) > 1 or duplicate else 'matched')
        rows.append({'mention': term, 'status': status,
                     'candidates': [{'kind': kind, 'qualified_name': name} for kind, name in identities],
                     'node_ids': sorted(r['node_id'] for r in found)})
    status = ('unverified' if any(r['status'] == 'unverified' for r in rows)
              else 'ambiguous' if any(r['status'] == 'ambiguous' for r in rows)
              else 'matched' if rows else 'unchecked')
    return {'version': 'explicit-projection-identity-v2', 'status': status,
            'explicit_scope': scope, 'mentions': rows,
            'coverage': 'Supported explicit identifiers only; no semantic absence or instance-port resolution.'}
