"""Compact, source-traceable retrieval view of compiler-observed semantics.

Only explicit engineering declarations and useful evidence records are retained.
Memberships and expression plumbing become witnessed relations/properties.
The compiler export remains authoritative; this is a versioned, partial view.
"""
from collections import Counter
import hashlib
import re
import networkx as nx

KINDS = {'Package': 'package', 'PartDefinition': 'part def', 'PartUsage': 'part',
         'PortDefinition': 'port def', 'PortUsage': 'port', 'AttributeUsage': 'attribute',
         'RequirementUsage': 'requirement', 'RequirementDefinition': 'requirement def',
         'ActionDefinition': 'action def', 'ActionUsage': 'action',
         'AnalysisCaseDefinition': 'analysis def', 'AnalysisCaseUsage': 'analysis',
         'InterfaceUsage': 'interface', 'InterfaceDefinition': 'interface def'}

CONNECTORS = {'InterfaceUsage', 'ConnectionUsage', 'BindingConnectorAsUsage', 'FlowUsage', 'AllocationUsage'}
EXTRA_KINDS = {'ReferenceUsage': 'reference', 'Feature': 'feature', 'CalculationDefinition': 'calc def',
               'CalculationUsage': 'calc', 'ItemUsage': 'item', 'ItemDefinition': 'item def',
               'StateUsage': 'state', 'StateDefinition': 'state def', 'ConnectionUsage': 'connection',
               'VerificationCaseUsage': 'verification', 'VerificationCaseDefinition': 'verification def'}
LEGACY_PROJECTION_VERSION = 'compiler-retrieval-v5'
FLOW_PROJECTION_VERSION = 'compiler-retrieval-v6'
PROJECTION_VERSION = 'compiler-retrieval-v7'
_PATH_SEGMENT = r"(?:[A-Za-z_]\w*|'(?:[^'\\]|\\.)+')"
_FLOW_PATH = re.compile(_PATH_SEGMENT + r'(?:\s*\.\s*' + _PATH_SEGMENT + r')*')
_ACTION_SUCCESSION = re.compile(r'first\s+([A-Za-z_]\w*)\s+then\s+([A-Za-z_]\w*)\s*;')


def build_retrieval_projection(document, *, allow_invalid=False, version=PROJECTION_VERSION):
    if version not in {LEGACY_PROJECTION_VERSION, FLOW_PROJECTION_VERSION, PROJECTION_VERSION}:
        raise ValueError('Unsupported retrieval projection version.')
    if document.get('schema') != 'sysml-compiler-observer-v1':
        raise ValueError('Unsupported compiler export.')
    if not {'elements', 'validation', 'property_errors', 'library_resource_errors'}.issubset(document):
        raise ValueError('Compiler validation is required.')
    invalid = bool(any(i['severity'] == 'ERROR' for i in document['validation'])
                   or document['property_errors'] or document['library_resource_errors'])
    if invalid and not allow_invalid:
        raise ValueError('Invalid compiler model: explicit diagnostic-only opt-in required.')
    rows = {r['id']: r for r in document['elements']}
    if len(rows) != len(document['elements']) or len({r['source_key'] for r in rows.values()}) != len(rows):
        raise ValueError('Duplicate compiler identity.')
    graph = nx.DiGraph(projection_version=version, diagnostic_only=invalid,
                       grounding_text_version='requirement-content-v1',
                       raw_input_elements=len(rows), provenance='compiler source spans and semantic links',
                       semantics='Explicit source evidence and compiler declaration references; no instance closure or expression evaluation')
    ids = {key: row['source_key'] for key, row in rows.items()}
    retained = set()
    issues = []

    def refs(row, field):
        value = row.get('fields', {}).get(field, [])
        return value if isinstance(value, list) else [value] if isinstance(value, dict) else []

    def related(row, field='ownedRelationship'):
        return [rows[r['id']] for r in refs(row, field) if r['id'] in rows]

    def descend(row):
        pending, seen = [row], set()
        while pending:
            current = pending.pop()
            if current['id'] in seen:
                continue
            seen.add(current['id'])
            yield current
            pending.extend(reversed(related(current) + related(current, 'ownedRelatedElement')))

    def location(row):
        return {k: row[k] for k in ['file', 'line', 'end_line'] if k in row}

    def explicit_redefinitions(row):
        return [r for r in related(row) if r['class'] == 'Redefinition'
                and r.get('source_backed') and not r['fields'].get('isImplied')]

    def binding_target(binding):
        targets = refs(binding, 'referencedFeature') or refs(binding, 'general')
        chain, terminal = [], targets
        if len(targets) == 1 and targets[0]['id'] in rows:
            target = rows[targets[0]['id']]
            chain = [r for r in related(target) if r['class'] == 'FeatureChaining']
            if chain:
                terminal = refs(chain[-1], 'chainingFeature') if all(
                    len(refs(r, 'chainingFeature')) == 1 for r in chain) else []
            elif target['class'] == 'Feature' and not target['fields'].get('declaredName'):
                terminal = []
        return terminal, chain

    def add_relation(u, v, relation, witnesses, **properties):
        evidence = [r['source_key'] for r in witnesses]
        data = {'relationship': relation, 'compiler_witnesses': evidence,
                **location(witnesses[0]), **properties}
        if graph.has_edge(u, v):
            previous = graph[u][v]
            if previous['relationship'] == relation:
                previous['compiler_witnesses'] = sorted(set(previous['compiler_witnesses'] + evidence))
                for name, value in properties.items():
                    previous.setdefault(name, value)
                return previous
            # DiGraph consumers cannot store parallel labels: preserve the second
            # relation as a record instead of silently replacing the first one.
            key = 'relation:' + hashlib.sha256((u+'\0'+relation+'\0'+v).encode()).hexdigest()[:24]
            if graph.has_edge(u, key) and graph.has_edge(key, v):
                previous = graph[u][key]
                combined = sorted(set(previous['compiler_witnesses'] + evidence))
                previous['compiler_witnesses'] = combined
                graph[key][v]['compiler_witnesses'] = combined
                graph.nodes[key]['compiler_witnesses'] = combined
                for name, value in properties.items():
                    previous.setdefault(name, value)
                    graph[key][v].setdefault(name, value)
                return previous
            graph.add_node(key, name=relation, qualified_name=relation, kind='relationship_record',
                           declared=False, represented_relationship=relation, compiler_witnesses=evidence, **location(witnesses[0]))
            graph.add_edge(u, key, **{**data, 'relationship': 'relation_record'})
            graph.add_edge(key, v, **{**data, 'relationship': 'relation_target'})
            return graph[u][key]
        else:
            graph.add_edge(u, v, **data)
            return graph[u][v]

    def ensure(ref):
        key = ids.get(ref['id']) or ref.get('source_key') or 'run-local:' + ref['id']
        if key not in graph:
            row = rows.get(ref['id'])
            fields = row['fields'] if row else ref
            name = fields.get('declaredName') or ref.get('class', 'Reference')
            graph.add_node(key, name=name, qualified_name=fields.get('qualifiedName') or name,
                           kind='unresolved_reference' if ref.get('proxy') else 'external_reference',
                           compiler_class=ref.get('class'), declared=False, boundary=True,
                           resolution_status='unresolved' if ref.get('proxy') else 'compiler_declaration_reference',
                           boundary_scope='input_not_projected' if row else 'library_or_unresolved',
                           compiler_witnesses=[key], **(location(row) if row else {}))
        return key

    for key, row in rows.items():
        fields, cls = row['fields'], row['class']
        membership = refs(row, 'owningRelationship')
        is_endpoint = any(r.get('class') == 'EndFeatureMembership' for r in membership)
        is_engineering = (cls in KINDS or cls in EXTRA_KINDS or cls.endswith(('Usage', 'Definition'))
                          or cls in CONNECTORS or cls == 'Dependency')
        redefinitions = explicit_redefinitions(row)
        keep = row.get('source_backed') and not fields.get('isImplied') and not is_endpoint and (
            (fields.get('declaredName') or redefinitions) and is_engineering
            or cls in CONNECTORS or cls in {'SatisfyRequirementUsage', 'ConstraintUsage',
                                           'VerificationCaseUsage', 'VerificationCaseDefinition'})
        if not keep:
            continue
        name = fields.get('declaredName') or (fields.get('name') if redefinitions else None) or f"{cls}@{row['file']}:{row.get('line')}"
        qualified = fields.get('qualifiedName') or name
        kind = KINDS.get(cls, EXTRA_KINDS.get(cls, cls))
        owner_refs = refs(row, 'owner')
        owner_scope = rows.get(owner_refs[0]['id'], {}).get('fields', {}).get('qualifiedName', '') if owner_refs else ''
        graph.add_node(ids[key], name=name, qualified_name=qualified, kind=kind,
                       # Retrieval's declared flag means an explicit local declaration,
                       # including a written override, never an inherited expansion.
                       compiler_class=cls, declared=bool(fields.get('declaredName') or redefinitions),
                       has_declared_name=bool(fields.get('declaredName')),
                       is_explicit_redefinition=bool(redefinitions),
                       scope=qualified.rsplit('::', 1)[0] if '::' in qualified else owner_scope,
                       is_reference=fields.get('isReference'), is_composite=fields.get('isComposite'),
                       short_name=fields.get('declaredShortName'),
                       compiler_witnesses=[row['source_key']], source=row.get('source'),
                       evidence_details=([f"Part declaration flags: isReference={fields.get('isReference')}; isComposite={fields.get('isComposite')}."]
                                         if cls == 'PartUsage' else []), **location(row))
        retained.add(key)
        if redefinitions:
            graph.nodes[ids[key]]['evidence_details'].append(
                'Explicit feature redefinition (local occurrence; no inherited value evaluation).')
        if cls == 'ConstraintUsage':
            graph.nodes[ids[key]]['evidence_details'].append('Constraint declaration (not evaluated): '+(row.get('source') or '').strip())

    for key in sorted(retained):
        row, source = rows[key], ids[key]
        for owner in refs(row, 'owner'):
            if owner['id'] in retained:
                add_relation(ids[owner['id']], source, 'owns', [row], property_origin='compiler_computed_owner')
        for typed in refs(row, 'type'):
            add_relation(source, ensure(typed), 'typed_by', [row],
                         resolution_status='unresolved' if typed.get('proxy') else 'compiler_resolved',
                         property_origin='compiler_computed_type')
        for relationship in related(row):
            cls = relationship['class']
            if not relationship.get('source_backed') or relationship['fields'].get('isImplied'):
                continue
            role = {'Subclassification': 'specializes', 'Specialization': 'specializes',
                    'Redefinition': 'redefines', 'Subsetting': 'subsets', 'ReferenceSubsetting': 'references_declaration'}.get(cls)
            if role:
                for general in refs(relationship, 'general'):
                    add_relation(source, ensure(general), role, [relationship],
                                 resolution_status='unresolved' if general.get('proxy') else 'compiler_resolved',
                                 property_origin='explicit_compiler_relationship')

    def feature_value(row):
        return [rows[ref['id']] for rel in related(row) if rel['class'] == 'FeatureValue'
                for ref in refs(rel, 'value') if ref['id'] in rows]

    def operands(expression):
        return [value for rel in related(expression) if rel['class'] == 'ParameterMembership'
                for member in related(rel, 'ownedRelatedElement') for value in feature_value(member)]

    def expression_references(expression):
        """Collapse one chain as a whole; never substitute its root for its leaf."""
        if expression['class'] == 'FeatureChainExpression':
            chains = [r for r in descend(expression) if r['class'] == 'FeatureChaining']
            direct_members = [ref for r in related(expression) if r['class'] == 'Membership'
                              and r.get('source_backed') for ref in refs(r, 'memberElement')]
            terminal = refs(chains[-1], 'chainingFeature') if chains else direct_members
            if len(terminal) == 1:
                yield terminal[0], expression, 'declaration_through_written_path'
            else:
                issues.append({'element': ids[expression['id']], 'reason': 'expression chain terminal not uniquely represented'})
            return
        if expression['class'] == 'FeatureReferenceExpression':
            referents = refs(expression, 'referent')
            if not referents:
                issues.append({'element': ids[expression['id']], 'reason': 'expression reference has no compiler referent'})
            for ref in referents:
                yield ref, expression, 'compiler_declaration_reference'
            return
        # Treat units as unit references, not engineering input dependencies.
        if expression['class'] == 'OperatorExpression' and expression['fields'].get('operator') == '[':
            args = operands(expression)
            if len(args) == 2:
                yield from expression_references(args[0])
                return
        for rel in related(expression):
            for child in related(rel, 'ownedRelatedElement'):
                yield from expression_references(child)

    def typed_connection_ends(row, source, memberships):
        """Preserve written role bindings without inventing pairwise direction.

        Custom connection ends may be ordinary FeatureMemberships, unlike the
        EndFeatureMembership shape of the binary interface syntax. A chained
        reference binds to its terminal declaration, never just its root.
        """
        preserved, resolved = [], 0
        for membership in memberships:
            members = related(membership, 'ownedRelatedElement')
            if len(members) != 1:
                continue
            end = members[0]
            if not end.get('source_backed') or end['fields'].get('isImplied'):
                continue
            bindings = [r for r in related(end) if r['class'] == 'ReferenceSubsetting'
                        and r.get('source_backed') and not r['fields'].get('isImplied')]
            if len(bindings) != 1 or not (bindings[0].get('source') or '').strip():
                continue
            binding = bindings[0]
            path = binding['source'].strip()
            role = end['fields'].get('declaredName') or end['fields'].get('name')
            occurrence = ids[end['id']]
            if occurrence not in graph or graph.nodes[occurrence].get('boundary'):
                graph.add_node(occurrence, name=role or path,
                               qualified_name=graph.nodes[source]['qualified_name']+'::'+(role or path),
                               kind='connection_end', declared=False, boundary=False,
                               compiler_class=end['class'], compiler_witnesses=[end['source_key']],
                               source=end.get('source'), scope=graph.nodes[source]['qualified_name'],
                               evidence_details=[], **location(end))
            data = graph.nodes[occurrence]
            data.update(is_connection_end=True, endpoint_path=path, endpoint_alias=role,
                        binding_semantics='compiler_declaration_reference_through_written_path')
            data['evidence_details'].append('Written connection end: '+end['source'].strip())
            add_relation(source, occurrence, 'owns', [membership, end])
            terminal, chain = binding_target(binding)
            data['declaration_chain'] = [ref.get('source_key') or 'run-local:'+ref['id']
                                         for r in chain for ref in refs(r, 'chainingFeature')]
            status = 'unsupported_reference_structure'
            if len(terminal) == 1:
                target = terminal[0]
                status = 'unresolved' if target.get('proxy') or any(
                    ref.get('proxy') for r in chain for ref in refs(r, 'chainingFeature')) else 'compiler_declaration_reference'
                add_relation(occurrence, ensure(target), 'references_declaration', [binding, *chain],
                             written_reference=path, resolution_status=status,
                             binding_semantics='declaration_through_written_path' if chain else 'direct_declaration_reference')
                if status == 'compiler_declaration_reference':
                    resolved += 1
            if status != 'compiler_declaration_reference':
                issues.append({'element': occurrence, 'reason': 'typed connection end target not resolved',
                               'resolution_status': status, 'written_reference': path})
            data['endpoint_resolution_status'] = status
            preserved.append((role, path))
        complete = len(memberships) >= 2 and len(preserved) == len(memberships)
        graph.nodes[source].update(
            connection_parse_status='typed_end_bindings_preserved' if complete else 'unsupported_endpoint_structure',
            written_end_count=len(memberships), preserved_end_count=len(preserved),
            connection_reference_status='compiler_declaration_references' if complete and resolved == len(preserved)
                                        else 'partial_or_unresolved')
        graph.nodes[source]['evidence_details'].append(
            'Typed connection end bindings (written roles; no flow direction inferred): '+
            '; '.join((role or '<unnamed>')+' ::> '+path for role, path in preserved))
        graph.nodes[source]['evidence_details'].append('Connection type: '+', '.join(
            rows.get(ref['id'], {}).get('fields', {}).get('qualifiedName') or ref.get('declaredName') or ref['class']
            for ref in refs(row, 'type')))
        if not complete:
            issues.append({'element': source, 'reason': 'typed connector has incomplete written end bindings',
                           'written_ends': len(memberships), 'preserved_ends': len(preserved)})

    for key in sorted(retained):
        row, source = rows[key], ids[key]
        for expression in feature_value(row):
            text = (expression.get('source') or '').strip()
            if not text:
                issues.append({'element': source, 'reason': 'value expression has no source span'})
                continue
            record = 'value:' + ids[expression['id']]
            details = ['Value expression (unevaluated): '+text]
            bindings = [r for r in related(row) if r['class'] == 'FeatureValue'
                        and any(v['id'] == expression['id'] for v in refs(r, 'value'))]
            properties = {'expression': text, 'evaluation_status': 'not_evaluated',
                          'is_default': any(r['fields'].get('isDefault', False) for r in bindings),
                          'is_initial': any(r['fields'].get('isInitial', False) for r in bindings)}
            if properties['is_default'] or properties['is_initial']:
                qualifier = 'Default' if properties['is_default'] else 'Initial'
                details[0] = qualifier + ' value expression (unevaluated): ' + text
            args = operands(expression)
            literal = expression if expression['class'].startswith('Literal') else None
            unit = None
            if expression['fields'].get('operator') == '[' and len(args) == 2 and args[0]['class'].startswith('Literal'):
                literal = args[0]
                unit_refs = refs(args[1], 'referent')
                if args[1]['class'] == 'FeatureReferenceExpression' and len(unit_refs) == 1:
                    unit = (args[1].get('source') or '').strip()
                    properties.update(unit=unit, unit_resolution='unresolved' if unit_refs[0].get('proxy') else 'compiler_resolved')
            if literal is not None and 'value' in literal['fields']:
                properties['literal_value'] = literal['fields']['value']
            graph.add_node(record, name=text, qualified_name=graph.nodes[source]['qualified_name']+'::value = '+text,
                           kind='value_record', declared=False, evidence_details=details,
                           compiler_witnesses=[ids[expression['id']]], **properties, **location(expression))
            add_relation(source, record, 'has_value', [expression])
            graph.nodes[source]['evidence_details'].extend(details)
            if unit:
                add_relation(record, ensure(unit_refs[0]), 'uses_unit', [args[1]],
                             resolution_status=properties['unit_resolution'])
            for ref, witness, semantics in expression_references(expression):
                add_relation(source, ensure(ref), 'depends_on', [witness],
                             written_reference=(witness.get('source') or '').strip(), dependency_semantics=semantics,
                             resolution_status='unresolved' if ref.get('proxy') else 'compiler_resolved')

    for key in sorted(retained):
        row, source = rows[key], ids[key]
        if row['class'] != 'ConstraintUsage':
            continue
        results = [r for r in related(row) if r['class'] == 'ResultExpressionMembership'
                   and r.get('source_backed') and not r['fields'].get('isImplied')]
        expressions = [e for membership in results for e in related(membership, 'ownedRelatedElement')
                       if e.get('source_backed') and (e.get('source') or '').strip()]
        references = []
        issue_count = len(issues)
        for expression in expressions:
            for ref, witness, semantics in expression_references(expression):
                if not witness.get('source_backed') or not (witness.get('source') or '').strip():
                    continue
                target = ensure(ref)
                reference = {'target': target, 'written_reference': witness['source'].strip(),
                             'dependency_semantics': semantics,
                             'resolution_status': 'unresolved' if ref.get('proxy') else 'compiler_declaration_reference',
                             'compiler_witnesses': [expression['source_key'], witness['source_key']], **location(witness)}
                references.append(reference)
                add_relation(source, target, 'depends_on', [expression, witness],
                             written_reference=reference['written_reference'], dependency_semantics=semantics,
                             resolution_status=reference['resolution_status'], dependency_role='constraint_predicate')
        graph.nodes[source].update(
            constraint_dependency_status=('partial_predicate_references' if len(issues) > issue_count or any(
                r['resolution_status'] == 'unresolved' for r in references) else 'written_predicate_references')
                if expressions else 'no_written_result_expression',
            predicate_expressions=[{'expression': e['source'].strip(), 'compiler_witnesses': [e['source_key']],
                                    'evaluation_status': 'not_evaluated', **location(e)} for e in expressions],
            constraint_references=references)
        graph.nodes[source]['evidence_details'].append('Constraint reference extraction status: '+
            graph.nodes[source]['constraint_dependency_status']+'; predicates are not evaluated.')
        if references:
            graph.nodes[source]['evidence_details'].append('Predicate references (not evaluated): '+
                '; '.join(r['written_reference']+' -> '+graph.nodes[r['target']]['qualified_name']+
                          ' ['+r['resolution_status']+']' for r in references))
        if results and len(expressions) != len(results):
            issues.append({'element': source, 'reason': 'constraint result expression is not source-backed or complete'})

    for membership in rows.values():
        if membership['class'] != 'RequirementVerificationMembership' or not membership.get('source_backed') or membership['fields'].get('isImplied'):
            continue
        pending = refs(membership, 'owningRelatedElement')
        case, seen = None, set()
        while len(pending) == 1 and pending[0]['id'] in rows and pending[0]['id'] not in seen:
            current = rows[pending[0]['id']]
            seen.add(current['id'])
            if current['class'] in {'VerificationCaseDefinition', 'VerificationCaseUsage'}:
                case = current
                break
            pending = refs(current, 'owner')
        record = 'verification:'+membership['source_key']
        statement = (membership.get('source') or '').strip()
        name = f"verify@{membership['file']}:{membership.get('line')}"
        graph.add_node(record, name=name, qualified_name=(
            graph.nodes[ids[case['id']]]['qualified_name']+'::' if case and case['id'] in retained else '')+name,
            kind='verification_record', declared=False, compiler_witnesses=[membership['source_key']],
            source=membership.get('source'), execution_status='not_established_by_projection',
            evidence_details=['Modeled verification declaration; execution or passing is not established: '+statement],
            **location(membership))
        if case and case['id'] in retained:
            add_relation(ids[case['id']], record, 'declares_verification', [membership])
        else:
            issues.append({'element': record, 'reason': 'verification case owner not projected'})
        members = related(membership, 'ownedRelatedElement')
        bindings = [r for member in members for r in related(member) if r['class'] == 'ReferenceSubsetting'
                    and r.get('source_backed') and not r['fields'].get('isImplied')]
        terminal, chain = binding_target(bindings[0]) if len(bindings) == 1 else ([], [])
        witnesses = [membership, *bindings, *chain]
        # Repeated verifies can reference an earlier verification's unnamed
        # local reference occurrence. Follow only those explicit reference
        # bindings, never inheritance or effective instance features.
        alias_seen = set()
        while len(terminal) == 1 and not terminal[0].get('proxy') and terminal[0]['id'] in rows:
            alias = rows[terminal[0]['id']]
            if alias['fields'].get('declaredName') or not any(
                    r.get('class') == 'RequirementVerificationMembership' for r in refs(alias, 'owningRelationship')):
                break
            if alias['id'] in alias_seen:
                terminal = []
                break
            alias_seen.add(alias['id'])
            alias_bindings = [r for r in related(alias) if r['class'] == 'ReferenceSubsetting'
                              and r.get('source_backed') and not r['fields'].get('isImplied')]
            if len(alias_bindings) != 1:
                terminal = []
                break
            terminal, alias_chain = binding_target(alias_bindings[0])
            witnesses.extend([*alias_bindings, *alias_chain])
            chain.extend(alias_chain)
        # An explicitly named inline requirement can itself be the target.
        if not bindings and len(members) == 1 and members[0]['class'] == 'RequirementUsage' and members[0]['fields'].get('declaredName'):
            terminal = [{'id': members[0]['id'], 'class': members[0]['class'], 'proxy': False}]
        status = 'unsupported_target_structure'
        if len(terminal) == 1 and (terminal[0].get('proxy') or terminal[0].get('class') in {'RequirementUsage', 'RequirementDefinition'}):
            ref = terminal[0]
            status = 'unresolved' if ref.get('proxy') or any(
                r.get('proxy') for c in chain for r in refs(c, 'chainingFeature')) else 'compiler_declaration_reference'
            add_relation(record, ensure(ref), 'verifies', witnesses,
                         source_statement=statement,
                         written_reference=(bindings[0].get('source') or '').strip() if bindings else members[0].get('source', '').strip(),
                         resolution_status=status, verification_semantics='modeled_declaration_not_test_result')
            if case and case['id'] in retained:
                edge = add_relation(ids[case['id']], ensure(ref), 'verifies', witnesses,
                                    source_statement=statement, resolution_status=status,
                                    verification_semantics='modeled_declaration_not_test_result')
                edge.setdefault('verification_records', []).append(record)
                case_data = graph.nodes[ids[case['id']]]
                notice = 'Verification case declaration; execution or passing is not established.'
                if notice not in case_data['evidence_details']:
                    case_data['evidence_details'].append(notice)
        graph.nodes[record]['verification_reference_status'] = status
        graph.nodes[record]['evidence_details'].append('Verification target resolution: '+status+'.')
        if status != 'compiler_declaration_reference':
            issues.append({'element': record, 'reason': 'verification target not resolved', 'resolution_status': status})

    for key in sorted(retained):
        row, source = rows[key], ids[key]
        if row['class'] not in {'RequirementUsage', 'RequirementDefinition'}:
            continue
        data = graph.nodes[source]
        content = []
        short_id = row['fields'].get('declaredShortName')
        if short_id:
            data['requirement_id'] = short_id
            data['evidence_details'].append('Requirement ID: '+short_id)
        # Only directly owned text is attached. Child/inherited requirements'
        # documentation must not become this requirement's own description.
        children = [child for membership in related(row) for child in related(membership, 'ownedRelatedElement')]
        for child in children:
            if not child.get('source_backed') or child['fields'].get('isImplied'):
                continue
            if child['class'] == 'Documentation' and isinstance(child['fields'].get('body'), str):
                content.append({'role': 'documentation', 'text': child['fields']['body'],
                                'compiler_witnesses': [child['source_key']], **location(child)})
            elif child['class'] == 'MetadataUsage' and any(ref.get('declaredName') == 'Rationale' for ref in refs(child, 'type')):
                texts = [member for membership in related(child) for member in related(membership, 'ownedRelatedElement')
                         if (member['fields'].get('declaredName') or member['fields'].get('name')) == 'text']
                literals = [(member, expr) for member in texts for expr in feature_value(member)
                            if expr['class'] == 'LiteralString' and isinstance(expr['fields'].get('value'), str)]
                for member, literal in literals:
                    content.append({'role': 'rationale', 'text': literal['fields']['value'],
                                    'compiler_witnesses': [child['source_key'], member['source_key'], literal['source_key']],
                                    **location(child)})
                if not literals:
                    issues.append({'element': source, 'reason': 'requirement rationale is not a direct text literal',
                                   'compiler_witnesses': [child['source_key']]})
        data['requirement_content'] = content
        for record in content:
            provenance = f" [source: {record['file']}:{record['line']}]" if record.get('file') and record.get('line') is not None else ''
            data['evidence_details'].append('Requirement '+record['role']+provenance+': '+record['text'])
        if short_id or content:
            data['grounding_text'] = '\n'.join(
                (["Requirement ID: "+short_id] if short_id else [])
                + [data['qualified_name']+' '+data['kind']]
                + [r['role']+': '+r['text'] for r in content])

    for key in sorted(retained):
        row, source = rows[key], ids[key]
        if row['class'] == 'SatisfyRequirementUsage':
            graph.nodes[source]['is_negated'] = row['fields'].get('isNegated', False)
            subject_members = [r for r in related(row) if r['class'] == 'SubjectMembership']
            subjects = [expr for member in subject_members for child in related(member, 'ownedRelatedElement')
                        for expr in feature_value(child)]
            requirements = refs(row, 'satisfiedRequirement')
            for field, role in [('satisfiedRequirement', 'satisfied_requirement'), ('satisfyingFeature', 'satisfying_feature')]:
                for ref in refs(row, field):
                    add_relation(source, ensure(ref), role, [row],
                                 resolution_status='unresolved' if ref.get('proxy') else 'compiler_declaration_reference')
            if len(subjects) == 1 and len(requirements) == 1:
                path = (subjects[0].get('source') or '').strip()
                requirement = requirements[0].get('declaredName')
                if path and requirement:
                    owner = refs(row, 'owner')
                    scope = rows.get(owner[0]['id'], {}).get('fields', {}).get('qualifiedName', '') if owner else ''
                    a, b = 'assertion-subject:'+source, 'assertion-requirement:'+source
                    for node, name, kind in [(a, path, 'satisfaction_subject'), (b, requirement, 'requirement_reference')]:
                        graph.add_node(node, name=name, qualified_name=(scope+'::' if scope else '')+name,
                                       scope=scope, kind=kind, declared=False, resolution_status='syntactic_endpoint_only',
                                       compiler_witnesses=[row['source_key']], **location(row))
                    add_relation(a, b, 'not_satisfies' if row['fields'].get('isNegated') else 'satisfies', [row], source_statement=row['source'].strip(),
                                 resolution_status='literal_assertion_only')
                    add_relation(source, a, 'assertion_subject', [row])
                    add_relation(source, b, 'assertion_requirement', [row])
            continue
        if row['class'] not in CONNECTORS:
            continue
        ends = [r for r in related(row) if r['class'] == 'EndFeatureMembership' and r.get('source_backed')
                and not r['fields'].get('isImplied')]
        feature_ends = [r for r in related(row) if r['class'] == 'FeatureMembership' and r.get('source_backed')
                        and not r['fields'].get('isImplied') and any(
                            c['fields'].get('isEnd') for c in related(r, 'ownedRelatedElement'))]
        if row['class'] == 'ConnectionUsage' and feature_ends:
            end_ids = {r['id'] for r in ends + feature_ends}
            typed_connection_ends(row, source, [r for r in related(row) if r['id'] in end_ids])
            continue
        if row['class'] == 'ConnectionUsage' and row['fields'].get('isAbstract') and not ends and not feature_ends and any(
                r['class'] == 'MultiplicityRange' and r.get('source_backed') for r in descend(row)):
            graph.nodes[source]['connection_parse_status'] = 'abstract_connection_collection'
            graph.nodes[source]['evidence_details'].append(
                'Abstract connection collection declaration; no concrete endpoint bindings are asserted.')
            continue
        collected = []
        written_flow = version != LEGACY_PROJECTION_VERSION and row['class'] == 'FlowUsage'
        for index, membership in enumerate(ends):
            members = related(membership, 'ownedRelatedElement')
            if len(members) != 1:
                continue
            end = members[0]
            direction = None
            if written_flow:
                # A flow's ReferenceSubsetting covers the owner prefix. The
                # compiler's FlowEnd span also includes the terminal feature.
                direction = end['fields'].get('name')
                if end['class'] != 'FlowEnd' or not end.get('source_backed') or direction not in {'source', 'target'}:
                    continue
                reference = end
            else:
                subsettings = [r for r in related(end) if r['class'] == 'ReferenceSubsetting' and r.get('source_backed')]
                if len(subsettings) != 1:
                    continue
                reference = subsettings[0]
            path = (reference.get('source') or '').strip()
            if not path or (written_flow and not _FLOW_PATH.fullmatch(path)):
                continue
            alias = end['fields'].get('declaredName')
            scope = graph.nodes[source]['scope']
            occurrence = 'endpoint:'+ids[end['id']]
            chain = [] if written_flow else [ref for r in descend(reference) if r['class'] == 'FeatureChaining'
                                              for ref in refs(r, 'chainingFeature')]
            graph.add_node(occurrence, name=path, qualified_name=(scope+'::' if scope else '')+path,
                           kind='port_binding', declared=False, scope=scope, endpoint_path=path, endpoint_alias=alias,
                           resolution_status='syntactic_endpoint_only', compiler_witnesses=[membership['source_key'], reference['source_key']],
                           declaration_chain=[{'source_key': ref.get('source_key'), 'name': ref.get('declaredName'), 'proxy': ref.get('proxy')}
                                              for ref in chain], **location(reference))
            collected.append((occurrence, path, alias, reference, direction))
        binary = len(ends) == 2 and len(collected) == 2
        if written_flow:
            binary = binary and {end[4] for end in collected} == {'source', 'target'}
            graph.nodes[source]['written_flow_endpoints'] = binary
        graph.nodes[source]['connection_parse_status'] = 'binary_endpoints_preserved' if binary else 'unsupported_endpoint_structure'
        for index, (occurrence, path, alias, reference, direction) in enumerate(collected):
            role = (direction or ['source', 'target'][index]) + '_endpoint' if binary else 'endpoint'
            add_relation(source, occurrence, role, [reference], endpoint_path=path, endpoint_alias=alias,
                         endpoint_position=index, resolution_status='syntactic_endpoint_only')
        if not binary:
            issues.append({'element': source, 'reason': 'connector not represented as two complete written endpoint records',
                           'written_ends': len(ends), 'preserved_ends': len(collected)})

    if version == PROJECTION_VERSION:
        for key, row in rows.items():
            if row['class'] != 'SuccessionAsUsage' or not row.get('source_backed') or row['fields'].get('isImplied'):
                continue
            statement = (row.get('source') or '').strip()
            written = _ACTION_SUCCESSION.fullmatch(statement)
            owners, earlier, later = (refs(row, field) for field in ('owner', 'source', 'target'))
            if not written or any(len(group) != 1 or group[0].get('proxy') or group[0]['id'] not in retained
                                  for group in (owners, earlier, later)):
                continue
            owner, before, after = (ids[group[0]['id']] for group in (owners, earlier, later))
            if before == after or any(graph.nodes[action].get('compiler_class') != 'ActionUsage'
                   or graph.nodes[action].get('name') != name
                   or graph.get_edge_data(owner, action, {}).get('relationship') != 'owns'
                   for action, name in zip((before, after), written.groups())):
                continue
            scope = graph.nodes[owner]['qualified_name']
            node = ids[key]
            name = f"SuccessionAsUsage@{row['file']}:{row['line']}"
            graph.add_node(node, kind='action_succession', compiler_class=row['class'], declared=False,
                           name=name, qualified_name=f'{scope}::{name}', scope=scope, source=statement,
                           compiler_witnesses=[node], **location(row))
            add_relation(owner, node, 'owns', [row])
            add_relation(node, before, 'earlier_action', [row])
            add_relation(node, after, 'later_action', [row])
            retained.add(key)

    graph.graph.update(retained_engineering_elements=len(retained), projected_nodes=len(graph),
                       projected_edges=graph.number_of_edges(), projection_issues=issues,
                       raw_class_counts=dict(Counter(r['class'] for r in rows.values())),
                       node_kind_counts=dict(Counter(d['kind'] for _, d in graph.nodes(data=True))))
    return graph
