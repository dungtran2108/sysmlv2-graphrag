"""Shared evidence selection and controlled response handling for pilot runs."""
import time

from sysml_graphrag.evaluation.budget import Budget, COMPOSITION_PACKING, pack_evidence
from sysml_graphrag.evaluation.comparators import EvidenceIndex, graph_units
from sysml_graphrag.retrieval.composition import composition_packing_edges


_WHOLE_INVENTORIES = {
    'direct_inventory': ('INVENTORY_UNVERIFIED', 'direct-part inventory'),
    'allocation_inventory': ('TRACEABILITY_UNVERIFIED', 'allocation evidence'),
    'action_order': ('TRACEABILITY_UNVERIFIED', 'action-order evidence'),
}


def rank_evidence(query, method, graph, vectors, embedder, router, indexes, node_digest):
    start = time.perf_counter()
    routed = None
    diagnostics = None
    priority_edges = []
    if method == 'graph_routed':
        routed = router.retrieve_context(query, graph, embedder, vectors,
                                         precomputed_node_text_sha256=node_digest)
        units = graph_units(routed['selected_evidence'])
        if len(units) == 1 and units[0].get('record', {}).get('kind') in _WHOLE_INVENTORIES:
            ranked = units
        else:
            index = EvidenceIndex(units, embedder) if units else None
            ranked = index.rank(query) if index else []
            diagnostics = index.embedding_diagnostics if index else None
            identity = routed.get('metadata', {}).get('query_identity')
            if routed['execute_llm'] and identity:
                priority_edges = composition_packing_edges(graph, query, identity)
    else:
        ranked = indexes[method].rank(query)
    return {'ranked': ranked, 'router_result': routed,
            'priority_edges': priority_edges,
            'ranking_embedding_diagnostics': diagnostics,
            'retrieval_seconds': time.perf_counter() - start}


def condition(query, method, selection, allowance, counter, config):
    """Build one budgeted condition. Generation is a separate explicit operation."""
    limit = config['max_evidence_candidates']
    ranked = selection['ranked']
    start = time.perf_counter()
    generator = config['generator']
    candidates = ranked[:limit]
    priority = []
    required = set(selection.get('priority_edges', ()))
    if required and config.get('packing_policy') == COMPOSITION_PACKING:
        covered = set()
        for unit in candidates:
            hits = required.intersection(map(tuple, unit.get('edges', ())))
            if hits - covered:
                priority.append(unit['id'])
                covered.update(hits)
        if covered != required:
            priority = []
    packed = pack_evidence(query['query'], candidates, counter,
                           Budget(allowance, generator['context_window'], generator['output_reserve']),
                           priority_unit_ids=priority)
    routed = selection['router_result']
    whole = _WHOLE_INVENTORIES.get(ranked[0].get('record', {}).get('kind')) if ranked else None
    if whole and not packed['selected_unit_ids']:
        route, description = whole
        reason = f'The complete cited {description} could not be packed within the evidence limits.'
        routed = {**routed, 'route': route, 'execute_llm': False,
                  'default_response': reason, 'metadata': {**routed['metadata'], 'reason': reason}}
    by_id = {unit['id']: unit for unit in candidates}
    return {
        'query_id': query['query_id'], 'query_family_id': query['query_family_id'],
        'query': query['query'], 'model': query['model'], 'partition': query['partition'],
        'method': method, 'evidence_allowance': allowance,
        'automatic_grounding': True, 'router_result': routed,
        'retrieval_seconds': selection['retrieval_seconds'],
        'ranking_embedding_diagnostics': selection['ranking_embedding_diagnostics'],
        'packing_seconds': time.perf_counter() - start,
        'candidate_limit': limit, 'candidate_limit_excluded_ids': [u['id'] for u in ranked[limit:]],
        'ranked_units': candidates,
        'packed_units': [by_id[unit_id] for unit_id in packed['selected_unit_ids']],
        'packed': packed, 'answer_score': None, 'retrieval_score': None,
        'generator_called': False,
        'timing_condition': 'single sequential run; indexes reused across budgets; descriptive only',
    }
