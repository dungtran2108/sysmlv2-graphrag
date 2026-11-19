"""Pilot comparators. These are declared implementations, not paper replications."""
import math
import re
from collections import Counter

import networkx as nx
import numpy as np

from candidate.retrieval.traversal import stringify_subgraph_triples


def words(text):
    # Include whole literal paths and camel-case components for both exact and
    # natural-language matches, without query expansion or answer-label aliases.
    literal = re.findall(r"[A-Za-z0-9_]+(?:(?:::|[.\-])[A-Za-z0-9_]+)*", text)
    parts = re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', text)
    return [word.lower() for word in literal + re.findall(r'[A-Za-z0-9]+', parts)]


class BM25:
    def __init__(self, texts, k1=1.5, b=0.75):
        self.docs = [Counter(words(text)) for text in texts]
        self.lengths = [sum(doc.values()) for doc in self.docs]
        self.average = sum(self.lengths) / len(self.docs) if self.docs else 1.0
        self.frequency = Counter(term for doc in self.docs for term in doc)
        self.k1, self.b = k1, b

    def scores(self, query):
        result = []
        for doc, length in zip(self.docs, self.lengths):
            score = 0.0
            for term in set(words(query)):
                frequency = doc[term]
                df = self.frequency[term]
                idf = math.log(1 + (len(self.docs) - df + 0.5) / (df + 0.5))
                denominator = frequency + self.k1 * (1 - self.b + self.b * length / self.average)
                score += idf * frequency * (self.k1 + 1) / denominator if denominator else 0
            result.append(score)
        return result


class EvidenceIndex:
    def __init__(self, units, embedder, hybrid=False, vectors=None):
        self.units, self.embedder, self.hybrid = units, embedder, hybrid
        texts = [row.get('embedding_text', row['text']) for row in units]
        self.vectors = (embedder.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
                        if vectors is None else np.asarray(vectors))
        if (self.vectors.ndim != 2 or self.vectors.shape[0] != len(units)
                or not np.isfinite(self.vectors).all()):
            raise ValueError('Invalid embedding cache shape or values.')
        lengths = embedder.tokenizer(texts, truncation=False, padding=False, return_length=True)['length']
        self.embedding_diagnostics = {'max_input_tokens': max(lengths, default=0),
            'inputs_over_encoder_limit': sum(length > embedder.max_seq_length for length in lengths),
            'separate_ranking_text_inputs': sum('embedding_text' in row for row in units),
            'encoder_limit': embedder.max_seq_length,
            'limitation': 'Embedding input truncation affects ranking; full original units remain available for evidence packing.'}
        self.lexical = BM25(texts) if hybrid else None

    def rank(self, query):
        vector = self.embedder.encode(query, normalize_embeddings=True, convert_to_numpy=True)
        dense = np.asarray(self.vectors @ vector).reshape(-1)
        dense_order = sorted(range(len(self.units)), key=lambda i: (-float(dense[i]), self.units[i]['id']))
        scores = {i: float(dense[i]) for i in dense_order}
        order = dense_order
        if self.hybrid:
            lexical = self.lexical.scores(query)
            lexical_order = sorted((i for i in range(len(self.units)) if lexical[i] > 0),
                                   key=lambda i: (-lexical[i], self.units[i]['id']))
            scores = {i: 1 / (60 + rank) for rank, i in enumerate(dense_order, 1)}
            for rank, i in enumerate(lexical_order, 1):
                scores[i] += 1 / (60 + rank)
            order = sorted(scores, key=lambda i: (-scores[i], self.units[i]['id']))
        return [{**self.units[i], 'score': scores[i]} for i in order]


def graph_units(evidence):
    """Atomic directed edge records plus isolated nodes, with complete identities.

    Keep named interface roles and verified flow endpoints together for packing.
    """
    graph = nx.DiGraph()
    graph.add_nodes_from((row['id'], {k: v for k, v in row.items() if k != 'id'})
                         for row in sorted(evidence['nodes'], key=lambda row: row['id']))
    graph.add_edges_from((row['source'], row['target'],
                          {k: v for k, v in row.items() if k not in {'source', 'target'}})
                         for row in sorted(evidence['edges'], key=lambda row: (row['source'], row['target'])))
    units, handled = [], set()
    for record in evidence.get('records', []):
        if record.get('kind') == 'action_order':
            lines = ['Written immediate action order.']
            for node in record['successions']:
                declaration = graph.nodes[node]
                start, end = declaration['line'], declaration.get('end_line')
                span = f'{start}-{end}' if type(end) is int and end > start else str(start)
                lines.append(f"- In {declaration['scope']}, {declaration['source']} [source: {declaration['file']}:{span}]")
            handled.update(map(tuple, record['edges']))
            units.append({'id': record['id'], 'text': '\n'.join(lines), 'edges': record['edges'], 'record': record})
            continue
        if record.get('kind') == 'allocation_inventory':
            lines = ['Written allocations by declaring scope. Paths in different scopes are separate model references.']
            for allocation, source, target in record['allocations']:
                declaration = graph.nodes[allocation]
                start, end = declaration['line'], declaration.get('end_line')
                span = f'{start}-{end}' if type(end) is int and end > start else str(start)
                lines.append(f"- In {declaration['scope']}, allocate {graph.nodes[source]['endpoint_path']} "
                             f"to {graph.nodes[target]['endpoint_path']}; [source: {declaration['file']}:{span}]")
            handled.update(map(tuple, record['edges']))
            units.append({'id': record['id'], 'text': '\n'.join(lines), 'edges': record['edges'], 'record': record})
            continue
        if record.get('kind') == 'direct_inventory':
            root = record['root']
            lines = [f"Direct part usages declared in {graph.nodes[root]['qualified_name']}:"]
            for _, usage in record['edges']:
                witness = graph[root][usage]
                start, end = witness['line'], witness.get('end_line')
                span = f'{start}-{end}' if type(end) is int and end > start else str(start)
                lines.append(f"- {graph.nodes[usage]['name']} [source: {witness['file']}:{span}]")
            handled.update(map(tuple, record['edges']))
            units.append({'id': record['id'], 'text': '\n'.join(lines),
                          'edges': record['edges'], 'record': record})
            continue
        if record.get('kind')!='inventory_member':
            continue
        edges=[tuple(e) for e in record['edges']]
        if not edges or not all(graph.has_edge(u,v) for u,v in edges):
            raise ValueError('Inventory member record is incomplete in selected evidence.')
        handled.update(edges)
        notice='Direct declaration/type evidence; inherited and realized instances excluded.\n'
        if not record['type_resolved']:
            notice+='Declared type unresolved or unsupported; no type is inferred.\n'
        if record.get('search_limits_reached'):
            notice+='Inventory limit reached; coverage is incomplete.\n'
        member=graph.edge_subgraph(edges)
        units.append({'id':'inventory:'+record['id'],'text':notice+stringify_subgraph_triples(member),
                      'embedding_text':'Direct declaration and type: '+'; '.join(d.get('qualified_name',str(n)) for n,d in member.nodes(data=True)),
                      'edges':[list(e) for e in edges],'record':record})
    for record in evidence.get('records', []):
        if record.get('kind') != 'traceability_path':
            continue
        edges = [tuple(edge) for edge in record['edges']]
        if not edges or not all(graph.has_edge(u, v) for u, v in edges):
            raise ValueError('Traceability path record is incomplete in selected evidence.')
        handled.update(edges)
        if record.get('intent') in {'satisfaction', 'satisfaction_allocation'}:
            notice = ('Written satisfy assertion and any requested allocation evidence. '
                      'Endpoint paths are literal and do not establish executed verification.\n')
            if record.get('other_assertions_unmatched'):
                notice += 'Other positive assertions lack the complete requested evidence in this projection.\n'
        else:
            notice = 'Modeled traceability path: candidate review obligations, not predicted failures or executed test results.\n'
        if record.get('search_limits_reached'):
            notice += 'Search limits reached; this path does not establish complete impact coverage.\n'
        path = graph.edge_subgraph(edges)
        # Rank using compact semantic labels; retain full cited evidence for
        # whole-path packing. Existing comparators without this field keep
        # their original full-text embedding input.
        ranking_text = ('Modeled '+record.get('intent', 'traceability').replace('_', ' ')+': '+
                        '; '.join(str(d.get('kind', 'element'))+': '+
                                  str(d.get('requirement_id', ''))+' '+str(d.get('name', n))
                                  for n, d in path.nodes(data=True))+
                        '. Relations: '+', '.join(sorted({e.get('relationship', '') for _, _, e in path.edges(data=True)})))
        units.append({'id': 'traceability:'+record['id'],
                      'text': notice+stringify_subgraph_triples(path), 'embedding_text': ranking_text,
                      'edges': [list(edge) for edge in edges], 'record': record})
    for node, data in graph.nodes(data=True):
        flow = data.get('written_flow_endpoints') is True
        if data.get('kind') != 'interface' and not flow:
            continue
        roles = [(u, v) for u, v, edge in graph.out_edges(node, data=True)
                 if edge.get('relationship') in {'source_endpoint', 'target_endpoint'}]
        if roles:
            handled.update(roles)
            if flow and (len(roles) != 2 or len({graph.edges[edge]['relationship'] for edge in roles}) != 2):
                continue
            unit = {'id': f'connector:{node}', 'text': stringify_subgraph_triples(graph.edge_subgraph(roles)),
                    'edges': [list(edge) for edge in roles]}
            if flow:
                paths = {graph.edges[edge]['relationship']: graph.nodes[edge[1]]['endpoint_path'] for edge in roles}
                unit['embedding_text'] = (f"Directed flow from output {paths['source_endpoint']} "
                                          f"to input {paths['target_endpoint']}.")
            units.append(unit)
    for source, target in graph.edges():
        if (source, target) not in handled:
            units.append({'id': f'edge:{source}->{target}',
                          'text': stringify_subgraph_triples(graph.edge_subgraph([(source, target)])),
                          'edges': [[source, target]]})
    for node in nx.isolates(graph):
        units.append({'id': f'node:{node}', 'text': stringify_subgraph_triples(graph.subgraph([node])), 'edges': []})
    for node, data in graph.nodes(data=True):
        if data.get('compiler_class') != 'TransitionUsage':
            continue
        source = (data.get('source') or '').strip()
        start, end = data.get('line'), data.get('end_line')
        if not source or not data.get('file') or type(start) is not int or start < 1:
            continue
        span = f'{start}-{end}' if type(end) is int and end > start else str(start)
        text = (f"Written state transition in {data.get('scope') or data['qualified_name']} "
                f"[source: {data['file']}:{span}]:\n{source}")
        units.append({'id': f'transition:{node}', 'text': text, 'edges': []})
    return units
