"""Reuse one compiler projection and two comparator indexes across questions."""
from pathlib import Path

from examples.apollo.demo import model_files
from examples.flashlight.comparators import declaration_chunks
from sysml_graphrag.evaluation.comparators import EvidenceIndex
from sysml_graphrag.evaluation.pilot_conditions import condition, rank_evidence
from sysml_graphrag.io import local_models
from sysml_graphrag.policy import CAP, CONFIG, Candidate, fit
from sysml_graphrag.retrieval.traversal import stringify_subgraph_triples


def one_hop_units(graph):
    """Project one indivisible radius-one neighborhood per center."""
    for center in graph:
        nodes = {center, *graph.predecessors(center), *graph.successors(center)}
        yield {'id': f'onehop:{center}', 'text': stringify_subgraph_triples(graph.subgraph(nodes))}


class ApolloComparison:
    """Run three independent prompt preparations on the same model snapshot."""

    def __init__(self, graph, staged_sources, encoder_path, tokenizer_path):
        encoder, tokenizer = local_models(encoder_path, tokenizer_path)
        sources = {name: path.read_text(encoding='utf-8-sig')
                   for name, path in model_files(Path(staged_sources)).items()}
        if not sources:
            raise ValueError('Provide the staged sources from the Apollo compiler run.')
        self.candidate = Candidate(graph, encoder, tokenizer)
        self.indexes = {
            'text_hybrid': EvidenceIndex(declaration_chunks(sources), encoder, hybrid=True),
            'one_hop': EvidenceIndex(list(one_hop_units(graph)), encoder),
        }

    def prepare(self, question):
        question = question.strip()
        if not question:
            raise ValueError('Supply a nonempty engineering question.')
        candidate = self.candidate
        card = dict(query=question, model='apollo', query_id='live',
                    query_family_id='live', partition='unscored')
        results = {}
        for method in self.indexes:
            selection = rank_evidence(question, method, candidate.graph, None,
                                      candidate.encoder, None, self.indexes, candidate.node_digest)
            row = condition(card, method, selection, CAP, candidate.counter, CONFIG)
            row.update(prompt_fit=fit(row, candidate.counter), generation_planned=True)
            results[method] = row
        results['graph_routed'] = candidate.prepare(question, query_id='live', model='apollo')
        return results
