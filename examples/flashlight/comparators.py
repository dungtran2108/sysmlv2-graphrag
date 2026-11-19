"""Pilot comparators. These are declared implementations, not paper replications."""
import math
import re
from collections import Counter

import numpy as np

from .source_lexing import clean_comments, segments


def declaration_chunks(sources):
    """Whole supported lexical declarations with enclosing scope and source lines.

    The shared lexer splits outside strings/comments. This is a regex-supported
    source baseline, not a compiler AST. Body declarations are separate chunks;
    repeated scope headers retain context without copying entire enclosing bodies.
    """
    chunks = []
    for path, text in sorted(sources.items()):
        lines, stack = text.splitlines(), []
        for statement, delimiter, line in segments(clean_comments(text)):
            if statement:
                end = line + statement.count('\n')
                snippet = '\n'.join(lines[line - 1:end])
                chunks.append({'id': f'{path}:{line}:{end}:{len(chunks)}',
                               'text': f'Source: {path}:{line}-{end}\n'
                                       + ('Enclosing declarations: ' + ' / '.join(stack) + '\n' if stack else '')
                                       + snippet})
            if delimiter == '{':
                stack.append(statement.strip())
            elif delimiter == '}' and stack:
                stack.pop()
    return chunks


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

