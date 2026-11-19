"""Local projection and model loading for candidate prompt preparation."""
import json
import os
from pathlib import Path

import networkx as nx


def read(path):
    return json.loads(Path(path).read_bytes())


def load_graph(path):
    document = read(path)
    metadata = document['metadata']
    if metadata.get('diagnostic_only'):
        raise ValueError('The candidate requires a valid or explicitly synthetic projection.')
    graph = nx.DiGraph(**metadata)
    for row in document['nodes']:
        data = dict(row)
        node = data.pop('id')
        if not isinstance(node, str) or not node or node in graph:
            raise ValueError(f'Invalid or duplicate node ID: {node!r}')
        graph.add_node(node, **data)
    if not graph:
        raise ValueError('A nonempty projection is required.')
    for row in document['edges']:
        data = dict(row)
        a, b = data.pop('source'), data.pop('target')
        if a not in graph or b not in graph or graph.has_edge(a, b) or not data.get('relationship'):
            raise ValueError(f'Invalid projected edge: {(a, b)!r}')
        graph.add_edge(a, b, **data)
    return graph


def local_models(encoder_path, tokenizer_path):
    os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    encoder_path = Path(encoder_path).resolve(strict=True)
    tokenizer_path = Path(tokenizer_path).resolve(strict=True)
    if not (encoder_path / 'model.safetensors').is_file() or not tokenizer_path.is_dir():
        raise ValueError('Provide a complete local encoder snapshot and chat tokenizer directory.')
    torch.set_num_threads(4)
    encoder = SentenceTransformer(str(encoder_path), local_files_only=True, device='cpu').eval()
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True)
    return encoder, tokenizer


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
