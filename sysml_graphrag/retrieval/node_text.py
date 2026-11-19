"""Shared node text for fresh grounding caches; baseline nodes are unchanged."""
import hashlib
import json


def node_grounding_text(node, data):
    return data.get('grounding_text') or (
        f"{data.get('qualified_name', data.get('name', str(node)))} {data.get('kind', '')}".strip())


def node_text_digest(graph):
    rows = [(str(n), node_grounding_text(n, d)) for n, d in graph.nodes(data=True)]
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False).encode('utf-8')).hexdigest()
