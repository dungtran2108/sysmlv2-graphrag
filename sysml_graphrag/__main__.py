"""Prepare and save one exact prompt or router stop without generation."""
import argparse
from pathlib import Path

from sysml_graphrag.io import load_graph, local_models, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph', type=Path, required=True)
    parser.add_argument('--encoder', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--question', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if not args.question.strip():
        parser.error('Supply a nonempty question.')
    graph = load_graph(args.graph)
    encoder, tokenizer = local_models(args.encoder, args.tokenizer)
    from sysml_graphrag.policy import Candidate

    row = Candidate(graph, encoder, tokenizer).prepare(args.question)
    write(args.out, row)
    print(row['router_result']['route'], row['renderer'], row['prompt_fit']['input_tokens_to_send'],
          'planned input tokens')


if __name__ == '__main__':
    main()
