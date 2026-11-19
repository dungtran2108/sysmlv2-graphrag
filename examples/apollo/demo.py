"""Compile pinned Apollo sources and prepare a scoped evidence prompt."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
from urllib.request import Request, urlopen

from sysml_graphrag.compiler_projection import LEGACY_PROJECTION_VERSION, build_retrieval_projection
from sysml_graphrag.io import local_models, read, write

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PINS = read(HERE / 'inputs.json')
QUESTION = 'What interface type and two written endpoint paths does ApolloSpacecraft declare for lesConnection?'
OLLAMA = 'http://127.0.0.1:11434'
CONTEXT = 8192
OUTPUT = 512


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def model_files(root):
    return {p.relative_to(root).as_posix(): p for p in root.rglob('*')
            if p.is_file() and p.suffix in {'.sysml', '.kerml'}}


def compile_model(source, pilot, out):
    source, pilot, out = (Path(p).resolve() for p in (source, pilot, out))
    if out.is_relative_to(ROOT) or out.is_relative_to(source) or out.is_relative_to(pilot):
        raise ValueError('Choose a new output directory outside the checkout and model inputs.')
    jar = pilot / 'jupyter-sysml-kernel-0.62.0-all.jar'
    observer = HERE / 'CompilerExport.java'
    if digest(jar) != PINS['compiler_jar_sha256'] or digest(observer) != PINS['observer_sha256']:
        raise ValueError('Compiler or observer digest differs from the pinned inputs.')
    library = pilot / 'sysml.library'
    if {name: digest(p) for name, p in model_files(library).items()} != read(
            HERE.parent / 'flashlight' / 'stdlib-sha256.json'):
        raise ValueError('Standard library inventory differs from the pinned release.')
    inventory = model_files(source)
    if set(inventory) != set(PINS['source_sha256']):
        raise ValueError('Apollo source inventory differs from the pinned revision.')
    # Normalize checkout line endings to the original compiler run's bytes.
    files = {name: path.read_bytes().replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')
             for name, path in inventory.items()}
    if {name: sha256(data).hexdigest() for name, data in files.items()} != PINS['source_sha256']:
        raise ValueError('Apollo source contents differ from the pinned revision.')
    java = subprocess.run(['java', '-version'], capture_output=True, text=True, check=True).stderr
    if not re.search(r'version "21(?:\.|")', java):
        raise ValueError('Use Java 21.')
    out.mkdir(parents=True, exist_ok=False)
    staged = out / 'sources'
    for name, data in files.items():
        path = staged / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    export = out / 'compiler-export.json'
    log = out / 'compiler.log'
    with log.open('w', encoding='utf-8') as stream:
        run = subprocess.run(['java', '-Xmx2g', '-cp', str(jar), str(observer), str(staged),
                              str(export), str(library)], stdout=stream, stderr=subprocess.STDOUT)
    if run.returncode:
        raise RuntimeError(f'Compiler observer failed. See {log}')
    return export


def project_model(export):
    graph = build_retrieval_projection(read(export), version=LEGACY_PROJECTION_VERSION)
    projection = Path(export).with_name('graph.json')
    write(projection, {'metadata': graph.graph,
                       'nodes': [{'id': node, **data} for node, data in graph.nodes(data=True)],
                       'edges': [{'source': a, 'target': b, **data} for a, b, data in graph.edges(data=True)]})
    return graph


def prepare_prompt(graph, encoder_path, tokenizer_path, question):
    encoder, tokenizer = local_models(encoder_path, tokenizer_path)
    from sysml_graphrag.policy import Candidate

    return Candidate(graph, encoder, tokenizer).prepare(question)


def ollama_json(endpoint, payload=None):
    request = Request(OLLAMA + endpoint, data=None if payload is None else json.dumps(payload).encode(),
                      headers={'Content-Type': 'application/json'})
    with urlopen(request, timeout=300 if payload is not None else 10) as response:
        return json.load(response)


def generate_local(row):
    """Send the exact prepared prompt to the pinned local model."""
    if not row['generation_planned']:
        return {'generator_called': False, 'answer': row['prompt_fit']['fixed_response']}
    if not row['prompt_fit']['pass_fit']:
        raise ValueError('The prepared prompt does not fit.')
    model = PINS['ollama_model']
    if not any(item['name'] == model and item['digest'] == PINS['ollama_digest']
               for item in ollama_json('/api/tags')['models']):
        raise ValueError('The installed Ollama model differs from the pinned model and digest.')
    result = ollama_json('/api/chat', {
        'model': model, 'messages': row['packed']['messages'], 'stream': False,
        'options': {'temperature': 0, 'seed': 0, 'num_ctx': CONTEXT, 'num_predict': OUTPUT},
    })
    return {'generator_called': True, 'answer': result['message']['content'],
            'model': result['model'], 'finish_reason': result.get('done_reason'),
            'input_tokens': result.get('prompt_eval_count'), 'output_tokens': result.get('eval_count'),
            'input_tokens_match': result.get('prompt_eval_count') == row['packed']['input_tokens']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'pilot', 'out', 'encoder', 'tokenizer'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--question', default=QUESTION)
    parser.add_argument('--generate', action='store_true')
    args = parser.parse_args()
    if not args.question.strip():
        parser.error('Supply a nonempty question.')
    export = compile_model(args.source, args.pilot, args.out)
    graph = project_model(export)
    row = prepare_prompt(graph, args.encoder, args.tokenizer, args.question)
    write(args.out / 'prompt.json', row)
    print(row['router_result']['route'], row['renderer'], row['prompt_fit']['input_tokens_to_send'])
    if args.generate:
        response = generate_local(row)
        write(args.out / 'answer.json', response)
        print(response['answer'])


if __name__ == '__main__':
    main()
