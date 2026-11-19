# Apollo and Vacuum candidate

The candidate prepares an evidence pack for a single natural-language question.
It checks projected identity, selects directly within a verified local
declaration, and falls back to the frozen router for other questions. It
preserves clarification stops. The pinned direct-selection path uses the
original MiniLM reranker, top-100 cutoff, whole-record packer and 4,096-token
Llama evidence cap. The complete prompt must fit a window of 8,192 tokens
with 512 reserved for output.

Ordinary local ownership, typing and connector questions use a Markdown tree.
The tree is rendered from the records already admitted by the packer and
checked against their directed edges and locators. Mixed and multi-step
requests use triples. The CLI writes exact chat messages or a fixed response
with token counts. It has no generation step.

## Offline example

Use Python 3.13 with the dependencies in `requirements-candidate.txt`.
Bring local snapshots of the all-MiniLM-L6-v2 encoder at revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41` and a Llama 3.1 Instruct chat
tokenizer with its chat template. The artifact counts tokens with the
supplied template. For parity with a specific Ollama build, use a tokenizer
verified against that build. Model and tokenizer files are obtained
separately under their upstream terms. The example itself runs offline.

From the checkout or extracted source snapshot, substitute the two local model
directories. Choose new output filenames.

```text
python -B -m sysml_graphrag --graph examples/assembly/assembly-graph.json --encoder <encoder-directory> --tokenizer <tokenizer-directory> --question "Which part usages does DemoAssembly directly declare, and what are their declared types?" --out assembly-prompt.json
python -B -m sysml_graphrag --graph examples/assembly/assembly-graph.json --encoder <encoder-directory> --tokenizer <tokenizer-directory> --question "What is the declared type of sensorHead?" --out assembly-clarification.json
```

The first command should report `DIRECT_SCOPED` and `markdown_tree`. Its
evidence should cite the `sensorHead` usage and its `OpticalSensor` type.
The second should report `CLARIFICATION_REQUIRED` and zero planned generator
input, with `DemoAssembly::sensorHead` and `BackupAssembly::sensorHead` in
the fixed response. The graph is a hand-authored synthetic projection, with
locators into `examples/assembly/assembly.sysml`.

## External projections

Use the original compact projection JSON shape. The top-level fields are
`metadata`, ordered `nodes` (each with an `id`), and ordered `edges`
(each with source and target IDs and a `relationship`). Identity and
route checks use projected qualified names, kinds and declared/scope fields.
Evidence rendering also uses source paths, lines, assertions and role
witnesses when present. Supply the compact graph produced by your compiler
adapter. The example describes the minimum shape for the inventory route.
Java, model sources, compiler binaries and private experiment artifacts are
separate inputs to the relevant workflows.

The existing Flashlight example uses a separate compiler-backed walkthrough
and a MiniLM-token evidence cap. Its results have a different budget and
projection from the candidate baseline.
