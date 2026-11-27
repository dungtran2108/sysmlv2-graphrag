# Apollo and Vacuum candidate

This package implements the Apollo and Vacuum Routed GraphRAG baseline
`routed-reverse-allocation-2026-11-25`. It prepares evidence for one question
against an ordered compiler projection.

- The identity guard preserves fixed responses for ambiguous or unverified
  references. Ordinary ownership, typing and connector questions can use
  direct scoped selection when the projection verifies one declaration.
- For deep retrieval, the policy can focus on verified typed composition,
  written satisfaction or multi-scope harness wiring. It restores both
  written roles and ownership for surviving binary connection, interface
  and allocation relations and packs each restored relation as one unit.
- For supported questions about requirements met by work assigned to a
  component, the policy follows compiler-witnessed incoming allocations and
  checks satisfaction in the correct typed scope. Whole traces compete with
  the existing evidence pool. When scopes are ambiguous or the trace cannot
  be verified, the router can return a fixed response with the available
  source locators. Standalone allocation, typing and satisfaction requests
  retain their normal routes.

The pinned MiniLM reranker supplies up to 100 candidates to the whole-record
packer. The Llama evidence cap is 4,096 tokens within an 8,192-token context,
with 512 reserved for output. Ordinary direct-scoped evidence is rendered
as a Markdown tree when its edges and locators can be checked against the
admitted records. Mixed and multi-step requests use triples. The CLI
prepares exact chat messages or a fixed response and reports both the
prepared prompt size and the input tokens planned for sending. A fixed
response has zero planned input tokens. The CLI runs offline.

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
Evidence rendering uses source paths, lines, assertions and written role
witnesses. Reverse allocation requires compiler witness keys, resolved
declaration chains and explicit satisfaction relationships. Supply the
compact graph produced by your compiler adapter. The example describes the minimum shape for the inventory route.
Java, model sources, compiler binaries and private experiment artifacts are
separate inputs to the relevant workflows.

The existing Flashlight example uses a separate compiler-backed walkthrough
and a MiniLM-token evidence cap. Its results have a different budget and
projection from the candidate baseline.
