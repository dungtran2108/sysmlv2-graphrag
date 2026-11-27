# Apollo 11 source-to-answer walkthrough

Start with the pinned Airbus Apollo 11 SysML v2 model. The notebook runs
the SysML Pilot compiler, builds a compact projection, and compares Text
Hybrid, One-Hop, and the simplified repository's Routed GraphRAG candidate.
The sample question asks for the interface type and both written endpoint
paths of `ApolloSpacecraft::lesConnection`.

## Inputs

- Clone the [Apollo 11 model](https://github.com/airbus/apollo-11-sysml-v2) outside
  this checkout and check out `6e9c93fe7d80c5ca3534bb14b10ab374a643ef2d`.
  The example verifies the source inventory and stages the original compiler
  run's CRLF line endings in scratch.
- Use Java 21 and the extracted `sysml` folder of the
  [SysML Pilot 0.62.0 release](https://github.com/Systems-Modeling/SysML-v2-Pilot-Implementation/releases/download/2026-08/jupyter-sysml-kernel-0.62.0.zip).
  The observer, compiler JAR, and standard library are checked against pinned hashes.
- Install the Python dependencies from `requirements-candidate.txt` and
  supply the local MiniLM and Llama 3.1 tokenizer directories described in
  the [candidate guide](../../CANDIDATE.md#offline-example).

## Run

Install JupyterLab in the environment with the candidate dependencies. Open
[Apollo.ipynb](Apollo.ipynb) from this checkout and fill in the four paths in
step 1. Step 2 compiles the model and builds the local indexes in a new
temporary directory. Change `question` in step 3 and rerun from there.
The notebook shows the chosen route, ranked record IDs, packed source evidence
and input token counts. Inspect `results[method]['packed']['messages']` to see
the exact messages. Each method gets 4,096 evidence tokens within the same
8,192-token context window with 512 reserved for output. Arbitrary questions
are unscored.

Allow several minutes for the first compile and CPU indexing. Keep the kernel
open to reuse the projection and indexes when you change the question. The
first question that uses the fallback router may also build node embeddings.
MiniLM can truncate long units for ranking, while the packer sees each original
unit and admits whole units under the Llama token budget.

The notebook can also read the four setup paths from `SYSML_APOLLO_SOURCE`,
`SYSML_PILOT_DIR`, `SYSML_ENCODER_DIR` and `SYSML_TOKENIZER_DIR`. Set
`SYSML_DEMO_SCRATCH` to an existing external directory if you want to choose
where it creates the temporary run folder.

The CLI runs the compiler and the Routed GraphRAG arm from the checkout root.

```text
python -B -m examples.apollo.demo --source <apollo-checkout> --pilot <pilot-sysml-directory> --encoder <minilm-directory> --tokenizer <llama-tokenizer-directory> --out <new-external-directory>
```

Expect `DIRECT_SCOPED` and `markdown_tree`. Inspect the evidence for
`LESInterface`, source `commandServiceModule.commandModule.lesInterfacePort`,
and target `launchEscapeSystem.cmInterfacePort`, with a locator at line 156
of `Technical/TechnicalComponentsPackage.sysml`. The paths are written
references in the declaration. The output includes `compiler.log`,
`compiler-export.json`, `graph.json`, and `prompt.json`.

Optional generation uses local Ollama at `127.0.0.1:11434`. Run the final
notebook cell after setting `generate = True`, or append `--generate` to the
CLI command. The example verifies the installed `llama3.1:latest` digest
listed in [inputs.json](inputs.json) and applies the replay's 8,192-token
context and 512-token completion limit. The notebook displays the answer and
token telemetry. The CLI saves them. Use a tokenizer verified against the installed model for a
matching input-token count. A fixed clarification stops before any call.

The source model, compiler distribution, encoder, tokenizer, and Ollama
weights are obtained separately. See the [upstream Apollo notice](https://github.com/airbus/apollo-11-sysml-v2/blob/6e9c93fe7d80c5ca3534bb14b10ab374a643ef2d/LICENSE.md)
for the model's terms.
