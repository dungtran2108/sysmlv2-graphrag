# Apollo 11 source-to-answer walkthrough

Start with the pinned Airbus Apollo 11 SysML v2 model. The walkthrough runs
the SysML Pilot compiler, projects its source-backed elements and relations,
and prepares an exact prompt with the same scoped retrieval policy used in
the Apollo/Vacuum replay. The sample question asks for the interface type
and both written endpoint paths of `ApolloSpacecraft::lesConnection`.

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

Open [Apollo.ipynb](Apollo.ipynb) in Jupyter from the checkout and fill in
the external paths. The notebook shows the compiler export, compact graph,
route, evidence, citation, and exact messages. Use a new output directory
outside this checkout and the input directories.

The CLI runs the same steps from the checkout root.

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
notebook cell after setting `GENERATE = True`, or append `--generate` to the
CLI command. The example verifies the installed `llama3.1:latest` digest
listed in [inputs.json](inputs.json) and applies the replay's 8,192-token
context and 512-token completion limit. It saves the answer and token
telemetry. Use a tokenizer verified against the installed model for a
matching input-token count. A fixed clarification stops before any call.

The source model, compiler distribution, encoder, tokenizer, and Ollama
weights are obtained separately. See the [upstream Apollo notice](https://github.com/airbus/apollo-11-sysml-v2/blob/6e9c93fe7d80c5ca3534bb14b10ab374a643ef2d/LICENSE.md)
for the model's terms.
