# SysML v2 evidence retrieval

[Candidate walkthrough](CANDIDATE.md) · [Release license proposal](LICENSE-PROPOSAL.md)

## SysML v2 Flashlight evidence walkthrough

One engineering case with two phrasings, each run independently with hybrid text
(BM25 and MiniLM with reciprocal-rank fusion), dense one-hop neighborhoods, and
bounded compiler-graph retrieval. All three use the same greedy whole-record packer
with a 1,024-token cap measured using the pinned MiniLM tokenizer. The simplified
projection and tokenizer differ from those used in the original research harness.

## Run

Use Python 3.13 and Java 21. In an external virtual environment install
`python -m pip install -r requirements.txt`. Install JupyterLab in that environment
to open `Flashlight.ipynb` from the checkout root, or use the command below.

- Download the unchanged [Flashlight source](https://github.com/Systems-Modeling/SysML-v2-Pilot-Implementation/blob/287bd59befe4971eb24bc33c1c46361a0ea14005/sysml/src/examples/Flashlight%20Example/Flashlight%20Example.sysml) using the Raw link.
- Extract the [Pilot 0.62.0 ZIP](https://github.com/Systems-Modeling/SysML-v2-Pilot-Implementation/releases/download/2026-08/jupyter-sysml-kernel-0.62.0.zip)
  to an external directory. ZIP SHA-256 is
  `5a015ed3f2b3d2dae14e9c3fb9602d9e1fef5f9ce33cf396c422329405cc3cb2`.
  Pass its extracted `sysml` directory. The runner checks source, JAR and library hashes.
- MiniLM downloads at [revision 1110a24](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41).
  For an offline run, append `--encoder <local-copy-of-that-snapshot>`.
  Keep model caches outside the checkout.

```text
python -B walkthrough.py --source <source-file> --pilot <extracted-sysml-directory> --scratch <external-directory>
```

The report, compiler log and export go under scratch. The notebook shows all six
scores and lets readers inspect any condition's ranked candidates and packed
records. The four facts require the cited interface and both endpoint declarations.
The scorer operates after retrieval and packing.

## Attribution

The model, compiler and library are from the [Systems Modeling SysML v2 Pilot
Implementation](https://github.com/Systems-Modeling/SysML-v2-Pilot-Implementation)
under [EPL-2.0](https://github.com/Systems-Modeling/SysML-v2-Pilot-Implementation/blob/287bd59befe4971eb24bc33c1c46361a0ea14005/LICENSE).
Preserve upstream notices and license on redistribution. MiniLM is Apache-2.0.
The observer, lexer and ranking code derive from the local GraphRAG–SysML v2 research
project. The repo includes neither upstream binaries nor a prebuilt export.
