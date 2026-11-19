# Third-party inputs and attribution

- The Flashlight walkthrough uses the external Systems Modeling
  [SysML v2 Pilot Implementation](https://github.com/Systems-Modeling/SysML-v2-Pilot-Implementation)
  and its Flashlight example and standard library under
  [EPL-2.0](https://www.eclipse.org/legal/epl-2.0/). Follow the pinned download
  instructions in the README. Compiler JARs, models and libraries remain
  separate downloads under their upstream terms.
- The separately downloaded
  [all-MiniLM-L6-v2 encoder](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
  uses Apache-2.0. The separately supplied
  [Llama 3.1 chat tokenizer](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
  comes with Meta's Llama 3.1 terms. The snapshot includes neither model
  weights nor tokenizer files.
- The installation lists reference Python dependencies. Those packages are
  distributed through their upstream channels under their own licenses.
- The Flashlight observer, lexical retrieval and ranking code and the
  candidate routing, ranking, packing and rendering port derive from the
  GraphRAG–SysML v2 research project. The assembly fixture was made for
  this walkthrough.
