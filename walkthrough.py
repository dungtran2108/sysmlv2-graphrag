"""Evidence-only comparison on the pinned upstream Flashlight model."""
import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

import networkx as nx
import torch
from sentence_transformers import SentenceTransformer

from comparators import EvidenceIndex, declaration_chunks

ROOT = Path(__file__).resolve().parent
SOURCE_NAME = "Flashlight Example.sysml"
SOURCE_HASH = "0b90c2fa75f7ff82495418c3bc366802c6646f8618878845af0bc3975286f8f9"
JAR_HASH = "b1ad9d64b1f0c75730facf25a5e2856bc9df2bb4bd39476df2fdf5ae68cd9350"
ENCODER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
QUESTIONS = (
    "How is the user connected to the flashlight in this model? Give the full port path and declared port type at each end.",
    "How does the `userToFlashlight` interface connect the user to the flashlight? Give the full port path and declared port type at each end.",
)
FACTS = {
    "user path": (22, "userToFlashlight connect user.onOffCmdPort to flashlight.onOffCmdPort"),
    "flashlight path": (22, "userToFlashlight connect user.onOffCmdPort to flashlight.onOffCmdPort"),
    "user type": (16, "port onOffCmdPort: OnOffCmdPort;"),
    "flashlight type": (27, "port onOffCmdPort: ~OnOffCmdPort;"),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def refs(row, field):
    value = row["fields"].get(field, [])
    return value if isinstance(value, list) else [value]


def project(export):
    """Project source-backed declaration records and resolved endpoint links."""
    rows = {row["id"]: row for row in export["elements"]}
    graph = nx.DiGraph()

    def children(row):
        return [rows[ref["id"]] for key in ("ownedRelationship", "ownedRelatedElement")
                for ref in refs(row, key) if ref["id"] in rows]

    def descendants(row):
        yield row
        for child in children(row):
            yield from descendants(child)

    for row in rows.values():
        if not row["source_backed"] or row["class"] not in {
            "Package", "PartUsage", "PortUsage", "PortDefinition", "InterfaceUsage",
            "AttributeDefinition", "ReferenceUsage", "ActionUsage", "SuccessionFlowUsage",
        }:
            continue
        # Keep the first declaration line, with its compiler-qualified identity.
        source = row["source"].strip().splitlines()[0].strip()
        key = row["source_key"]
        name = row["fields"].get("qualifiedName") or row.get("declaredName") or key
        types = [ref for ref in refs(row, "type") if not ref.get("proxy")]
        graph.add_node(key, name=name, line=row["line"], source=source, kind=row["class"],
                       compiler_types=[ref["class"] for ref in types] if row["class"] == "PortUsage" else [])

    for row in rows.values():
        key = row["source_key"]
        if key not in graph:
            continue
        for ref in refs(row, "owner"):
            owner = ref.get("source_key")
            if owner in graph:
                graph.add_edge(owner, key, relation="owns")
        for ref in refs(row, "type"):
            target = ref.get("source_key")
            if target in graph:
                graph.add_edge(key, target, relation="typed_by")
        if row["class"] != "InterfaceUsage":
            continue
        for membership in children(row):
            if membership["class"] != "EndFeatureMembership" or not membership["source_backed"]:
                continue
            for subset in descendants(membership):
                if subset["class"] != "ReferenceSubsetting" or not subset["source_backed"]:
                    continue
                chain = [ref for item in descendants(subset) if item["class"] == "FeatureChaining"
                         for ref in refs(item, "chainingFeature")]
                if not chain or chain[-1].get("proxy"):
                    continue
                terminal = chain[-1].get("source_key")
                if terminal in graph and graph.nodes[terminal]["kind"] == "PortUsage":
                    graph.add_edge(key, terminal, relation="endpoint", path=subset["source"].strip())
    return graph


def record(graph, key):
    node = graph.nodes[key]
    return f"{node['name']} [source: {SOURCE_NAME}:{node['line']}]\n{node['source']}"


def neighborhood(graph, center):
    keys = {center, *graph.predecessors(center), *graph.successors(center)}
    subgraph = graph.subgraph(keys)
    relations = [f"{graph.nodes[u]['name']} --{data['relation']}--> {graph.nodes[v]['name']}"
                 for u, v, data in sorted(subgraph.edges(data=True))]
    return {"id": center, "text": "\n".join([*(record(graph, key) for key in sorted(keys)), *relations])}


def interface_record(graph, key):
    ends = [(target, edge["path"]) for _, target, edge in graph.out_edges(key, data=True)
            if edge["relation"] == "endpoint"]
    lines = [record(graph, key)]
    for target, path in sorted(ends, key=lambda pair: pair[1]):
        port = graph.nodes[target]
        lines.append(f"Written endpoint {path}")
        # The compiler supplies the type reference. The cited declaration keeps
        # its exact spelling, including the conjugation marker.
        if len(port["compiler_types"]) == 1:
            lines.append(record(graph, target))
    return {"id": key, "text": "\n".join(lines)}


def pack(ranked, tokenizer, budget=1024):
    selected, omitted = [], []

    def render(records):
        return "\n\n".join(record["text"] for record in records)

    def tokens(text):
        return len(tokenizer.encode(text, add_special_tokens=False, truncation=False, verbose=False))

    for item in ranked:
        if tokens(render([*selected, item])) <= budget:
            selected.append(item)
        else:
            omitted.append(item["id"])
    return selected, omitted, tokens(render(selected))


def score(ranked, packed):
    result = {}
    for name, (line, statement) in FACTS.items():
        citation = re.compile(rf"{re.escape(SOURCE_NAME)}:{line}(?:-{line})?(?![0-9])")

        def supports(item):
            # Each eligible record must contain both the cited span and its
            # exact declaration. The fixture has one declaration per fact span.
            return bool(citation.search(item["text"]) and statement in item["text"])

        ranks = [rank for rank, item in enumerate(ranked, 1) if supports(item)]
        witnesses = [item["id"] for item in packed if supports(item)]
        result[name] = {"status": "packed" if witnesses else "packing omission" if ranks else "absent from candidates",
                        "candidate_ranks": ranks, "packed_ids": witnesses}
    return result


def run(source, pilot, scratch, encoder=None):
    source, pilot, scratch = (Path(p).resolve() for p in (source, pilot, scratch))
    if scratch.is_relative_to(ROOT):
        raise ValueError("Scratch must be outside the checkout.")
    jar = pilot / "jupyter-sysml-kernel-0.62.0-all.jar"
    stdlib = pilot / "sysml.library"
    source_bytes = source.read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(source_bytes).hexdigest() != SOURCE_HASH or digest(jar) != JAR_HASH:
        raise ValueError("Pinned source or compiler JAR hash mismatch.")
    expected_library = json.loads((ROOT / "stdlib-sha256.json").read_text(encoding="utf-8"))
    actual_library = {p.relative_to(stdlib).as_posix(): digest(p) for p in stdlib.rglob("*")
                      if p.suffix in {".sysml", ".kerml"}}
    if actual_library != expected_library:
        raise ValueError("Pinned standard library inventory mismatch.")
    scratch.mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix="flashlight-", dir=scratch))
    inputs = folder / "source"
    inputs.mkdir()
    (inputs / SOURCE_NAME).write_bytes(source_bytes)
    export_path = folder / "model.json"
    java_version = subprocess.run(["java", "-version"], capture_output=True, text=True, check=True).stderr
    if not re.search(r'version "21(?:\.|")', java_version):
        raise ValueError("Use Java 21.")
    with (folder / "compiler.log").open("w", encoding="utf-8") as log:
        subprocess.run(["java", "-Xmx2g", "-cp", str(jar), str(ROOT / "CompilerExport.java"),
                        str(inputs), str(export_path), str(stdlib)], stdout=log, stderr=log, check=True)
    export = json.loads(export_path.read_text(encoding="utf-8"))
    if export["validation"] or export["property_errors"] or export["library_resource_errors"]:
        raise ValueError(f"Compiler validation failed. Inspect {export_path}")
    graph = project(export)
    if len(graph) == 0:
        raise ValueError("No source-backed declarations were projected.")
    torch.set_num_threads(4)
    model = SentenceTransformer(str(encoder) if encoder else "sentence-transformers/all-MiniLM-L6-v2",
                                revision=None if encoder else ENCODER_REVISION, device="cpu",
                                local_files_only=bool(encoder))
    texts = declaration_chunks({SOURCE_NAME: source.read_text(encoding="utf-8")})
    neighborhoods = [neighborhood(graph, key) for key in sorted(graph)]
    interfaces = [interface_record(graph, key) for key, node in sorted(graph.nodes(data=True))
                  if node["kind"] == "InterfaceUsage"]
    indexes = {
        "text_hybrid": EvidenceIndex(texts, model, hybrid=True),
        "one_hop": EvidenceIndex(neighborhoods, model),
        "graph_bounded": EvidenceIndex(interfaces, model, hybrid=True),
    }
    conditions = []
    for question in QUESTIONS:
        for method, index in indexes.items():
            ranked = index.rank(question)[:100]
            packed, omitted, tokens = pack(ranked, model.tokenizer)
            facts = score(ranked, packed)
            conditions.append({"question": question, "method": method, "evidence_tokens": tokens,
                               "pass": all(fact["status"] == "packed" for fact in facts.values()),
                               "facts": facts, "ranked": ranked, "packed": packed, "omitted_ids": omitted})
    report = {"case": "flashlight-interface", "budget": 1024, "tokenizer": "all-MiniLM-L6-v2",
              "encoder_revision": ENCODER_REVISION, "source_sha256_lf": SOURCE_HASH,
              "export_sha256": digest(export_path), "compiler_jar_sha256": digest(jar),
              "java_version": java_version, "conditions": conditions}
    path = folder / "report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--pilot", required=True, type=Path)
    parser.add_argument("--scratch", required=True, type=Path)
    parser.add_argument("--encoder", type=Path, help="Local copy of the pinned MiniLM snapshot")
    print(run(**vars(parser.parse_args())))

