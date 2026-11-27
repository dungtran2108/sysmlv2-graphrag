"""Resolve requirements through a component's incoming written allocations."""
import re

from sysml_graphrag.retrieval.satisfaction import (
    REJECTED, allocation_endpoints, retrieve_satisfaction, targets,
)
from sysml_graphrag.retrieval.traceability import normalized_label

_SCOPE = re.compile(r"^in (?P<scope>[^,]+),\s*", re.IGNORECASE)
_COMPOUND = re.compile(r"\b(?:and|or|not|except|also|including|excluding|without)\b", re.IGNORECASE)
_TARGET = re.compile(
    r"\b(?:assigned|allocated) to (?:the )?(?P<target>[\w.:-]+(?: [\w.:-]+)*?)"
    r"(?: (?:helps? (?:to )?meet|satisf(?:y|ies)|meet(?:s)?)|$)", re.IGNORECASE,
)
_WORK = r"(?:work|behaviou?rs?|tasks?|activities|activity)"
_COMPONENT = r"(?P<target>[\w.:-]+(?: [\w.:-]+)*?)"
_ACTION = r"(?:satisfy|satisfies|meet|meets|help(?:s)? meet)"
_PARAPHRASES = tuple(re.compile(pattern, re.IGNORECASE) for pattern in (
    rf"(?:which|what) requirements (?:are (?:met|satisfied) by|does) (?:the )?{_WORK}"
    rf" (?:that )?(?:the )?{_COMPONENT} (?:is|are) responsible for(?: {_ACTION})?",
    rf"(?:which|what) requirements (?:are (?:met|satisfied) by|does) (?:the )?{_WORK}"
    rf" (?:handled|carried out|performed) by (?:the )?{_COMPONENT}(?: {_ACTION})?",
    rf"for (?:the )?{_COMPONENT}, (?:which|what) requirements (?:does its assigned {_WORK} {_ACTION}"
    rf"|are (?:met|satisfied) by its assigned {_WORK})",
))


def recognize(question):
    question = " ".join(question.strip().rstrip("?.").split())
    scope = _SCOPE.match(question)
    text = question[scope.end():] if scope else question
    if _COMPOUND.search(text):
        return None
    scope_name = scope["scope"].strip() if scope else None
    if (re.match(r"^(?:which|what) requirements\b", text, re.IGNORECASE)
            and re.search(r"\b(?:satisfy|satisfies|satisfied|met|meet|meets)\b", text, re.IGNORECASE)):
        match = _TARGET.search(text)
        if match:
            return {"target": match["target"], "scope": scope_name}
    for pattern in _PARAPHRASES:
        match = pattern.fullmatch(text)
        if match and (not re.search(r"\brequirements does\b", text, re.IGNORECASE)
                      or re.search(rf" {_ACTION}$", text, re.IGNORECASE)):
            return {"target": match["target"], "scope": scope_name}
    return None


def label(text):
    return re.sub(r"(?<=[a-z]) (?=\d\b)", "", normalized_label(text))


def locator(data):
    if data.get("file") and data.get("line") is not None:
        return f"[source: {data['file']}:{data['line']}]"
    return None


def trace_from_endpoint(graph, endpoint):
    allocations = [node for node, _, edge in graph.in_edges(endpoint, data=True)
                   if edge["relationship"] == "target_endpoint"
                   and edge.get("resolution_status") not in REJECTED
                   and graph.nodes[node].get("compiler_class") == "AllocationUsage"]
    if len(allocations) != 1:
        return None, None, []
    allocation = allocations[0]
    endpoints = allocation_endpoints(graph, allocation)
    if not endpoints or endpoints[1] != endpoint:
        return allocation, None, []
    source, target = endpoints
    path = graph.nodes[source].get("declaration_chain") or []
    if not path or any(ref.get("proxy") or not ref.get("source_key") for ref in path):
        return allocation, None, []
    features = [node for node, data in graph.nodes(data=True)
                if data.get("declared") and path[-1]["source_key"] in data.get("compiler_witnesses", [])]
    if len(features) != 1:
        return allocation, None, []
    feature = features[0]
    assertions = [node for node, _, edge in graph.in_edges(feature, data=True)
                  if edge["relationship"] == "satisfying_feature"
                  and edge.get("resolution_status") not in REJECTED
                  and graph.nodes[node].get("compiler_class") == "SatisfyRequirementUsage"
                  and graph.nodes[node].get("is_negated") is False]
    requirements = sorted({req for assertion in assertions
                           for req in targets(graph, assertion, "satisfied_requirement")})
    records = []
    for requirement in requirements:
        _, matches, _ = retrieve_satisfaction(graph, [requirement], "satisfaction_allocation", max_paths=32)
        for record in matches:
            edges = set(map(tuple, record["edges"]))
            if {(allocation, source), (allocation, target)} <= edges and any(
                    (assertion, feature) in edges for assertion in assertions):
                records.append(record)
    return allocation, feature, records


def select_reverse_allocation(graph, question):
    request = recognize(question)
    if request is None:
        return {"status": "outside_intent"}, []
    target = request["target"]
    names = {label(target), label(re.sub(r"\s+(?:task|component|part)$", "", target, flags=re.IGNORECASE))}
    is_path = "." in target or "::" in target
    parts = {node: data for node, data in graph.nodes(data=True)
             if data.get("declared") and data.get("kind") == "part"}

    def scope_matches(value):
        scope = request["scope"]
        return scope is None or label(value) == label(scope) or label(value.rsplit("::", 1)[-1]) == label(scope)

    eligible = {node: data for node, data in parts.items()
                if label(data.get("qualified_name" if is_path else "name") or "") in names}
    by_compiler_key = {}
    for node, data in parts.items():
        for key in data.get("compiler_witnesses", []):
            by_compiler_key.setdefault(key, []).append(node)

    # Combine statements targeting one path in one scope, and clarify distinct scopes.
    matches = {}
    for allocation, endpoint, relation in graph.edges(data=True):
        data = graph.nodes[allocation]
        scope = data.get("scope")
        if (relation.get("relationship") != "target_endpoint"
                or relation.get("resolution_status") in REJECTED
                or data.get("compiler_class") != "AllocationUsage"
                or not scope or not scope_matches(scope)):
            continue
        occurrence = graph.nodes[endpoint]
        path = occurrence.get("endpoint_path")
        chain = occurrence.get("declaration_chain") or []
        if not path or not locator(occurrence) or not chain or any(
                ref.get("proxy") or not ref.get("source_key") for ref in chain):
            continue
        ids = by_compiler_key.get(chain[-1]["source_key"], [])
        if len(ids) != 1 or (ids[0] not in eligible and not (is_path and label(path) in names)):
            continue
        matches.setdefault((scope, path, ids[0]), []).append(endpoint)

    decision = dict(request=request, candidate_endpoints=[], traces=[])
    for (scope, path, component), endpoints in sorted(matches.items()):
        decision["candidate_endpoints"].append(dict(scope=scope, path=path, component=component,
                                                    endpoints=endpoints, locator=locator(graph.nodes[endpoints[0]])))
    if len(matches) > 1:
        options = "; ".join(f"{row['scope']} ({row['path']}) {row['locator']}"
                            for row in decision["candidate_endpoints"])
        return {**decision, "status": "clarification",
                "response": f"Which allocation scope do you mean? {options}."}, []
    if not matches:
        scoped = [data for data in eligible.values() if scope_matches(data.get("scope") or "")]
        if len(scoped) > 1:
            options = "; ".join(f"{data['qualified_name']} {locator(data) or ''}".rstrip() for data in scoped)
            return {**decision, "status": "clarification",
                    "response": f"Which scoped component do you mean? {options}."}, []
        component = scoped[0] if scoped and locator(scoped[0]) else None
        qualified_name = component["qualified_name"] if component else None
        prefix = f"The model declares {qualified_name} {locator(component)}. " if component else ""
        return {**decision, "status": "no_trace", "grounded_component": qualified_name,
                "response": prefix + "I could not verify the requested allocation-to-requirement trace in the available projection."}, []

    records = {}
    for endpoints in matches.values():
        for endpoint in endpoints:
            allocation, feature, paths = trace_from_endpoint(graph, endpoint)
            decision["traces"].append(dict(endpoint=endpoint, allocation=allocation, feature=feature,
                                          supported_records=[path["id"] for path in paths]))
            records.update((path["id"], path) for path in paths)
    if not records:
        return {**decision, "status": "no_trace",
                "response": "I verified the written allocation endpoint, but could not verify a scoped satisfaction trace in the available projection."}, []
    return {**decision, "status": "paths_found"}, list(records.values())
