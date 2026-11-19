"""Decode admitted tree records for a relationship and locator equivalence check."""
import re
from collections import defaultdict

NODE = re.compile(r'^( *)(?:└── )?(Reference: )?\[([^\]]+)\] (.*)$')
BRANCH = re.compile(r'^( *)├── \(([^)]+)\)$')
SOURCE = re.compile(r'\[source: ([^\]]+)\]')



def name(data, node):
    return data.get('qualified_name',data.get('name',str(node)))


def locator(data):
    if data.get('file') and data.get('line') is not None:
        return f"{data['file']}:{data['line']}"
    return None


def parse_tree(text, graph):
    lines = text.splitlines()
    names = defaultdict(list)
    for n,data in graph.nodes(data=True):
        names[name(data,n)].append(n)
    levels, pending, definitions = {}, {}, {}
    observed = []
    for i,line in enumerate(lines,1):
        if branch := BRANCH.fullmatch(line):
            depth = len(branch[1])//2
            pending[depth+1] = (levels[depth-1],branch[2],i)
            continue
        node = NODE.fullmatch(line)
        if not node:
            continue
        depth = len(node[1])//2
        ident = node[4]
        before_children = []
        for trailing in lines[i:]:
            if not trailing or BRANCH.fullmatch(trailing) or NODE.fullmatch(trailing):
                break
            before_children.append(trailing)
        node_loc = SOURCE.findall('\n'.join(before_children))
        if node[2]:
            ident, explicit_id = ident.rsplit(' (id: ',1)
            target = explicit_id.removesuffix(')')
            assert target in names[ident]
        else:
            candidates = [n for n in names[ident]
                          if graph.nodes[n].get('kind','Element')==node[3]
                          and (locator(graph.nodes[n]) is None or locator(graph.nodes[n]) in node_loc)]
            assert len(candidates)==1, (ident,node_loc,candidates)
            target = candidates[0]
        data = graph.nodes[target]
        assert node[3]==data.get('kind','Element')
        if node[2]:
            assert target in definitions
        else:
            expected_loc = locator(data)
            if expected_loc:
                assert expected_loc in node_loc, (target,node_loc)
            for detail in data.get('evidence_details',[]):
                assert ' '.join(('Evidence: '+detail).split()) in ' '.join('\n'.join(before_children).split())
            definitions[target] = dict(line=i, locator=expected_loc,
                locator_line=next((j for j in range(i+1,i+1+len(before_children))
                                   if expected_loc and expected_loc in lines[j-1]),None))
        levels[depth]=target
        if depth:
            origin,relation,branch_line = pending.pop(depth)
            observed.append(dict(source=origin, role=relation, target=target, tree_branch_line=branch_line,
                                 tree_target_line=i, cross_link=bool(node[2]),
                                 edge_metadata='\n'.join(lines[branch_line:i-1])))
    assert not pending
    assert set(definitions)==set(graph)
    return lines,observed,definitions
