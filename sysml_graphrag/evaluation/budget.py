"""Measured evidence selection and complete chat-input accounting.

Only a generator's verified chat tokenizer is accepted. No character estimates,
invisible truncation, or ground-truth labels are used to select evidence.
"""
from dataclasses import dataclass

RANKED_PACKING = 'ranked_whole_units_v1'
COMPOSITION_PACKING = 'composition_bundle_v2'


@dataclass(frozen=True)
class Budget:
    evidence_tokens: int
    context_window: int
    output_reserve: int

    def __post_init__(self):
        if any(type(value) is not int or value <= 0 for value in
               (self.evidence_tokens, self.context_window, self.output_reserve)):
            raise ValueError('Budgets must be positive integers.')
        if self.output_reserve >= self.context_window:
            raise ValueError('Output reserve must leave space for input.')


class ChatTokenCounter:
    def __init__(self, tokenizer):
        if not getattr(tokenizer, 'chat_template', None):
            raise ValueError('A verified generator chat template is required.')
        self.tokenizer = tokenizer

    def text(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def messages(self, messages):
        return len(self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_dict=False))


SYSTEM_PROMPT = (
    'Answer the original SysML v2 question using only the supplied evidence. '
    'Cite source paths and lines for factual claims. Distinguish literal references '
    'from resolved instances. If evidence is insufficient, say so without claiming '
    'the source model lacks the information. Ask for scope when identifiers are '
    'ambiguous. Candidate impacts mean review obligations under explicit modeled '
    'relations, not predicted failures; shared ownership alone is not a dependency.'
)


def messages_for(query, evidence):
    return [{'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': f'QUESTION:\n{query}\n\nEVIDENCE:\n{evidence}'}]


def pack_evidence(query, units, counter, budget, *, priority_unit_ids=()):
    """Reserve a complete optional group before greedy whole-unit packing.

    A graph edge unit includes both endpoint identities and provenance; a text
    unit is a whole declaration chunk. Selection never reads answer labels.
    """
    if counter.messages(messages_for(query, '')) + budget.output_reserve > budget.context_window:
        raise ValueError('System prompt/question plus output reserve exceed the context window.')

    def fits(rows):
        trial = '\n\n'.join(row['text'] for row in rows)
        return (counter.text(trial) <= budget.evidence_tokens
                and counter.messages(messages_for(query, trial)) + budget.output_reserve <= budget.context_window)

    selected, omitted = [], []
    requested = set(priority_unit_ids)
    if requested:
        units = list(units)
    priority = [unit for unit in units if unit['id'] in requested] if requested else []
    reserved = set()
    if requested and {unit['id'] for unit in priority} == requested and fits(priority):
        selected.extend(priority)
        reserved = requested
    for unit in units:
        if unit['id'] in reserved:
            continue
        if fits(selected + [unit]):
            selected.append(unit)
        else:
            omitted.append(unit['id'])
    evidence = '\n\n'.join(row['text'] for row in selected)
    messages = messages_for(query, evidence)
    result = {'messages': messages, 'evidence': evidence,
            'selected_unit_ids': [row['id'] for row in selected],
            'omitted_unit_ids': omitted, 'selection_overflow': bool(omitted),
            'evidence_tokens': counter.text(evidence),
            'input_tokens': counter.messages(messages),
            'output_reserve': budget.output_reserve,
            'evidence_allowance': budget.evidence_tokens,
            'context_window': budget.context_window,
            'complete_evidence': None}
    if requested:
        result['reserved_priority_unit_ids'] = [unit['id'] for unit in priority] if reserved else []
    return result
