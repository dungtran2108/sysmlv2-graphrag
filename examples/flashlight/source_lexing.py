"""Shared lexical boundaries for the supported source projection and text chunks.

This is not a SysML parser. Quotes/comments protect delimiters and source offsets
are preserved, so graph projection and raw-text retrieval can cite original lines.
"""
import re

LEXEMES = re.compile(r'/\*[\s\S]*?\*/|//[^\n]*|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[{};]')


def clean_comments(text):
    def replace(match):
        token = match.group()
        if token.startswith(('/*', '//')):
            return ''.join('\n' if char == '\n' else ' ' for char in token)
        return token
    return LEXEMES.sub(replace, text)


def segments(text):
    """Split delimiters outside quotes, retaining source line offsets."""
    start = 0
    for match in LEXEMES.finditer(text):
        if match.group() in {'{', '}', ';'}:
            raw = text[start:match.start()]
            offset = start + len(raw) - len(raw.lstrip())
            yield raw.strip(), match.group(), text[:offset].count('\n') + 1
            start = match.end()
