"""Split ai_answer into sentences with Python-computed offsets."""
import re

SENTENCE_RE = re.compile(r"[^\s][^\n]*?[.!?](?=\s|$)")


def segment(text):
    """Return [(idx, start, end, sentence)]; start inclusive, end exclusive."""
    return [(i, m.start(), m.end(), m.group()) for i, m in enumerate(SENTENCE_RE.finditer(text))]
