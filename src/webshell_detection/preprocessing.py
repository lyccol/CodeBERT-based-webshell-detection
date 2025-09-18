"""Utilities for cleaning and normalising PHP source files."""

from __future__ import annotations

import re

__all__ = [
    "strip_php_tags",
    "remove_php_comments",
    "collapse_whitespace",
    "preprocess_php_code",
    "code_pre",
]

_COMMENT_BLOCK_RE = re.compile(r"/\*.*?\*/", re.S)
_PHP_TAG_RE = re.compile(r"<\?php|<\?|\?>", re.IGNORECASE)
_MULTIPLE_WHITESPACE_RE = re.compile(r"\s+")


def strip_php_tags(text: str) -> str:
    """Remove PHP open and close tags from *text*.

    Parameters
    ----------
    text:
        Source code to clean.
    """

    return _PHP_TAG_RE.sub(" ", text)


def remove_php_comments(text: str) -> str:
    """Remove single and multi-line PHP comments from *text*."""

    text = _COMMENT_BLOCK_RE.sub(" ", text)
    text = re.sub(r"(?://|#).*", " ", text)
    return text


def collapse_whitespace(text: str) -> str:
    """Collapse repeated whitespace into a single space and strip the result."""

    return _MULTIPLE_WHITESPACE_RE.sub(" ", text).strip()


def preprocess_php_code(text: str, *, max_characters: int | None = 10000) -> str:
    """Apply a sequence of normalisation steps to PHP *text*.

    The normalisation removes PHP tags, comments and collapses whitespace.
    If *max_characters* is provided the output will be truncated to at most
    that many characters. This mirrors the common preprocessing performed in
    webshell detection datasets where very long payloads add little signal.
    """

    for fn in (strip_php_tags, remove_php_comments, collapse_whitespace):
        text = fn(text)
    if max_characters is not None:
        text = text[:max_characters]
    return text


# ``code_pre`` was the name used in the original project.  Keep it as a public
# alias so that older scripts keep working while encouraging users to migrate to
# :func:`preprocess_php_code`.
code_pre = preprocess_php_code
