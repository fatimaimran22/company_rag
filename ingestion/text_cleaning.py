"""Pure string cleaning. No PyMuPDF here: every function takes and returns text.

clean_text() runs the steps in order; each step is a separate function so you
can try them one at a time in a Python shell.
"""

import re

ZERO_WIDTH_SPACE = "\u200b"

# Bullet glyphs at the start of a line -> Markdown-style "-".
# Order matters: the Word sub-bullet "o" is only recognised when it is glued
# to a zero-width space ("o\u200bCPU"), so it must run before ZWSPs are removed;
# otherwise ordinary words starting with "o" would be at risk.
_BULLETS = [
    (re.compile(r"^[ \t]*o\u200b[\s\u200b]*", re.M), "  - "),
    (re.compile(r"^[ \t]*○[\s\u200b]*", re.M), "  - "),
    (re.compile(r"^[ \t]*[●•▪][\s\u200b]*", re.M), "- "),
]

_CHAR_MAP = str.maketrans(
    {
        ZERO_WIDTH_SPACE: " ",  # "1.\u200bPurpose" -> "1. Purpose"
        "\u200c": None,  # zero-width non-joiner
        "\u200d": None,  # zero-width joiner
        "\u2060": None,  # word joiner
        "\ufeff": None,  # byte-order mark
        "\u00ad": None,  # soft hyphen
        "\u00a0": " ",  # non-breaking space
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
    }
)

# A line that starts a new item and must NOT be glued onto the line above it.
_STARTS_NEW_ITEM = re.compile(
    r"""^(
        \s*-\s                              # bullet  "- " / "  - "
      | \|                                  # Markdown table row
      | \d+\.(\d+\.?)*\s                    # numbered heading/clause "1. ", "1.0 ", "3.2.1.1. "
      | \d+-[A-Z]                           # "1-You are required..." (not "0-4 hours")
      | [a-z]\.\s                           # "a. MDM Enrollment"
      | Part\s\d+[.:]                       # Working Hours sub-heading "Part 3. In case..."
      | [A-Z][A-Za-z ()/&.-]{0,30}:(\s|$)   # label "Note:", "Half Day:", "Part 1:"
    )""",
    re.X,
)


def normalize_bullets(text: str) -> str:
    for pattern, replacement in _BULLETS:
        text = pattern.sub(replacement, text)
    return text


def normalize_characters(text: str) -> str:
    return text.translate(_CHAR_MAP)


def normalize_whitespace(text: str) -> str:
    """Collapse runs of spaces/tabs, strip line ends; keep sub-bullet indentation."""
    out = []
    for line in text.split("\n"):
        indent = "  " if line.startswith("  - ") else ""
        out.append(indent + re.sub(r"[ \t]+", " ", line).strip())
    return "\n".join(out)


def unwrap_lines(text: str) -> str:
    """Rejoin lines that the PDF wrapped only because the page was too narrow."""
    out: list[str] = []
    for line in text.split("\n"):
        prev = out[-1] if out else ""
        if (
            prev.strip()
            and line.strip()
            and (prev.strip() == "-" or not _STARTS_NEW_ITEM.match(line))
            and not prev.endswith(":")
            and not prev.lstrip().startswith("|")
        ):
            out[-1] = prev + " " + line.lstrip()
        else:
            out.append(line)
    return "\n".join(out)


def collapse_blank_lines(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def clean_text(text: str) -> str:
    text = normalize_bullets(text)
    text = normalize_characters(text)
    text = normalize_whitespace(text)
    text = unwrap_lines(text)
    return collapse_blank_lines(text)
