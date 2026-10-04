"""Find a document's headings and build its section tree.

Two heading styles exist in our PDFs:

* "sop" documents (BYOM, Night Shift, Test Project) use numbered headings
  ("1. Purpose:", "1.0 SCOPE:", "3.2.1.1. When a candidate...") and, in BYOM,
  lettered sub-sections ("a. Initiation").
* "plain" documents (Working Hours) use short title lines ("Extra Working Days")
  with "Part 1: ..." sub-sections.

Numbered lines are only accepted as headings when the number is a valid next
step in the outline. That rejects lists that merely look like headings, such
as the "Notes: 1. ... 2. ... 3. ..." list inside Test Project section 3.2.1.
"""

import re
from dataclasses import dataclass, field

from config import DOCUMENTS
from .document import DocStream, _lines_with_offsets

TITLE_MAX = 70  # longer heading lines are labelled by number only ("4.1")

_NUMBERED = re.compile(r"^(\d+(?:\.\d+)*\.?)\s+(\S.*)$")  # "3.2.1.1. When..." / "1.0 SCOPE:"
_LETTERED = re.compile(r"^([a-z])\.\s+(\S.*)$")  # "b. Approval, Verification..."
_PART = re.compile(r"^Part\s+\d+\b")  # "Part 2: Weekly minimum hours..."


@dataclass
class Section:
    label: str  # how it appears in a breadcrumb, e.g. "4. Allowance Entitlement Structure"
    start: int  # offset of the heading line in DocStream.text
    end: int = 0  # where the next heading (of any level) starts: this section's OWN text
    parent: "Section | None" = None
    children: list["Section"] = field(default_factory=list)

    @property
    def end_full(self) -> int:
        """End of this section including all its sub-sections."""
        return max([self.end] + [c.end_full for c in self.children])

    def path(self) -> list["Section"]:
        node, out = self, []
        while node is not None:
            out.append(node)
            node = node.parent
        return out[::-1]


@dataclass
class Outline:
    stream: DocStream
    root: Section  # label = document name; its children are top-level sections
    flat: list[Section]  # every heading in document order

    def breadcrumb(self, section: Section) -> str:
        return " > ".join(s.label for s in section.path())

    def section_at(self, offset: int) -> str:
        """Breadcrumb of the section containing a character offset."""
        cover = self.stream.cover
        if cover and offset < self.stream.content_start:
            return f"{self.root.label} > Document Information"
        current = self.root
        for s in self.flat:
            if s.start <= offset:
                current = s
            else:
                break
        return self.breadcrumb(current)


def _label(number: str, title: str) -> str:
    title = title.strip().rstrip(":").strip()
    return f"{number} {title}" if len(title) <= TITLE_MAX else number


def _display_number(token: str) -> str:
    """'3.2.' -> '3.2', '1.0' -> '1.0', but a single level keeps its dot: '4.'"""
    return token.rstrip(".") if "." in token[:-1] else token


def _part_label(line: str) -> str:
    """'Part 1: Minimum Daily Hours Requirement' (or just 'Part 3' if very long)."""
    line = line.strip().rstrip(":").strip()
    return line if len(line) <= TITLE_MAX else _PART.match(line).group()


def _number_key(token: str) -> tuple[int, ...]:
    """'4.1.' -> (4, 1); '1.0' -> (1,), so that 4.1 becomes a child of 4.0."""
    parts = [int(p) for p in token.strip(".").split(".")]
    while len(parts) > 1 and parts[-1] == 0:
        parts.pop()
    return tuple(parts)


def _is_next_number(current: tuple[int, ...], candidate: tuple[int, ...]) -> bool:
    """Is `candidate` a legal next heading after `current` in a numbered outline?

    Legal: first child (3.2 -> 3.2.1) or next sibling at any level
    (3.2.1.3 -> 3.2.1.4, 3.2.2, 3.3 or 4).
    """
    if not current:
        return candidate == (1,)
    if candidate == current + (1,):
        return True
    n = len(candidate)
    return n <= len(current) and candidate[:-1] == current[: n - 1] and candidate[-1] == current[n - 1] + 1


def _is_title_line(lines: list[str], i: int) -> bool:
    """A short, standalone line with no closing punctuation, e.g. 'Extra Working Days'."""
    line = lines[i]
    standalone = (i == 0 or not lines[i - 1].strip()) and (i + 1 >= len(lines) or not lines[i + 1].strip())
    return (
        standalone
        and 0 < len(line.split()) <= 6
        and line[0].isupper()
        and line[-1] not in ".:;,?!"
        and "|" not in line
        and not _PART.match(line)
    )


def build_outline(stream: DocStream) -> Outline:
    template = next((c.template for c in DOCUMENTS.values() if c.name == stream.doc), "plain")
    root = Section(stream.doc, stream.content_start)
    flat: list[Section] = []

    def add(section: Section, parent: Section):
        section.parent = parent
        parent.children.append(section)
        flat.append(section)

    numbered: dict[tuple[int, ...], Section] = {}  # sop: number key -> section
    current_number: tuple[int, ...] = ()
    last_letter: Section | None = None
    last_title: Section | None = None  # plain: current level-1 heading

    rows = list(_lines_with_offsets(stream.text, stream.content_start, len(stream.text)))
    texts = [line for _, _, line in rows]
    for i, (offset, _, line) in enumerate(rows):
        if template == "sop":
            if m := _NUMBERED.match(line):
                key = _number_key(m.group(1))
                if _is_next_number(current_number, key):
                    parent = next((numbered[key[:k]] for k in range(len(key) - 1, 0, -1) if key[:k] in numbered), root)
                    section = Section(_label(_display_number(m.group(1)), m.group(2)), offset)
                    add(section, parent)
                    numbered[key] = section
                    current_number, last_letter = key, None
            elif (m := _LETTERED.match(line)) and current_number:
                letter = m.group(1)
                expected = "a" if last_letter is None else chr(ord(last_letter.label[0]) + 1)
                if letter == expected:
                    section = Section(_label(f"{letter}.", m.group(2)), offset)
                    add(section, numbered[current_number])
                    last_letter = section
        else:
            if _PART.match(line):
                add(Section(_part_label(line), offset), last_title or root)
            elif _is_title_line(texts, i):
                last_title = Section(line.strip(), offset)
                add(last_title, root)

    # Each section's own text runs until the next heading of any level.
    for section, nxt in zip(flat, flat[1:] + [None]):
        section.end = nxt.start if nxt else len(stream.text)
    root.end = flat[0].start if flat else len(stream.text)
    return Outline(stream, root, flat)
