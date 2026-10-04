"""Make each chunk understandable on its own before it is embedded.

A chunk is embedded (and later retrieved) WITHOUT its neighbours. A chunk that
starts with "- The mentioned allowance will be effective..." or with a table
row does not say which policy it belongs to. So every chunk gets an
`embedding_text`:

    BYOM Allowance (Bring Your Own Machine) > 4. Allowance Entitlement Structure

    <the chunk's text, unchanged>

`text` stays the plain content (for display and citations); `embedding_text`
is what the embedding model will see. Version and effective date are kept in
metadata only: they rarely help match a question, and would add the same
noise to every chunk of a document.
"""

def context_line(doc: str, title: str | None, section: str) -> str:
    """Breadcrumb with the document's full title added when it says more than the short name.

    "BYOM Allowance > 4. Allowance..." -> "BYOM Allowance (Bring Your Own Machine) > 4. Allowance..."
    """
    if title and title.lower() != doc.lower() and section.startswith(doc):
        return f"{doc} ({title}){section[len(doc):]}"
    return section


def embedding_text(doc: str, title: str | None, section: str, text: str) -> str:
    return f"{context_line(doc, title, section)}\n\n{text}"


def table_info(text: str, row_split: bool) -> list[dict]:
    """Describe the Markdown tables inside a chunk: columns, row count, row-split or whole."""
    tables, current = [], []
    for line in text.split("\n") + [""]:
        if line.startswith("|"):
            current.append(line)
        elif current:
            cells = lambda row: [c.strip() for c in row.strip().strip("|").split("|")]
            tables.append(
                {
                    "columns": cells(current[0]),
                    "rows": max(len(current) - 2, 0),  # minus header and "---" lines
                    "row_split": row_split,
                }
            )
            current = []
    return tables
