# DocGround

DocGround is an evaluation-first retrieval-augmented generation project for scikit-learn documentation. Milestone 1 converts a local HTML snapshot into structured JSON for later use.

## Reproduce Milestone 1

Use Python 3.14 and install dependencies with `python -m pip install -r requirements.txt`. Place the scikit-learn 1.9.1 chapter at `data/raw/common_pitfalls.html` and run these commands from the project root with the virtual environment active.

```console
python -m pytest tests -q
python -m src.ingestion data/raw/common_pitfalls.html data/processed/common_pitfalls.json --expected-sha256 21FAC11C181AB35624DCF8673F0475D403D4F6ECF0F9BEAD01A573D7642B515C
```

The HTML declares the canonical source below. The `/stable/` URL can move to a newer release. The command checks the SHA-256 of the raw bytes before processing. The original snapshot is needed to reproduce the artifact. Both data directories are Git-ignored. [Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html)

## JSON schema version 1

| Field | Contents |
| --- | --- |
| `source` | Local filename, canonical or supplied URL, HTML title, documentation version when present in the scikit-learn title, and raw SHA-256 |
| `processing` | UTF-8 encoding, HTML parser name, and BeautifulSoup version |
| `sections` | Sections in heading order, with `id`, `parent_id`, `heading`, `level`, `heading_path`, `anchors`, and ordered `blocks` |

Paragraph and code blocks contain `text`. Lists store an `ordered` flag and an array of items, each containing blocks. Tables store a caption and rows with their group and cells. Each cell contains blocks, a header flag, and any explicit spans or `scope`. Collapsible panels store a summary, anchor, and nested blocks. Introductory text without a heading receives an untitled section.

Section IDs hash the source URL and preferred HTML anchor. For headings without anchors, the input uses the parent section ID, heading level and heading text. An occurrence counter disambiguates duplicates. Prose whitespace is normalized. Code retains the decoded text of `<pre>` elements, including indentation, prompts and displayed output.

The parser accepts recoverable HTML. It targets the main article and preserves textual content. Inline formatting, hyperlink destinations, images, and definition-list structure are outside this schema. The pinned chapter has no tables, so the suite tests table handling with a synthetic fixture.

## Milestone 2A — Chunking and evidence coordinates

Run once per processed schema-v1 document. Both file paths are required. The only configurable setting is `--max-chars`, a positive integer with a default of 1,200.

```console
python -m src.chunking data/processed/common_pitfalls.json data/processed/common_pitfalls.chunks.json --max-chars 1200
```

**Canonical text.** Traverse sections and blocks in stored array order. Emit paragraph and code text verbatim. Visit list items in order, a panel's summary before its contents, and a table's caption before its cells in row order. Skip absent or empty summaries and captions. Join the emitted strings with exactly two LF characters, including between sections. There is no added leading or trailing separator. Headings are metadata. A section with no emitted text contributes no chunks. The representation follows the processed JSON order, which can differ from the original HTML reading order.

**Packing.** Within a section, append a whole text field when the resulting chunk fits, counting the separator. Otherwise start a new chunk. Any single oversized field gets its own chunk, including long paragraphs and code listings. The limit counts Python Unicode code points. Blocks can share a chunk, and compound blocks can span several chunks. Packing has zero overlap. The canonical document text can be reconstructed by joining chunk texts with the same two LF characters.

### Chunk schema v1

The artifact stores `chunk_schema_version`, `max_chars`, and ordered `chunks`.

| Chunk field | Meaning |
| --- | --- |
| `document_id` | Revision-specific SHA-256 of the entire processed document, serialized as UTF-8 JSON with sorted keys, compact separators and literal Unicode. JSON file indentation has no effect. Changes to processing metadata create a new revision. |
| `source_sha256`, `source_url` | Raw HTML digest and documentation URL from ingestion. |
| `section_id`, `heading_path`, `anchor` | Owning section, heading hierarchy, and first section anchor or null. |
| `start`, `end`, `text` | Zero-based, half-open document character range and its exact canonical substring. The join between adjacent chunks occupies two characters outside both chunks. |
| `fields` | Ordered mappings with `path`, `start`, `end`, and `anchor`. The path is a JSON Pointer relative to the owning section, such as `/blocks/0/items/1/0/text`. Each range uses document coordinates. Panel fields inherit the closest available panel anchor. |
| `id` | SHA-256 of the compact UTF-8 JSON array `["chunk-v1", document_id, section_id, start, end]`. A range that survives repacking keeps its ID. |

### Evidence matching contract

Annotate each required evidence unit with `document_id`, `section_id`, `start`, and `end` in the canonical document. To translate a quote at local offsets `[a, b)` in a mapped JSON field, add that field's `start` to both offsets. Character offsets refer exclusively to the processed representation.

`evidence_coverage(required_units, chunks)` returns one Boolean per required unit in input order. Credit requires **one chunk to contain the entire nonempty span** in the same document revision and section. Its first and last characters must belong to mapped text fields. A unit can cross fields inside one chunk. Splitting that unit across two returned chunks earns zero credit for it. Shared words or headings earn no credit on their own.

For a question requiring multiple units, use `all(flags)` for full coverage or `sum(flags) / len(flags)` for the fraction covered. Different chunks may cover different units. Repeating a chunk gives no additional credit. This rule measures coverage of the annotated locations. Equivalent passages elsewhere require separate annotation policy. Table layout, list markers, and heading strings are excluded from the canonical body. Raw HTML byte positions and token counts are outside this coordinate system.
