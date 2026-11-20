"""Locate and review human-selected evidence against one processed document."""

import argparse
import json
from pathlib import Path
import sys

from src.chunking import SEPARATOR, _field, _range, chunk_document


def validate_annotations(annotations, corpus_revision, fields):
    """Return draft questions after validating schema and mapped endpoints."""
    version = _field(annotations, "annotation_schema_version", int, "")
    if type(version) is not int or version != 1:
        raise ValueError("Expected annotation schema version 1")
    if not _field(annotations, "annotation_version", str, "").strip():
        raise ValueError("annotation_version must be nonempty")
    if _field(annotations, "split", str, "") not in ("development", "frozen-test"):
        raise ValueError("split must be development or frozen-test")
    if _field(annotations, "corpus_revision", str, "") != corpus_revision:
        raise ValueError("Annotation corpus_revision does not match the corpus")
    questions = _field(annotations, "questions", list, "")
    if not questions:
        raise ValueError("Annotations must contain at least one question")

    seen = set()
    for i, question in enumerate(questions):
        path = f"/questions/{i}"
        question_id = _field(question, "id", str, path)
        text = _field(question, "question", str, path)
        if not question_id.strip() or question_id in seen or not text.strip():
            raise ValueError(f"Expected a unique nonempty ID and question text at {path}")
        seen.add(question_id)
        answerable = _field(question, "answerable", bool, path)
        units = _field(question, "evidence_units", list, path)
        if answerable != bool(units):
            raise ValueError(f"Evidence units must be present only for answerable questions at {path}")
        _validate_evidence_units(units, fields, path)
    return questions


def _validate_evidence_units(units, fields, path):
    for j, alternatives in enumerate(units):
        unit_path = f"{path}/evidence_units/{j}"
        if not isinstance(alternatives, list) or not alternatives:
            raise ValueError(f"A unit requires at least one alternative span at {unit_path}")
        for k, span in enumerate(alternatives):
            location = f"{unit_path}/{k}"
            section = _field(span, "section_id", str, location)
            start, end = _range(span, location)
            for key, offset in (("start_block_path", start), ("end_block_path", end - 1)):
                field = fields.get((section, _field(span, key, str, location)))
                if field is None or not field["start"] <= offset < field["end"]:
                    raise ValueError(f"Evidence endpoint does not match {key} at {location}")


class EvidenceDocument:
    def __init__(self, document: dict):
        chunks = chunk_document(document)["chunks"]
        self.revision = chunks[0]["document_id"]
        self.text = SEPARATOR.join(chunk["text"] for chunk in chunks)
        self.sections = {}
        self.fields = {}
        for chunk in chunks:
            section_id = chunk["section_id"]
            section = self.sections.setdefault(section_id, {
                "heading_path": chunk["heading_path"], "anchor": chunk["anchor"], "fields": [],
            })
            section["fields"].extend(chunk["fields"])
            for field in chunk["fields"]:
                self.fields[section_id, field["path"]] = field

    def select_sections(self, selector):
        if selector is None:
            return self.sections
        if selector in self.sections:
            return {selector: self.sections[selector]}
        matches = {key: section for key, section in self.sections.items()
                   if section["anchor"] == selector}
        if not matches:
            raise ValueError(f"Unknown section ID or anchor: {selector!r}")
        return matches

    def find(self, quote: str, selector=None):
        if not quote.strip():
            raise ValueError("Quote must contain a non-whitespace character")
        for section_id, section in self.select_sections(selector).items():
            fields = section["fields"]
            end_of_section = fields[-1]["end"]
            start = self.text.find(quote, fields[0]["start"], end_of_section)
            while start != -1:
                end = start + len(quote)
                first = next((f for f in fields if f["start"] <= start < f["end"]), None)
                last = next((f for f in fields if f["start"] < end <= f["end"]), None)
                if first is not None and last is not None:
                    yield {
                        "section_id": section_id, "start_block_path": first["path"],
                        "end_block_path": last["path"], "start": start, "end": end,
                    }
                start = self.text.find(quote, start + 1, end_of_section)


def _section_label(section):
    heading = " > ".join(section["heading_path"]) or "(untitled)"
    return f"{heading} (section anchor {section['anchor']!r})"


def _browse(corpus, selector):
    print(f"Corpus revision {corpus.revision}")
    for section_id, section in corpus.select_sections(selector).items():
        fields = section["fields"]
        print(f"\n{_section_label(section)}")
        print(f"{section_id} [{fields[0]['start']}, {fields[-1]['end']})")
        if selector is not None:
            for field in fields:
                start, end = field["start"], field["end"]
                print(f"  {field['path']} [{start}, {end}) (field anchor {field['anchor']!r})")
                print("    " + json.dumps(corpus.text[start:end], ensure_ascii=False))


def _print_span(corpus, span):
    print(f"  {_section_label(corpus.sections[span['section_id']])}")
    print("  " + json.dumps(span, ensure_ascii=False))
    print("  " + json.dumps(corpus.text[span["start"]:span["end"]], ensure_ascii=False))


def _review(corpus, annotations, questions):
    print(f"{annotations['split']} | annotation version {annotations['annotation_version']}")
    print(f"Corpus revision {corpus.revision}")
    for question in questions:
        status = "answerable" if question["answerable"] else "unanswerable"
        print(f"\n{question['id']} ({status})")
        print(question["question"])
        for number, alternatives in enumerate(question["evidence_units"], start=1):
            for alternative, span in enumerate(alternatives, start=1):
                print(f"Required unit {number}, alternative {alternative}")
                _print_span(corpus, span)


def _argument_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("document", type=Path, help="Processed schema-v1 JSON")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("browse", help="List sections or show all text fields in one section")
    find = commands.add_parser("find", help="List every valid, case-sensitive literal match")
    find.add_argument("quote")
    for command in ("browse", "find"):
        commands.choices[command].add_argument("--section", help="Section ID or section anchor")
    review = commands.add_parser("review", help="Validate a draft and display all selected excerpts")
    export = commands.add_parser("export", help="Write a validated schema-v1 annotation file")
    for command in (review, export):
        command.add_argument("draft", type=Path)
    export.add_argument("output", type=Path, help="New JSON file")
    return parser

def main() -> None:
    parser = _argument_parser()
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    try:
        corpus = EvidenceDocument(json.loads(args.document.read_text(encoding="utf-8")))
        if args.command == "browse":
            _browse(corpus, args.section)
        elif args.command == "find":
            print(f"Corpus revision {corpus.revision}")
            matches = list(corpus.find(args.quote, args.section))
            if not matches:
                print("No valid literal matches.")
            for span in matches:
                _print_span(corpus, span)
        else:
            annotations = json.loads(args.draft.read_text(encoding="utf-8"))
            questions = validate_annotations(annotations, corpus.revision, corpus.fields)
            if args.command == "review":
                _review(corpus, annotations, questions)
            else:
                output = json.dumps(annotations, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
                with args.output.open("x", encoding="utf-8", newline="\n") as file:
                    file.write(output)
    except (OSError, ValueError) as error:
        parser.exit(1, f"annotations: {error}\n")


if __name__ == "__main__":
    main()
