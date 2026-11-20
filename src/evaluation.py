"""Evaluate BM25 against one versioned, human-reviewed annotation split."""

import argparse
import json
from pathlib import Path
from statistics import mean
from time import perf_counter_ns

from src.chunking import _field, evidence_coverage
from src.retrieval import B, BM25, DEFAULT_TOP_K, K1, TOKENIZER, _range, _validate_top_k


PROTOCOL_VERSION = 1
METRICS = ("precision_at_k", "recall_at_k", "reciprocal_rank_at_k", "complete_evidence_coverage_at_k")


def _validate_annotations(annotations, index):
    version = _field(annotations, "annotation_schema_version", int, "")
    if type(version) is not int or version != 1:
        raise ValueError("Expected annotation schema version 1")
    if not _field(annotations, "annotation_version", str, "").strip():
        raise ValueError("annotation_version must be nonempty")
    if _field(annotations, "split", str, "") not in ("development", "frozen-test"):
        raise ValueError("split must be development or frozen-test")
    if _field(annotations, "corpus_revision", str, "") != index.corpus_revision:
        raise ValueError("Annotation corpus_revision does not match the chunk artifact")
    questions = _field(annotations, "questions", list, "")
    if not questions:
        raise ValueError("Annotations must contain at least one question")

    fields = {(chunk["section_id"], field["path"]): field
              for chunk in index.chunks for field in chunk["fields"]}
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
    return questions


def _relevant_ids(chunks, spans):
    """Each relevant chunk intersects an annotated span in at least one source character."""
    return {
        chunk["id"] for chunk in chunks
        if any(chunk["section_id"] == span["section_id"]
               and any(field["start"] < span["end"] and span["start"] < field["end"]
                       for field in chunk["fields"])
               for span in spans)
    }


def _score_question(index, units, hits, top_k):
    relevant = _relevant_ids(index.chunks, [span for unit in units for span in unit])
    ranks = [rank for rank, hit in enumerate(hits, start=1) if hit["id"] in relevant]
    covered = [any(evidence_coverage(
        [span | {"document_id": index.corpus_revision} for span in alternatives], hits
    )) for alternatives in units]
    return {
        "precision_at_k": len(ranks) / top_k,
        "recall_at_k": len(ranks) / len(relevant),
        "reciprocal_rank_at_k": 1 / ranks[0] if ranks else 0.0,
        "complete_evidence_coverage_at_k": float(all(covered)),
        "evidence_units_covered": covered,
    }


def evaluate(index: BM25, annotations: dict, top_k: int = DEFAULT_TOP_K) -> dict:
    _validate_top_k(top_k)
    questions = _validate_annotations(annotations, index)
    rows = []
    for question in questions:
        start = perf_counter_ns()
        hits = index.search(question["question"], top_k)
        elapsed_ms = (perf_counter_ns() - start) / 1_000_000
        row = {
            "question_id": question["id"],
            "answerable": question["answerable"],
            "retrieved_chunk_ids": [hit["id"] for hit in hits],
            "retrieval_latency_ms": elapsed_ms,
        }
        if question["answerable"]:
            row.update(_score_question(index, question["evidence_units"], hits, top_k))
        rows.append(row)

    answerable = [row for row in rows if row["answerable"]]
    averages = {key: mean(row[key] for row in answerable) if answerable else None for key in METRICS}
    averages["mrr_at_k"] = averages.pop("reciprocal_rank_at_k")
    return {
        "evaluation_protocol_version": PROTOCOL_VERSION,
        "annotation_schema_version": annotations["annotation_schema_version"],
        "annotation_version": annotations["annotation_version"],
        "split": annotations["split"],
        "corpus_revision": index.corpus_revision,
        "source_sha256": index.source_sha256,
        "chunk_artifact_sha256": index.artifact_sha256,
        "chunk_schema_version": 1,
        "chunk_max_chars": index.max_chars,
        "tokenizer": TOKENIZER.copy(),
        "bm25": {"k1": K1, "b": B, "idf": "log1p((N - df + 0.5) / (df + 0.5))"},
        "top_k": top_k,
        "summary": averages | {
            "answerable_questions": len(answerable),
            "unanswerable_questions": len(rows) - len(answerable),
            "unanswerable_no_results": sum(not row["retrieved_chunk_ids"] for row in rows if not row["answerable"]),
            "mean_retrieval_latency_ms": mean(row["retrieval_latency_ms"] for row in rows),
        },
        "questions": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chunks", type=Path)
    parser.add_argument("annotations", type=Path)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()
    try:
        annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
        report = evaluate(BM25(args.chunks), annotations, args.top_k)
    except (OSError, ValueError) as error:
        parser.exit(1, f"evaluation: {error}\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
