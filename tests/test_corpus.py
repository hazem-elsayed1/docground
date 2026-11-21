import json
from hashlib import sha256
from io import BytesIO
import subprocess
import sys

import pytest

from scripts import fetch_common_pitfalls
from src.chunking import evidence_coverage
from src.corpus import fetch_corpus, load_manifest


@pytest.fixture
def corpus_inputs(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    manifest = {
        "manifest_schema_version": 1,
        "corpus_version": "fixture-v1",
        "documentation_version": "1.9.1",
        "attribution": {
            "project": "scikit-learn", "copyright": "The scikit-learn developers",
            "license": "BSD-3-Clause",
            "license_url": "https://github.com/scikit-learn/scikit-learn/blob/1.9.1/COPYING",
        },
        "documents": [],
    }
    for name in ("zeta", "alpha"):
        url = f"https://scikit-learn.org/1.9/modules/{name}.html"
        payload = (
            f'<title>Fixture \u2014 scikit-learn 1.9.1 documentation</title>'
            f'<link rel="canonical" href="{url}">'
            '<article><section id="shared"><h1>Usage</h1><pre>if ready:\n    fit()\n</pre></section></article>'
        ).encode()
        (raw_dir / f"{name}.html").write_bytes(payload)
        manifest["documents"].append({"id": name, "url": url, "sha256": sha256(payload).hexdigest()})
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, raw_dir


@pytest.mark.parametrize(("key", "value", "message"), [
    ("manifest_schema_version", True, "schema version"),
    ("documentation_version", "stable", "documentation version"),
    ("documents", [], "at least one"),
    ("license_url", "https://example.com/license", "license URL"),
    ("id", "../escape", "document ID"),
    ("id", "con", "document ID"),
    ("id", "zeta", "Duplicate"),
    ("url", "https://scikit-learn.org/stable/modules/alpha.html", "versioned HTTPS"),
    ("url", "https://scikit-learn.org/1.9/modules/zeta.html", "Duplicate"),
    ("sha256", "not-a-hash", "SHA-256"),
])
def test_invalid_manifest_is_rejected(corpus_inputs, key, value, message):
    manifest_path, _ = corpus_inputs
    manifest = json.loads(manifest_path.read_bytes())
    if key in manifest:
        manifest[key] = value
    elif key == "license_url":
        manifest["attribution"][key] = value
    else:
        manifest["documents"][1][key] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_manifest(manifest_path)


def run_cli(action, manifest_path, raw_dir, output_dir):
    return subprocess.run(
        [sys.executable, "-m", "src.corpus", action, "--manifest", str(manifest_path),
         "--raw-dir", str(raw_dir), "--output-dir", str(output_dir)],
        capture_output=True, encoding="utf-8",
    )


def test_build_writes_reproducible_documents_and_chunks_in_manifest_order(corpus_inputs, tmp_path):
    manifest_path, raw_dir = corpus_inputs
    first, second = (tmp_path / name for name in ("first", "second"))
    for directory in (first, second):
        result = run_cli("build", manifest_path, raw_dir, directory)
        assert result.returncode == 0, result.stderr
    for file in ("documents.json", "chunks.json"):
        assert (first / file).read_bytes() == (second / file).read_bytes()

    processed = json.loads((first / "documents.json").read_bytes())
    chunked = json.loads((first / "chunks.json").read_bytes())
    manifest = load_manifest(manifest_path)
    assert [entry["id"] for entry in processed["documents"]] == ["zeta", "alpha"]
    assert [entry["id"] for entry in chunked["documents"]] == ["zeta", "alpha"]
    assert processed["attribution"] == chunked["attribution"] == manifest["attribution"]
    assert processed["manifest_sha256"] == chunked["manifest_sha256"]
    chunks = []
    for entry, doc, group in zip(manifest["documents"], processed["documents"], chunked["documents"], strict=True):
        [chunk] = group["chunk_artifact"]["chunks"]
        assert doc["url"] == group["url"] == chunk["source_url"] == entry["url"]
        assert doc["document_id"] == group["document_id"] == chunk["document_id"]
        assert doc["document"]["source"]["sha256"] == group["sha256"] == chunk["source_sha256"] == entry["sha256"]
        assert chunk["text"] == "if ready:\n    fit()\n"
        assert chunk["start"] == 0
        assert chunk["fields"][0]["path"] == "/blocks/0/text"
        assert chunk["anchor"] == "shared"
        chunks.append(chunk)
    assert all(chunks[0][key] != chunks[1][key] for key in ("id", "document_id", "section_id"))
    span = {key: chunks[0][key] for key in ("document_id", "section_id", "start", "end")}
    assert evidence_coverage([span], [chunks[0]]) == [True]
    assert evidence_coverage([span], [chunks[1]]) == [False]
    assert evidence_coverage([span | {"section_id": chunks[1]["section_id"]}], chunks) == [False]


def test_manifest_fetch_uses_pins_without_overwriting(corpus_inputs, tmp_path, monkeypatch):
    manifest_path, raw_dir = corpus_inputs
    manifest = load_manifest(manifest_path)
    payloads = {doc["url"]: (raw_dir / f"{doc['id']}.html").read_bytes() for doc in manifest["documents"]}
    downloaded = tmp_path / "new-raw"
    monkeypatch.setattr(fetch_common_pitfalls._opener, "open", lambda url, timeout: BytesIO(payloads[url]))
    fetch_corpus(manifest, downloaded)
    for doc in manifest["documents"]:
        assert (downloaded / f"{doc['id']}.html").read_bytes() == payloads[doc["url"]]

    existing = downloaded / "zeta.html"
    existing.write_bytes(b"locally existing different file")
    with pytest.raises(ValueError, match="expected SHA-256"):
        fetch_corpus(manifest, downloaded)
    assert existing.read_bytes() == b"locally existing different file"


@pytest.mark.parametrize(("failure", "message"), [
    ("missing", "No such file"),
    ("hash", "SHA-256 mismatch"),
    ("version", "Documentation version mismatch"),
    ("section collision", "Duplicate section ID"),
])
def test_invalid_source_does_not_publish_corpus(corpus_inputs, tmp_path, failure, message):
    manifest_path, raw_dir = corpus_inputs
    path = raw_dir / "alpha.html"
    if failure == "missing":
        path.unlink()
    elif failure == "hash":
        path.write_bytes(b"changed")
    else:
        old = path.read_bytes()
        changed = (old.replace(b"1.9.1 documentation", b"1.9.2 documentation") if failure == "version"
                   else old.replace(b"/alpha.html", b"/zeta.html"))
        assert changed != old
        path.write_bytes(changed)
        manifest = json.loads(manifest_path.read_bytes())
        manifest["documents"][1]["sha256"] = sha256(changed).hexdigest()
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    output_dir = tmp_path / "output"
    result = run_cli("build", manifest_path, raw_dir, output_dir)
    assert result.returncode == 1
    assert message in result.stderr
    assert not output_dir.exists()
