"""Fetch the byte-pinned scikit-learn Common Pitfalls HTML snapshot."""

import argparse
from hashlib import sha256
from pathlib import Path
from urllib.request import HTTPRedirectHandler, build_opener


SOURCE_URL = "https://scikit-learn.github.io/1.9/common_pitfalls.html"
OFFICIAL_URL = "https://scikit-learn.org/1.9/common_pitfalls.html"
SOURCE_SHA256 = "21fac11c181ab35624dcf8673f0475d403d4f6ecf0f9bead01a573d7642b515c"
DEFAULT_PATH = Path("data/raw/common_pitfalls.html")
MAX_BYTES = 1_000_000


class _SnapshotRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # The official mirror sends an HTTP Location. Fetch the same target over TLS.
        if newurl not in (OFFICIAL_URL, OFFICIAL_URL.replace("https:", "http:", 1)):
            raise ValueError(f"Unexpected snapshot redirect: {newurl}")
        return super().redirect_request(req, fp, code, msg, headers, OFFICIAL_URL)


_opener = build_opener(_SnapshotRedirect())


def _verify(payload: bytes, origin: str) -> None:
    actual = sha256(payload).hexdigest()
    if actual != SOURCE_SHA256:
        raise ValueError(f"{origin}: expected SHA-256 {SOURCE_SHA256}, got {actual}")


def fetch_snapshot(path: Path) -> str:
    try:
        with path.open("rb") as existing:
            payload = existing.read(MAX_BYTES + 1)
    except FileNotFoundError:
        pass
    else:
        _verify(payload, str(path))
        return "Verified existing snapshot"

    with _opener.open(SOURCE_URL, timeout=30) as response:
        payload = response.read(MAX_BYTES + 1)
    if len(payload) > MAX_BYTES:
        raise ValueError("Downloaded snapshot exceeds the size limit")
    _verify(payload, SOURCE_URL)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as destination:
        destination.write(payload)
    return "Saved verified snapshot"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, nargs="?", default=DEFAULT_PATH)
    args = parser.parse_args()
    try:
        status = fetch_snapshot(args.path)
    except (OSError, ValueError) as error:
        parser.exit(1, f"fetch_common_pitfalls: {error}\n")
    print(f"{status} at {args.path} (SHA-256 {SOURCE_SHA256})")


if __name__ == "__main__":
    main()
