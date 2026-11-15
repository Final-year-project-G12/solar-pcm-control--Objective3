"""
hub/vendor_fix.py - make the vendored Objective-2 files byte-exact again, whatever git did to their line endings.

Rajasthan's O3 contract stores a SHA-256 of each Objective-2 file it was built from, and refuses to run if one differs. The hashes
were taken on Windows, so the files are CRLF. A git checkout may hand us LF instead (core.autocrlf=true on the committer's machine
normalises them to LF when they are committed, and a Linux runner then checks them out as LF). .gitattributes marks the vendor
folder `-text` to prevent that, but a repo that was committed before the attribute existed keeps its LF blobs until they are
re-added, so this module repairs the bytes at run time instead of trusting the history.

Rule: for every file named in the contract's provenance block, if its hash does not match, try the same content with CRLF line
endings (and, if that is not it either, with LF). Rewrite the file only when a variant matches the contract hash exactly; a file that
matches no variant is reported and left alone (that is a real difference, not a line-ending one).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
VENDOR = REPO / "deploy" / "vendor" / "objective2-rajasthan"
CONTRACT = REPO / "rajasthan" / "contracts" / "obj3_environment_contract_rajasthan_v3.json"


def expected_hashes() -> dict[str, str]:
    prov = json.loads(CONTRACT.read_text(encoding="utf-8"))["provenance"]
    want = {f"results/{k}": h for k, h in prov.get("selection", {}).items()}
    want.update({f"results/{k}": prov[k] for k in ("g10_heldout_results.csv", "held_out_test.lock", "wp6_band.csv") if k in prov})
    want.update({k: prov[k] for k in ("configs/system_config_shared.yaml", "configs/design_space.yaml",
                                      "configs/states/rajasthan.yaml") if k in prov})
    return want


def _variants(raw: bytes):
    lf = raw.replace(b"\r\n", b"\n")
    yield raw
    yield lf.replace(b"\n", b"\r\n")
    yield lf


def ensure_vendor_bytes() -> list[str]:
    """Repair line endings of the vendored files. Returns the files that still do not match the contract (empty = all good)."""
    bad = []
    for rel, h in expected_hashes().items():
        p = VENDOR / rel
        if not p.exists():
            bad.append(f"{rel} (missing)")
            continue
        raw = p.read_bytes()
        for cand in _variants(raw):
            if hashlib.sha256(cand).hexdigest() == h:
                if cand != raw:
                    p.write_bytes(cand)
                break
        else:
            bad.append(rel)
    return bad
