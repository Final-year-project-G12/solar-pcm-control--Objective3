"""
Phase O3-0 — Contract audit script (Rajasthan).

Run from the rajasthan O3 root:
    python scripts/phase0_audit.py

Loads the Rajasthan v3 contract, runs all validation checks (version, 50 L / 1.5 m2 fixed system,
flow envelope, limits, three regimes, held-out data, and the provenance hashes of the O2 files it
was built from), records the contract SHA-256, and prints a summary. Exits with code 1 if any check
fails (so it can gate CI). No flags: this phase is a single file read; --n-jobs / --n-envs are not needed.
"""

import json
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
import config_O2  # noqa: E402

config_O2.setup_paths()

CONTRACT_PATH = config_O2.CONTRACT_FILE


def main():
    print("=" * 60)
    print("Phase O3-0 — Contract Audit (Rajasthan)")
    print("=" * 60)

    from o3.utils.contract_loader import load_contract, validate_contract, contract_sha256, get_regime_design

    if not CONTRACT_PATH.exists():
        print(f"[FAIL] Contract not found: {CONTRACT_PATH}")
        sys.exit(1)
    print(f"[OK]   Contract file exists: {CONTRACT_PATH.name}")

    sha = contract_sha256(CONTRACT_PATH)
    print(f"[INFO] SHA-256: {sha[:16]}...")

    c = load_contract(CONTRACT_PATH)
    print(f"[INFO] State:   {c.get('state')}")
    print(f"[INFO] Version: {c.get('contract_version')}")
    print(f"[INFO] Generated: {c.get('generated_utc', 'unknown')}")

    errors = validate_contract(c)
    if errors:
        print(f"\n[FAIL] {len(errors)} validation error(s):")
        for i, e in enumerate(errors, 1):
            print(f"  [{i}] {e}")
        sys.exit(1)
    print("[OK]   All validation checks passed (incl. O2 provenance hashes).")

    # the copy in contracts/ must be byte-identical to the O2 original
    if config_O2.O2_CONTRACT_V3.exists():
        same = contract_sha256(config_O2.O2_CONTRACT_V3) == sha
        print(f"[{'OK' if same else 'WARN'}]   contracts/ copy {'equals' if same else 'DIFFERS FROM'} the O2 original")

    gl = c["global_limits"]
    print("\n--- Hard limits ---")
    print(f"  Delivery target:   {gl['delivery_target_temp_C']}°C")
    print(f"  Max water:         {gl['max_safe_water_temp_C']}°C")
    print(f"  Max PCM:           {gl['max_safe_pcm_temp_C']}°C")
    print(f"  Flow range:        {gl['flow_min_kg_s']}–{gl['flow_max_kg_s']} kg/s")

    fs = c["fixed_system"]
    print("\n--- Fixed system ---")
    print(f"  Tank volume:       {fs['tank_volume_L']} L")
    print(f"  Collector area:    {fs['collector_area_m2']} m²   ({fs['tank_volume_L'] / fs['collector_area_m2']:.1f} L/m²)")

    print("\n--- Regime designs ---")
    for rid in [0, 1, 2]:
        try:
            rd = get_regime_design(c, rid)
        except KeyError:
            print(f"  Regime {rid}: NOT FOUND")
            continue
        rows = [r for r in rd.get("held_out_2011_2015", []) if r.get("role") == "chosen"]
        years_done = sorted(set(r["year"] for r in rows))
        extra = f"  [{rd['small_sample_policy'][:46]}...]" if rd.get("small_sample_policy") else ""
        print(f"  Regime {rid}: PCM={rd['selected_pcm']:14s} N={rd['n_capsules']:3d} capsules  "
              f"flow={rd['nominal_flow_kg_s']:.4f} kg/s  held-out years={years_done}{extra}")

    out = config_O2.RESULTS_DIR / "phase0_audit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"contract": CONTRACT_PATH.name, "sha256": sha, "version": c["contract_version"],
                               "tank_L": fs["tank_volume_L"], "collector_m2": fs["collector_area_m2"],
                               "status": "PASSED"}, indent=2))
    print(f"\n[OK]   Phase O3-0 audit PASSED. Record: {out}")
    print("       Ready for Phase O3-1 (environment).")


if __name__ == "__main__":
    main()
