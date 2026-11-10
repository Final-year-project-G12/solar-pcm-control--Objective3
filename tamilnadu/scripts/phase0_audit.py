"""
Phase O3-0 — Contract audit script.

Run from the objective3 root:
    python scripts/phase0_audit.py

Loads the contract, runs all validation checks, and prints a summary.
Exits with code 1 if any check fails (so it can gate CI).
"""

import pathlib
import sys

CONTRACT_PATH = pathlib.Path(__file__).parents[1] / "contracts" / "obj3_environment_contract_tamilnadu_v3.json"


def main():
    print("=" * 60)
    print("Phase O3-0 — Contract Audit")
    print("=" * 60)

    repo_root = pathlib.Path(__file__).parents[1]
    o2_root   = repo_root.parent / "objective2_design_optimization"
    for p in [str(o2_root), str(repo_root)]:
        if p not in sys.path:
            sys.path.insert(0, p)

    from o3.utils.contract_loader import (
        load_contract, validate_contract, contract_sha256, get_regime_design
    )

    if not CONTRACT_PATH.exists():
        print(f"[FAIL] Contract not found: {CONTRACT_PATH}")
        sys.exit(1)

    print(f"[OK]   Contract file exists: {CONTRACT_PATH.name}")

    sha = contract_sha256(CONTRACT_PATH)
    print(f"[INFO] SHA-256: {sha[:16]}...")

    c = load_contract(CONTRACT_PATH)
    print(f"[INFO] Version: {c.get('contract_version')}")
    print(f"[INFO] Generated: {c.get('generated_utc', 'unknown')}")

    errors = validate_contract(c)
    if errors:
        print(f"\n[FAIL] {len(errors)} validation error(s):")
        for i, e in enumerate(errors, 1):
            print(f"  [{i}] {e}")
        sys.exit(1)

    print("[OK]   All validation checks passed.")

    gl = c["global_limits"]
    print(f"\n--- Hard limits ---")
    print(f"  Delivery target:   {gl['delivery_target_temp_C']}°C")
    print(f"  Max water:         {gl['max_safe_water_temp_C']}°C")
    print(f"  Max PCM:           {gl['max_safe_pcm_temp_C']}°C")
    print(f"  Flow range:        {gl['flow_min_kg_s']}–{gl['flow_max_kg_s']} kg/s")

    fs = c["fixed_system"]
    print(f"\n--- Fixed system ---")
    print(f"  Tank volume:       {fs['tank_volume_L']} L")
    print(f"  Collector area:    {fs['collector_area_m2']} m²")

    print(f"\n--- Regime designs ---")
    for rid in [0, 1, 2]:
        try:
            rd = get_regime_design(c, rid)
        except KeyError:
            print(f"  Regime {rid}: NOT FOUND")
            continue
        rows = rd.get("held_out_2011_2015", [])
        years_done = sorted(set(r["year"] for r in rows))
        print(
            f"  Regime {rid}: PCM={rd['selected_pcm']:15s} "
            f"N={rd['n_capsules']:3d} capsules  "
            f"flow={rd['nominal_flow_kg_s']:.4f} kg/s  "
            f"held-out years={years_done}"
        )

    print(f"\n[OK]   Phase O3-0 audit PASSED. Ready for Phase O3-1 (environment).")


if __name__ == "__main__":
    main()
