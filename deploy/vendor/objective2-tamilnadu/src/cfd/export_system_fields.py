"""
src/cfd/export_system_fields.py
==================================
Pulls the solved temperature / melt-fraction fields out of the system case
as plain numbers, and reports the integral quantities.

Why not Fluent's own contour images: this install refuses every picture call
with `api-set-var: the object is not active` -- the graphics subsystem is not
available in the headless modes pyfluent can launch here. Rather than keep
guessing at launch flags, the fields are exported as ASCII and plotted with
the same matplotlib/palette as every other figure in the project, which also
makes the figure reproducible without ANSYS.

Run:  python -m src.cfd.export_system_fields
"""

import traceback
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
CASE_DIR = BASE / "ansys"
CASE = CASE_DIR / "system_tank_pcm_panel_solved.cas.h5"
DATA = CASE_DIR / "system_tank_pcm_panel_solved.dat.h5"
ASCII_OUT = CASE_DIR / "system_fields.txt"
WORK = Path(r"C:\fluent_work")

FIELDS = ["temperature", "liquid-fraction"]


def step(label, fn):
    print(f"\n=== {label} ===", flush=True)
    try:
        out = fn()
        print(f"OK: {label}", flush=True)
        return out
    except Exception as e:
        print(f"FAILED: {label}: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        return None


def main():
    import ansys.fluent.core as pyfluent

    WORK.mkdir(parents=True, exist_ok=True)
    solver = pyfluent.launch_fluent(
        product_version="26.1.0", dimension=2, precision="double",
        processor_count=2, ui_mode="no_gui_or_graphics",
        start_timeout=300, cwd=str(WORK),
    )
    print("Fluent connected.", flush=True)

    step("read case+data",
         lambda: solver.settings.file.read_case_data(file_name=str(CASE)))

    # Integral quantities -- these are the numbers worth quoting.
    def report():
        rep = solver.settings.results.report
        out = {}
        for zone in ("pcm", "water", "panel", "absorber", "riser", "shell"):
            try:
                v = rep.volume_integrals.volume_average(
                    field="temperature", cell_zones=[zone])
                out[f"T_avg_{zone}_K"] = v
            except Exception as e:
                print(f"  T_avg {zone}: {type(e).__name__}: {e}")
        for field in ("temperature", "liquid-fraction"):
            for stat in ("maximum", "minimum"):
                try:
                    fn = getattr(rep.volume_integrals, stat)
                    out[f"{stat}_{field}"] = fn(field=field, cell_zones=["pcm"])
                except Exception as e:
                    print(f"  {stat} {field}: {type(e).__name__}: {e}")
        print("  " + "\n  ".join(f"{k} = {v}" for k, v in out.items()))
        return out
    step("integral report", report)

    def export_ascii():
        solver.settings.file.export.ascii(
            file_name=str(ASCII_OUT),
            surface_name_list=[],
            delimiter="space",
            cell_func_domain=FIELDS,
            location="cell-center",
        )
    if step("export ascii (settings API)", export_ascii) is None:
        step("export ascii (TUI fallback)",
             lambda: solver.tui.file.export.ascii(
                 str(ASCII_OUT), "()", "yes", *FIELDS, "()", "no"))

    step("exit", solver.exit)
    if ASCII_OUT.exists():
        print(f"\nwrote {ASCII_OUT}  ({ASCII_OUT.stat().st_size/1024:.0f} KB)")
    else:
        print("\nno ascii export produced")


if __name__ == "__main__":
    main()
