"""
src/cfd/setup_system_case.py
===============================
Sets up and solves the whole-system Fluent case -- tank + PCM capsules +
collector panel -- from `system_tank_pcm_panel.msh`, then writes contour
images of the temperature field and the PCM melt fraction.

Built on the pattern proven in ansys/pyfluent_full_setup2.py,
which is the script that produced the existing solved capsule cases. Three
things are carried over from it deliberately:

  * PLANAR, not axisymmetric. That script records three independent attempts
    at true axisymmetric mode, all diverging immediately with the same
    momentum/pressure blow-up near the axis. Planar is the verified-stable
    interpretation, and this geometry has no axis of revolution anyway
    (the panel sits beside the tank).
  * PCM as a FLUID material carrying melting_heat / tsolidus / tliqidus, with
    `define/models/solidification-melting`. Fluent's enthalpy-porosity model
    is a fluid-zone model; a solid zone cannot melt.
  * A WIDE mushy band. That script's own notes show the run surviving longer
    as the solidus-liquidus gap widened (1 K -> 3 K -> 5 K -> 10 K), and
    identify band width -- not timestep -- as the dominant stability lever.

Difference from that script: every zone here has a NAME (water, pcm, shell,
panel, absorber, riser, tank_wall, ...) because make_system_mesh.py writes
them, so none of the zone-identity guesswork that script had to do by
reading HDF5 adjacency applies.

The closed-loop geometry has no inlet or outlet: every external boundary is
a wall. Heat enters as a solar flux on the absorber and leaves through the
tank/panel/riser walls, which is the thermosiphon arrangement this system
actually is.

Run:  python -m src.cfd.setup_system_case [--iters N] [--steady|--transient]
"""

import argparse
import os
import traceback
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
CASE_DIR = BASE / "ansys"
MESH = CASE_DIR / "system_tank_pcm_panel.msh"
WORK = Path(r"C:\fluent_work")

# --- physical inputs -------------------------------------------------------
# Solar gain on the absorber: mid-day beam on a good day x the collector's
# optical efficiency, matching fr_tau_alpha in system_config_shared.yaml.
SOLAR_FLUX_W_M2 = 800.0 * 0.75
T_INIT_K = 303.15          # 30 C cold start, same as solver.initial_water_temp_C
T_AMB_K = 303.15
H_TANK_W_M2K = 1.2         # insulated tank
H_PANEL_W_M2K = 6.0        # panel glazing loses more

# n-Hexacosane (C26) -- Objective 1's rank-2 PCM for regimes 0 and 1.
PCM = dict(name="c26", density=770.0, cp=2100.0, k=0.21,
           latent_J_kg=256000.0, Tm_C=56.5, band_K=10.0)


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--flow", action="store_true",
                    help="solve momentum too; default is energy-only, which is "
                         "far more robust and still shows the whole thermal path")
    args = ap.parse_args()

    import ansys.fluent.core as pyfluent

    WORK.mkdir(parents=True, exist_ok=True)
    print(f"mesh: {MESH}  ({MESH.stat().st_size/1024:.0f} KB)")

    # ui_mode="no_gui", NOT "no_gui_or_graphics". The latter maps to Fluent's
    # -g, which disables the graphics subsystem outright, and every
    # picture/contour call then fails with
    #   "api-set-var: the object is not active  ...  hardcopy-format".
    # No GUI is still wanted; the graphics engine is needed to render to PNG.
    solver = pyfluent.launch_fluent(
        product_version="26.1.0", dimension=2, precision="double",
        processor_count=args.procs, ui_mode="no_gui",
        start_timeout=300, cwd=str(WORK),
    )
    print("Fluent connected.", flush=True)

    # read_mesh, not read_case: a .msh is a mesh, and read_case leaves the
    # session with nothing loaded, after which every settings branch reports
    # itself "currently inactive" and the whole script no-ops.
    step("read mesh", lambda: solver.settings.file.read_mesh(file_name=str(MESH)))
    step("planar", lambda: setattr(solver.setup.general.solver, "two_dim_space", "planar"))
    step("energy on", lambda: setattr(solver.setup.models.energy, "enabled", True))
    step("laminar", lambda: setattr(solver.setup.models.viscous, "model", "laminar"))
    step("gravity", lambda: solver.tui.define.operating_conditions.gravity("yes", "0", "-9.81"))
    step("solidification-melting on",
         lambda: solver.tui.define.models.solidification_melting("yes"))

    def make_pcm():
        solver.setup.materials.fluid.create(PCM["name"])
        m = solver.setup.materials.fluid[PCM["name"]]
        m.density.option = "constant"; m.density.value = PCM["density"]
        m.specific_heat.option = "constant"; m.specific_heat.value = PCM["cp"]
        m.thermal_conductivity.option = "constant"; m.thermal_conductivity.value = PCM["k"]
        m.melting_heat.option = "constant"; m.melting_heat.value = PCM["latent_J_kg"]
        half = PCM["band_K"] / 2.0
        m.tsolidus.option = "constant"
        m.tsolidus.value = PCM["Tm_C"] + 273.15 - half
        m.tliqidus.option = "constant"
        m.tliqidus.value = PCM["Tm_C"] + 273.15 + half
    step("create PCM material", make_pcm)

    def make_water():
        solver.setup.materials.database.copy_materials(
            type="fluid", form=False, names=["water-liquid"])
        w = solver.setup.materials.fluid["water-liquid"]
        w.density.value = 985.0
        w.specific_heat.value = 4186.0
        w.thermal_conductivity.value = 0.63
        w.viscosity.value = 5.0e-4
    step("water material", make_water)

    def make_alu():
        solver.setup.materials.database.copy_materials(
            type="solid", form=False, names=["aluminum"])
    step("aluminium material", make_alu)

    def assign():
        fz = solver.setup.cell_zone_conditions.fluid
        sz = solver.setup.cell_zone_conditions.solid
        for z in ("water", "panel", "riser"):
            fz[z].general.material = "water-liquid"
        fz["pcm"].general.material = PCM["name"]
        for z in ("shell", "absorber"):
            sz[z].general.material = "aluminum"
    step("assign zone materials", assign)

    def walls():
        w = solver.setup.boundary_conditions.wall
        # Solar gain lands on the ABSORBER exterior only -- that is the whole
        # reason the boundary zones are named per adjacent cell zone.
        w["wall_absorber_ext"].thermal.thermal_condition = "Heat Flux"
        w["wall_absorber_ext"].thermal.heat_flux.value = SOLAR_FLUX_W_M2
        for name, h in (("wall_water_ext", H_TANK_W_M2K),
                        ("wall_riser_ext", H_PANEL_W_M2K),
                        ("wall_panel_ext", H_PANEL_W_M2K)):
            w[name].thermal.thermal_condition = "Convection"
            w[name].thermal.heat_transfer_coeff.value = h
            w[name].thermal.free_stream_temp.value = T_AMB_K
    step("wall boundary conditions", walls)

    if not args.flow:
        # Energy-only: freeze the momentum equations. The point of this case is
        # the thermal path through panel -> riser -> tank -> capsules and the
        # melt front; solving natural convection in a closed multi-zone cavity
        # with a mushy region is where the earlier attempts diverged, and it is
        # not needed to show either.
        def freeze_flow():
            eq = solver.settings.solution.controls.equations
            for key in ("flow", "Flow"):
                if key in eq:
                    eq[key] = False
                    return key
            raise KeyError(f"no flow equation toggle; have {list(eq)}")
        step("freeze flow equations", freeze_flow)

    step("initialize", lambda: (
        setattr(solver.solution.initialization, "initialization_type", "standard"),
        solver.tui.solve.initialize.set_defaults("temperature", str(T_INIT_K)),
        solver.solution.initialization.standard_initialize()))

    step("save mesh-only case",
         lambda: solver.file.write_case(
             file_name=str(CASE_DIR / "system_tank_pcm_panel_setup.cas.h5")))

    step(f"iterate {args.iters}",
         lambda: solver.solution.run_calculation.iterate(iter_count=args.iters))

    step("save case+data",
         lambda: solver.file.write_case_data(
             file_name=str(CASE_DIR / "system_tank_pcm_panel_solved.cas.h5")))

    def contours():
        # The settings API, not the TUI: `tui.display.set.picture.driver` is a
        # menu OBJECT here, not a callable ('driver' object is not callable).
        g = solver.settings.results.graphics
        g.picture.driver_options.hardcopy_format = "png"
        g.picture.use_window_resolution = False
        g.picture.x_resolution = 1500
        g.picture.y_resolution = 1100
        for field, fname in (("temperature", "img_system_temperature.png"),
                             ("liquid-fraction", "img_system_liquid_fraction.png")):
            try:
                name = f"ct_{field.replace('-', '_')}"
                g.contour.create(name)
                c = g.contour[name]
                c.field = field
                c.surfaces_list = []
                c.filled = True
                c.node_values = True
                c.display()
                g.picture.save_picture(file_name=str(CASE_DIR / fname))
                print(f"  wrote {fname}")
            except Exception as e:
                print(f"  contour {field} failed: {type(e).__name__}: {e}")
    step("contours", contours)

    step("exit", solver.exit)
    print("\ndone.")


if __name__ == "__main__":
    main()
