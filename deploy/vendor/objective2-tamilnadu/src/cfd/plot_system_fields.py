"""
src/cfd/plot_system_fields.py
================================
Maps the solved Fluent fields back onto the generating grid and plots them.

Fluent's own picture calls fail on this install (`api-set-var: the object is
not active` -- no graphics subsystem in any headless mode pyfluent can
launch here), and its ASCII export silently dropped the field list. So the
fields are read straight out of the solved `.dat.h5`:

    /results/1/phase-1/cells/SV_T      temperature, all cells
    /results/1/phase-1/cells/SV_LIQF   liquid fraction, meltable zones

Those arrays are in Fluent's cell order, which is the order
make_system_mesh.py assigned: zone by zone in ZONE_ORDER, and within a zone
by sorted (i, j). The mapping is re-derived here from the generator itself
rather than reimplemented, so the two cannot drift apart; it is then CHECKED
(the absorber must be the hottest zone, since it is the only one with a heat
source) before the figure is drawn.

Sequential field -> one hue, light to dark. Not a rainbow.
"""

from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

from src.cfd.make_system_mesh import write_system_msh   # noqa: E402

BASE = Path(__file__).resolve().parents[2]
CASE_DIR = BASE / "ansys"
DAT = CASE_DIR / "system_tank_pcm_panel_solved.dat.h5"

SURFACE = "#fcfcfb"
INK = "#1a1a19"
INK_MUTED = "#6b6b68"


def load_layout():
    lay = {}
    write_system_msh(CASE_DIR / "system_tank_pcm_panel.msh", _layout_out=lay)
    return lay


def read_fields():
    with h5py.File(DAT, "r") as f:
        cells = f["/results/1/phase-1/cells"]
        T = np.asarray(cells["SV_T"]["1"]).ravel()
        liqf = {k: np.asarray(cells["SV_LIQF"][k]).ravel()
                for k in cells["SV_LIQF"]}
    return T, liqf


def to_grid(values, lay, zones=None):
    """Scatter a per-cell array back onto the (ny, nx) grid."""
    nx, ny = lay["nx"], lay["ny"]
    g = np.full((ny, nx), np.nan)
    for (i, j), c in lay["cid"].items():
        if zones and lay["tag"][(i, j)] not in zones:
            continue
        if c - 1 < values.size:
            g[j, i] = values[c - 1]
    return g


def main():
    lay = load_layout()
    T, liqf = read_fields()
    sizes = lay["zone_sizes"]
    total = sum(n for _, n in sizes)
    if T.size != total:
        raise SystemExit(f"cell count mismatch: data {T.size} vs layout {total} "
                         "-- the mesh was regenerated with different settings "
                         "than the one that was solved")

    # per-zone stats + the ordering check
    stats, i = {}, 0
    for name, n in sizes:
        stats[name] = T[i:i + n] - 273.15
        i += n
    hottest = max(stats, key=lambda k: stats[k].mean())
    if hottest != "absorber":
        raise SystemExit(f"cell-order check FAILED: '{hottest}' is hotter than "
                         "the absorber, which is the only zone with a heat "
                         "source -- the .dat does not map onto this grid")
    print(f"{'zone':10s} {'n':>6s} {'min':>8s} {'mean':>8s} {'max':>8s}   (deg C)")
    for name, n in sizes:
        s = stats[name]
        print(f"{name:10s} {n:6d} {s.min():8.2f} {s.mean():8.2f} {s.max():8.2f}")
    print("cell-order check: OK (absorber is hottest)")

    gT = to_grid(T - 273.15, lay)
    xs, ys = lay["xs"], lay["ys"]
    extent = [xs[0], xs[-1], ys[0], ys[-1]]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 7.6), dpi=170)
    fig.patch.set_facecolor(SURFACE)

    # --- temperature, full system ---
    ax = axes[0]
    ax.set_facecolor(SURFACE)
    im = ax.imshow(np.ma.masked_invalid(gT), origin="lower", extent=extent,
                   cmap="inferno", interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("temperature  (°C)", color=INK_MUTED, fontsize=9)
    cb.ax.tick_params(colors=INK_MUTED, labelsize=8)
    cb.outline.set_edgecolor("#d9d9d6")
    ax.set_title("Temperature — whole system", color=INK, fontsize=11, loc="left", pad=10)

    # --- temperature, tank only: the system range hides the tank detail ---
    ax2 = axes[1]
    ax2.set_facecolor(SURFACE)
    gTank = to_grid(T - 273.15, lay, zones={"water", "pcm", "shell"})
    im2 = ax2.imshow(np.ma.masked_invalid(gTank), origin="lower", extent=extent,
                     cmap="inferno", interpolation="nearest")
    cb2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.03)
    cb2.set_label("temperature  (°C)", color=INK_MUTED, fontsize=9)
    cb2.ax.tick_params(colors=INK_MUTED, labelsize=8)
    cb2.outline.set_edgecolor("#d9d9d6")
    ax2.set_title("Tank and capsules only — rescaled", color=INK, fontsize=11,
                  loc="left", pad=10)

    for a in axes:
        a.set_aspect("equal")
        a.set_xlabel("x  (m)", color=INK_MUTED, fontsize=9)
        a.tick_params(colors=INK_MUTED, labelsize=8, length=3, width=0.7)
        for s in a.spines.values():
            s.set_color("#d9d9d6"); s.set_linewidth(0.8)
    axes[0].set_ylabel("y  (m)", color=INK_MUTED, fontsize=9)

    pcm = stats["pcm"]
    note = (f"absorber {stats['absorber'].mean():.0f} °C → panel "
            f"{stats['panel'].mean():.0f} °C → riser {stats['riser'].mean():.0f} "
            f"°C → water {stats['water'].mean():.1f} °C → PCM "
            f"{pcm.mean():.1f} °C   ·   PCM stays solid (melts at 56.5 °C); "
            f"energy-only, momentum frozen")
    fig.text(0.012, 0.015, note, color=INK_MUTED, fontsize=8.5)
    fig.suptitle("ANSYS Fluent — tank + PCM capsules + collector panel",
                 color=INK, fontsize=12.5, x=0.012, ha="left", y=0.985)

    out = CASE_DIR / "system_fields.png"
    fig.savefig(out, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
