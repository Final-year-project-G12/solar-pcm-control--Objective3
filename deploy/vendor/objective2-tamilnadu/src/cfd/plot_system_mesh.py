"""
src/cfd/plot_system_mesh.py
==============================
Renders the system mesh's cell zones so the geometry can be checked by eye
before Fluent is trusted with it. This is a VERIFICATION render, not a
results figure -- it shows which cell belongs to which zone, nothing else.

Palette: the project's validated categorical slots, assigned to zones in
fixed slot order (never cycled, never by rank). Three of the six sit below
3:1 against the surface, so every zone also carries a direct label -- the
relief rule -- and identity is never colour-alone.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches          # noqa: E402
import matplotlib.pyplot as plt                # noqa: E402
import numpy as np                             # noqa: E402

from src.cfd.make_system_mesh import (          # noqa: E402
    capsule_centres, tank_dimensions,
)

SURFACE = "#fcfcfb"
INK = "#1a1a19"
INK_MUTED = "#6b6b68"
# validated light-mode categorical slots, in order
ZONE_COLOR = {
    "water":    "#2a78d6",
    "pcm":      "#eb6834",
    "shell":    "#1baf7a",
    "panel":    "#eda100",
    "absorber": "#e87ba4",
    "riser":    "#6250d6",
}
ZONE_LABEL = {
    "water": "Tank water", "pcm": "PCM", "shell": "Capsule shell (Al)",
    "panel": "Collector channel", "absorber": "Absorber plate", "riser": "Riser",
}


def render(out_png: Path, volume_L=100.0, collector_area_m2=2.5, r_in_m=0.020,
           wall_t_m=0.001, fraction=0.15, arrangement="staggered",
           nx=260, ny=200):
    import math
    tank_d, tank_h = tank_dimensions(volume_L)
    d_out = 2.0 * (r_in_m + wall_t_m)
    panel_h = min(collector_area_m2 / 1.0, tank_h * 1.15)
    panel_chan_w, panel_abs_w, gap, riser_w = 0.030, 0.012, 0.055, 0.022
    panel_x1 = panel_abs_w + panel_chan_w
    riser_x0 = panel_x1 + (gap - riser_w) / 2.0
    riser_x1 = riser_x0 + riser_w
    tank_x0 = panel_x1 + gap
    tank_x1 = tank_x0 + tank_d
    W, H = tank_x1, max(tank_h, panel_h)

    caps, _ = capsule_centres(tank_d, tank_h, d_out, fraction, arrangement)
    caps = [(tank_x0 + cx, cy) for cx, cy in caps]

    names = list(ZONE_COLOR)
    grid = np.full((ny, nx), -1, dtype=int)
    for i in range(nx):
        xc = W * (i + 0.5) / nx
        for j in range(ny):
            yc = H * (j + 0.5) / ny
            z = None
            if xc < panel_x1 and yc <= panel_h:
                z = "absorber" if xc < panel_abs_w else "panel"
            elif riser_x0 <= xc <= riser_x1 and yc <= panel_h:
                z = "riser"
            elif tank_x0 <= xc <= tank_x1 and 0.0 <= yc <= tank_h:
                z = "water"
                for (cx, cy) in caps:
                    rr = math.hypot(xc - cx, yc - cy)
                    if rr <= r_in_m:
                        z = "pcm"; break
                    if rr <= d_out / 2.0:
                        z = "shell"; break
            if z:
                grid[j, i] = names.index(z)

    cmap = matplotlib.colors.ListedColormap([ZONE_COLOR[n] for n in names])
    shown = np.ma.masked_where(grid < 0, grid)

    fig, ax = plt.subplots(figsize=(7.2, 8.4), dpi=170)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.imshow(shown, origin="lower", extent=[0, W, 0, H], cmap=cmap,
              vmin=0, vmax=len(names) - 1, interpolation="nearest")

    # direct labels (the relief rule: three slots are under 3:1 on this surface)
    def lab(x, y, text, color=INK):
        ax.annotate(text, (x, y), color=color, fontsize=8.5, ha="center",
                    va="center", fontweight="medium",
                    bbox=dict(boxstyle="round,pad=0.22", fc=SURFACE,
                              ec="#d9d9d6", lw=0.7, alpha=0.93))

    lab(panel_abs_w / 2, panel_h * 0.90, "Absorber")
    lab(panel_abs_w + panel_chan_w / 2, panel_h * 0.62, "Collector\nchannel")
    lab((riser_x0 + riser_x1) / 2, panel_h * 0.35, "Riser")
    lab(tank_x0 + tank_d * 0.5, tank_h * 0.965, "Tank water")
    if caps:
        cx, cy = caps[len(caps) // 2]
        ax.annotate("PCM capsules", (cx, cy), xytext=(tank_x1 * 0.995, tank_h * 0.5),
                    textcoords="data", color=INK, fontsize=8.5, ha="right",
                    va="center",
                    bbox=dict(boxstyle="round,pad=0.22", fc=SURFACE, ec="#d9d9d6", lw=0.7),
                    arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=0.9,
                                    shrinkA=2, shrinkB=6))

    ax.set_title("Tank + PCM capsules + collector panel — Fluent cell zones",
                 color=INK, fontsize=11.5, pad=12, loc="left")
    ax.set_xlabel("x  (m)", color=INK_MUTED, fontsize=9)
    ax.set_ylabel("y  (m)", color=INK_MUTED, fontsize=9)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=3, width=0.7)
    for s in ax.spines.values():
        s.set_color("#d9d9d6"); s.set_linewidth(0.8)
    ax.set_aspect("equal")

    counts = {n: int((grid == k).sum()) for k, n in enumerate(names)}
    handles = [mpatches.Patch(facecolor=ZONE_COLOR[n], edgecolor=SURFACE, lw=1.4,
                              label=f"{ZONE_LABEL[n]}  ({counts[n]:,} cells)")
               for n in names]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              frameon=False, fontsize=8.5, labelcolor=INK, handlelength=1.1,
              handleheight=1.1, borderpad=0)

    ax.annotate(f"{len(caps)} capsules · outer Ø {d_out*1000:.0f} mm · "
                f"tank Ø {tank_d*1000:.0f} × {tank_h*1000:.0f} mm · "
                f"2-D planar slice",
                (0, -0.065), xycoords="axes fraction", color=INK_MUTED, fontsize=8)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return {"png": str(out_png), "zone_cells": counts, "n_capsules": len(caps)}


if __name__ == "__main__":
    import json
    import sys
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "ansys/system_mesh_zones.png")
    print(json.dumps(render(out), indent=2))
