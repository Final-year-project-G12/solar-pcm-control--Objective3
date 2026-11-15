"""
src/cfd/make_system_mesh.py
==============================
Writes a 2-D planar Fluent/Gambit ASCII `.msh` of the WHOLE system in one
domain: the storage tank, the PCM capsule array inside it, and the flat-plate
collector panel beside it.

WHY A SEPARATE MESH FROM make_tank_mesh.py. That one is a bare axisymmetric
tank for the G4 draw case -- one fluid zone, no capsules, no panel. This one
exists to SHOW the structure (tank + PCM + panel) and to let Fluent solve
conduction/convection across all of it, which needs several cell zones:

    water      tank water, minus the capsules                  (fluid)
    pcm        the capsule interiors                           (solid, PCM props)
    shell      the aluminium capsule walls                     (solid)
    panel      collector channel water                         (fluid)
    absorber   collector absorber plate                         (solid)
    riser      the pipe run connecting panel outlet to tank     (fluid)

PLANAR, NOT AXISYMMETRIC -- and that is a deliberate trade. The tank alone is
a revolved rectangle, so make_tank_mesh.py uses Fluent's axisymmetric solver
and gets true annular cell volumes. The moment the panel sits BESIDE the tank
the assembly has no axis of revolution, so this mesh is a vertical planar
slice. Consequence: absolute litres and kWh from this mesh are per unit depth
and are NOT comparable to the node model. This mesh is for structure,
temperature fields and melt-front shape. Every energy number that gets
reported comes from the node model or from the axisymmetric draw case.

HOW THE ZONES ARE BUILT. A single structured quad grid covers the bounding
box; each cell is then tagged by where its centroid falls (inside a capsule
-> pcm or shell; inside the panel channel -> panel; etc.). Cells are
renumbered so each zone is one contiguous id block, because the Fluent format
declares a cell zone as a first..last range. Faces between two DIFFERENT cell
zones become coupled walls so conjugate heat transfer is solved across them;
faces inside one zone are interior.

Format: (2) dims, (10) nodes, (12) cell zones, (13) face zones, (45) names.
A face carries (left cell, right cell) with 0 meaning outside the domain.
"""

import math
from collections import defaultdict
from pathlib import Path


# Cell-id blocks are laid out in this order. Changing it changes the mapping
# from a solved Fluent .dat back onto this grid -- see plot_system_fields.py.
ZONE_ORDER = ["water", "pcm", "shell", "panel", "absorber", "riser"]


def tank_dimensions(volume_L: float, height_to_diameter: float = 2.0):
    v = volume_L / 1000.0
    d = (4.0 * v / (math.pi * height_to_diameter)) ** (1.0 / 3.0)
    return d, height_to_diameter * d


def capsule_centres(tank_w, tank_h, d_out, fraction, arrangement="staggered",
                    margin_frac=0.08):
        """Capsule centre positions filling the tank to roughly `fraction` by area.

        Planar section, so "volume fraction" becomes area fraction here; the
        count is chosen to hit the target area fraction rather than being read
        from the design vector, because a planar slice cannot hold the same
        number of capsules as the real cylinder.
        """
        m = margin_frac * tank_w
        x0, x1 = m, tank_w - m
        y0, y1 = m, tank_h - m
        a_cap = math.pi * (d_out / 2.0) ** 2
        target_n = max(1, int(fraction * tank_w * tank_h / a_cap))

        pitch = d_out * 1.35                      # leaves a water passage between capsules
        ncol = max(1, int((x1 - x0) // pitch))
        row_pitch = pitch * 0.9
        nrow_needed = max(1, -(-target_n // ncol))        # ceil
        nrow_fit = max(1, int((y1 - y0) // row_pitch))
        nrow = min(nrow_needed, nrow_fit)

        # Centre the capsule block vertically. Filling from the floor upward
        # (what an earlier revision did) puts every capsule in the bottom
        # third, which is not the geometry being modelled: the node model
        # places the capsules in a band anchored at the tank CENTRE
        # (`stack_anchor`), because capsules sitting only in the cold bottom
        # layer would barely see the hot charging water at all.
        block_h = (nrow - 1) * row_pitch + d_out
        y_start = y0 + max(0.0, ((y1 - y0) - block_h) / 2.0) + d_out / 2.0

        pts = []
        for r in range(nrow):
            y = y_start + r * row_pitch
            if y + d_out / 2 > y1:
                break
            offset = (pitch / 2.0) if (arrangement == "staggered" and r % 2) else 0.0
            for c in range(ncol):
                x = x0 + d_out / 2 + c * pitch + offset
                if x + d_out / 2 > x1:
                    break
                pts.append((x, y))
        return pts[:target_n], target_n


def write_system_msh(path: Path, volume_L=100.0, collector_area_m2=2.5,
                     r_in_m=0.020, wall_t_m=0.001, fraction=0.15,
                     arrangement="staggered", nx=260, ny=200,
                     _layout_out: dict = None) -> dict:
    tank_d, tank_h = tank_dimensions(volume_L)
    d_out = 2.0 * (r_in_m + wall_t_m)

    # Panel: a flat-plate collector of the configured area, drawn as a channel
    # of that height at the left of the domain. Depth (into the page) is unit,
    # so the drawn height is the collector's own height for a 1 m-wide panel.
    panel_h = min(collector_area_m2 / 1.0, tank_h * 1.15)
    panel_chan_w = 0.030            # water channel thickness
    panel_abs_w = 0.012             # absorber plate thickness
    gap = 0.055                     # space between panel and tank, holds the riser
    riser_w = 0.022

    panel_x1 = panel_abs_w + panel_chan_w
    tank_x0 = panel_x1 + gap
    tank_x1 = tank_x0 + tank_d

    # The riser is TWO horizontal connectors spanning the full gap, not a
    # free-standing vertical strip. An earlier revision drew it centred in the
    # gap with clearance on both sides, so it touched neither the panel nor
    # the tank: the solved case then had the absorber at 143 C while the tank,
    # PCM and riser all sat at exactly the 30.00 C initial value, because no
    # thermal path existed between collector and storage at all. Connecting at
    # top and bottom is also what the real pipework does -- hot water leaves
    # the top of the collector for the top of the tank, cold returns from the
    # bottom of the tank to the bottom of the collector.
    riser_y_bot = 0.030
    riser_y_top = tank_h - 0.030 - riser_w

    W = tank_x1
    H = max(tank_h, panel_h)
    tank_y0 = 0.0

    caps, target_n = capsule_centres(tank_d, tank_h, d_out, fraction, arrangement)
    caps = [(tank_x0 + cx, tank_y0 + cy) for cx, cy in caps]

    xs = [W * i / nx for i in range(nx + 1)]
    ys = [H * j / ny for j in range(ny + 1)]

    def classify(xc, yc):
        """Zone of the cell whose centroid is (xc, yc), or None if outside."""
        # panel assembly
        if xc < panel_x1 and yc <= panel_h:
            return "absorber" if xc < panel_abs_w else "panel"
        # the two connectors bridging panel and tank, top and bottom
        if panel_x1 <= xc <= tank_x0 and (
                riser_y_bot <= yc <= riser_y_bot + riser_w
                or riser_y_top <= yc <= riser_y_top + riser_w):
            return "riser"
        # tank
        if tank_x0 <= xc <= tank_x1 and tank_y0 <= yc <= tank_h:
            for (cx, cy) in caps:
                rr = math.hypot(xc - cx, yc - cy)
                if rr <= r_in_m:
                    return "pcm"
                if rr <= d_out / 2.0:
                    return "shell"
            return "water"
        return None

    # ---- classify, then renumber so each zone is a contiguous block ----
    # Cell ids are assigned zone by zone in ZONE_ORDER, and within a zone in
    # sorted (i, j) order. That ordering is not an implementation detail: it
    # is what lets a solved Fluent .dat file be mapped back onto this grid
    # (see plot_system_fields.py), so it must not change silently.
    tag = {}
    for i in range(nx):
        for j in range(ny):
            z = classify((xs[i] + xs[i + 1]) / 2.0, (ys[j] + ys[j + 1]) / 2.0)
            if z:
                tag[(i, j)] = z

    cid = {}
    zone_cells = defaultdict(list)
    next_id = 1
    for z in ZONE_ORDER:
        for key in sorted(k for k, v in tag.items() if v == z):
            cid[key] = next_id
            zone_cells[z].append(next_id)
            next_id += 1
    n_cells = next_id - 1
    layout = {"tag": tag, "cid": cid, "xs": xs, "ys": ys, "nx": nx, "ny": ny,
              "zone_order": list(ZONE_ORDER),
              "zone_sizes": [(z, len(zone_cells[z])) for z in ZONE_ORDER
                             if zone_cells[z]]}
    if _layout_out is not None:
        _layout_out.update(layout)

    # ---- nodes: keep only those touching a live cell, renumber ----
    used_nodes = set()
    for (i, j) in tag:
        used_nodes |= {(i, j), (i + 1, j), (i, j + 1), (i + 1, j + 1)}
    nid = {}
    for n, key in enumerate(sorted(used_nodes), start=1):
        nid[key] = n
    n_nodes = len(nid)

    # ---- faces ----
    bnd = defaultdict(list)          # boundary face zones
    inter = defaultdict(list)        # interior faces, per cell zone
    coupled = defaultdict(list)      # faces between two different cell zones

    def add(a, b, c0, c1, z0, z1):
        """Record one face. (a, b) must be ordered so the right-hand-rule
        normal points from c0 TO c1 -- that is the Fluent convention, and it
        is why a boundary face is written `n0 n1 cell 0`: the normal leaves
        the domain. A face whose nodes imply the opposite orientation is
        reversed here rather than at every call site.

        An earlier revision emitted vertical faces with c0/c1 swapped while
        horizontal faces were correct. Fluent read the whole file happily --
        right node, cell and face counts, every zone named -- and then failed
        in Fill_Domain with no clue as to why, in serial and in parallel
        alike. Mixed face orientation is the cause.
        """
        if c0 and c1:
            if z0 == z1:
                inter[z0].append((a, b, c0, c1))
            else:
                coupled[tuple(sorted((z0, z1)))].append((a, b, c0, c1))
            return
        if c0:
            c, z = c0, z0
        else:
            # the live cell is on the c1 side, so the normal points inward;
            # flip the nodes so it points out of the domain
            a, b = b, a
            c, z = c1, z1
        # ONE boundary face zone per adjacent cell zone. An earlier revision
        # lumped the panel and absorber exteriors into a single "panel_wall",
        # which made that face zone border two different cell zones; Fluent
        # splits such a zone on read ("Separating wall zone 18 into zones 18
        # and 21") and then dies in Fill_Domain. Naming by the zone the face
        # actually bounds keeps every face zone single-sided, and it is what
        # lets the solar flux be applied to the absorber alone.
        bnd[f"wall_{z}_ext"].append((a, b, c, 0))

    # Vertical faces (constant x), between (i-1,j) and (i,j).
    # Nodes go (i,j) -> (i,j+1), i.e. +y, so the right-hand normal is +x and
    # points at the RIGHT cell (i,j). Hence c0 = left, c1 = right.
    for i in range(nx + 1):
        for j in range(ny):
            left = tag.get((i - 1, j))
            right = tag.get((i, j))
            if not left and not right:
                continue
            a, b = nid[(i, j)], nid[(i, j + 1)]
            add(a, b,
                cid.get((i - 1, j), 0) if left else 0,
                cid.get((i, j), 0) if right else 0,
                left, right)

    # Horizontal faces (constant y), between (i,j-1) and (i,j).
    # Nodes go (i,j) -> (i+1,j), i.e. +x, so the normal is -y and points at
    # the cell BELOW. Hence c0 = above, c1 = below.
    for j in range(ny + 1):
        for i in range(nx):
            below = tag.get((i, j - 1))
            above = tag.get((i, j))
            if not below and not above:
                continue
            a, b = nid[(i, j)], nid[(i + 1, j)]
            add(a, b,
                cid.get((i, j), 0) if above else 0,
                cid.get((i, j - 1), 0) if below else 0,
                above, below)

    # ---- assemble zone table with ids ----
    zid = 1
    cell_zones, face_zones = [], []
    for z in ZONE_ORDER:
        if zone_cells[z]:
            zid += 1
            # PCM is declared FLUID, not solid. Fluent's solidification/melting
            # model is a fluid-zone model: the enthalpy-porosity formulation
            # needs a momentum equation to damp in the mushy region, and the
            # latent heat lives on a FLUID material (melting_heat / tsolidus /
            # tliqidus). A solid zone cannot melt. The shell and absorber are
            # genuinely solid and stay that way.
            kind = "fluid" if z in ("water", "panel", "riser", "pcm") else "solid"
            cell_zones.append((zid, z, kind, zone_cells[z][0], zone_cells[z][-1]))
    for z, faces in inter.items():
        if faces:
            zid += 1
            face_zones.append((zid, f"int_{z}", "interior", 2, faces))
    for (a, b), faces in coupled.items():
        if faces:
            zid += 1
            face_zones.append((zid, f"wall_{a}_{b}", "wall", 3, faces))
    for name, faces in bnd.items():
        if faces:
            zid += 1
            face_zones.append((zid, name, "wall", 3, faces))

    # --- self-check before anything is written -----------------------------
    # Fluent's failure mode for a malformed mesh is "Failed in Fill_Domain"
    # with no indication of which face or cell is at fault, so the invariants
    # are checked here where they can still be pointed at.
    face_count = defaultdict(int)
    for _zid, _name, _kind, _bc, faces in (
            [(0, k, 0, 0, v) for k, v in inter.items()]
            + [(0, k, 0, 0, v) for k, v in coupled.items()]
            + [(0, k, 0, 0, v) for k, v in bnd.items()]):
        for a, b, c0, c1 in faces:
            if c0:
                face_count[c0] += 1
            if c1:
                face_count[c1] += 1
            if not c0 and not c1:
                raise ValueError("face with no adjacent cell")
            if c0 == c1:
                raise ValueError(f"face with the same cell on both sides: {c0}")
    bad = {c: n for c, n in face_count.items() if n != 4}
    if bad or len(face_count) != n_cells:
        sample = list(bad.items())[:5]
        raise ValueError(
            f"quad mesh invariant violated: {len(bad)} cell(s) do not have "
            f"exactly 4 faces (e.g. {sample}); "
            f"{len(face_count)} cells referenced vs {n_cells} declared")

    inv_node = {v: k for k, v in nid.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="ascii") as f:
        f.write('(0 "Tank + PCM capsules + collector panel, '
                'src/cfd/make_system_mesh.py")\n')
        f.write("(2 2)\n")
        f.write(f"(10 (0 1 {n_nodes:x} 0 2))\n")
        f.write(f"(12 (0 1 {n_cells:x} 0 0))\n")
        n_faces = sum(len(fz[4]) for fz in face_zones)
        f.write(f"(13 (0 1 {n_faces:x} 0 0))\n")

        f.write(f"(10 (1 1 {n_nodes:x} 1 2)(\n")
        for n in range(1, n_nodes + 1):
            i, j = inv_node[n]
            f.write(f"{xs[i]:.9g} {ys[j]:.9g}\n")
        f.write("))\n")

        for z_id, name, kind, c0, c1 in cell_zones:
            f.write(f"(12 ({z_id:x} {c0:x} {c1:x} 1 3))\n")

        first = 1
        for z_id, name, kind, bc_code, faces in face_zones:
            last = first + len(faces) - 1
            f.write(f"(13 ({z_id:x} {first:x} {last:x} {bc_code} 2)(\n")
            for a, b, ca, cb in faces:
                f.write(f"{a:x} {b:x} {ca:x} {cb:x}\n")
            f.write("))\n")
            first = last + 1

        for z_id, name, kind, _c0, _c1 in cell_zones:
            f.write(f"(45 ({z_id} {kind} {name})())\n")
        for z_id, name, kind, _bc, _faces in face_zones:
            f.write(f"(45 ({z_id} {kind} {name})())\n")

    return {
        "path": str(path),
        "tank_diameter_m": round(tank_d, 5), "tank_height_m": round(tank_h, 5),
        "capsule_outer_d_m": round(d_out, 5), "n_capsules_drawn": len(caps),
        "target_n_for_fraction": target_n,
        "panel_height_m": round(panel_h, 4),
        "domain_W_m": round(W, 5), "domain_H_m": round(H, 5),
        "n_cells": n_cells, "n_nodes": n_nodes,
        "cell_zones": {z: len(zone_cells[z]) for z in ZONE_ORDER if zone_cells[z]},
        "face_zones": {fz[1]: len(fz[4]) for fz in face_zones},
    }


if __name__ == "__main__":
    import json
    import sys
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "ansys/system_tank_pcm_panel.msh")
    print(json.dumps(write_system_msh(out), indent=2))
