"""
Trace a rectangular house outline (3 clicks) on a scanned plan (PDF/PNG/JPG),
extrude it up to a flat, single pitched or double pitched roof, add windows,
panels and roof skylights,
visualize in 3D and export per-component geometry (azimuth, slope, areas) to CSV.
Windows and panels can be added (separate numbering, different colours).
The inputs are saved in <plan>.project.json so you can reopen and edit later.

Usage:   python plan_to_zone.py plan.pdf
Install: pip install numpy matplotlib pymupdf

Outputs: <plan>.project.json (inputs, for editing) and <plan>_components.csv (results)

Conventions
  azimuth : direction of the outward normal, clockwise from SOUTH
            (0=S, 90=W, 180=N, 270=E)
  slope   : floor=0, wall=90, flat roof=180; a roof pitched by an angle p has 180-p
  dist_corner_m : (windows/panels only) horizontal distance, measured along the
            parent wall, between the window/panel edge on the positive-azimuth
            side and the corner of the wall on that same side

Roof (menu option 2)
  flat           : the wall height
  single pitched : the heights of 3 points of the outline (the roof is the plane
                   through them)
  double pitched : horizontal ridge: its height and 2 points locating it, then the
                   height of one outline point on each side of the ridge. The roof
                   is split into roof_0 (on the left of the ridge when going from
                   the 1st to the 2nd ridge point) and roof_1.
  The top of the walls follows the roof, so the wall height can vary along a wall
  (windows/panels must stay below it); height_m of a wall = its mean height
  (area / width). Roof areas are the true (inclined) areas, volume = building volume.

Roof skylights (menu option 4): single rectangles in the roof, shown in blue.
  2 clicks = the opposite corners (diagonal). Flat roof: the edges are parallel
  to the outline. Pitched roof: one edge is parallel to the roof contour lines.
  csv: id skylight_0, skylight_1 ..., type skylight, parent = the roof (roof, or
  roof_0 / roof_1), same azimuth and slope as that roof; the roof gets a
  net_area_m2 (its area less its skylights).
  Pitched roof: 2 opposite corners; one edge is parallel to the contour lines of the
  roof, so width_m = length of that edge, height_m = the other dimension (along
  the slope). A skylight must lie on one side of the ridge.
  Reference point = the lowest point of the roof (part) on its positive-azimuth side.
  sill_height_m = distance between the horizontal lowest skylight edge and the
  reference point; dist_corner_m = distance between the sloped skylight edge
  nearest to the reference point and the reference point.
  Flat roof: the reference point is the positive-azimuth end of the outline wall
  whose azimuth is the nearest to South (the first one if several are as near).
  sill_height_m = distance between the reference point and the skylight edge
  nearest to it; width_m = length of that edge; height_m = the other dimension;
  dist_corner_m = distance between the reference point and the nearest skylight
  edge perpendicular to the sill edge.
"""
import sys, os, json, csv, math, time, queue, threading
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# A window/panel zone whose clicked points are farther than this (metres) from
# the nearest wall of the outline is skipped.
MAX_DIST_M = 0.5


# ---------------------------------------------------------------- helpers
def load_image(path, dpi=200):
    if path.lower().endswith(".pdf"):
        import fitz  # pymupdf
        pix = fitz.open(path)[0].get_pixmap(dpi=dpi)
        return np.frombuffer(pix.samples, np.uint8).reshape(pix.h, pix.w, pix.n)
    return plt.imread(path)


def get_input(prompt=""):
    """input() that keeps all matplotlib windows alive (rotatable 3D view, etc.)
    while waiting for what is typed in the terminal."""
    q = queue.Queue()

    def worker():
        try:
            q.put(input(prompt))
        except EOFError:
            q.put("q")

    threading.Thread(target=worker, daemon=True).start()
    while q.empty():
        for num in plt.get_fignums():
            try:
                plt.figure(num).canvas.flush_events()
            except Exception:
                pass
        time.sleep(0.03)
    return q.get()


def ask(prompt, default):
    while True:
        v = get_input(f"{prompt} [{default}]: ").strip()
        if not v:
            return float(default)
        try:
            return float(v)
        except ValueError:
            print("  Please enter a number.")


def shoelace(p):
    x, y = p[:, 0], p[:, 1]
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


def seg_proj(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1)
    return t, np.linalg.norm(p - (a + t * ab))


# ---------------------------------------------------------------- plan display
def rot_cw(p, W, H):
    """One clockwise quarter-turn of points p in a W x H frame (y down)."""
    return np.c_[H - p[:, 1], p[:, 0]], (H, W)


def rot_ccw(p, W, H):
    """Inverse of rot_cw; (W, H) is the size of the rotated frame."""
    return np.c_[p[:, 1], W - p[:, 0]], (H, W)


class Plan:
    """One persistent figure showing the plan, the outline, the window zones
    (numbered) and the North direction. All coordinates are stored in the frame
    of the ORIGINAL image; the plan can be displayed rotated (st['rotation'] =
    number of clockwise quarter-turns) without affecting the stored data."""

    def __init__(self, img):
        self.img = img
        self.H0, self.W0 = img.shape[:2]
        self.fig = None
        self.ax = None

    def _dims(self, k):
        return (self.W0, self.H0) if k % 2 == 0 else (self.H0, self.W0)

    def to_display(self, pts, k):
        p = np.array(pts, float).reshape(-1, 2)
        W, H = self.W0, self.H0
        for _ in range(k % 4):
            p, (W, H) = rot_cw(p, W, H)
        return p

    def to_original(self, pts, k):
        p = np.array(pts, float).reshape(-1, 2)
        W, H = self._dims(k)
        for _ in range(k % 4):
            p, (W, H) = rot_ccw(p, W, H)
        return p

    def show(self, st, title="", marks=()):
        if self.fig is None or not plt.fignum_exists(self.fig.number):
            self.fig, self.ax = plt.subplots(figsize=(10, 8))
        k = st.get("rotation", 0) % 4
        W, H = self._dims(k)
        ax = self.ax
        ax.clear()
        ax.imshow(np.rot90(self.img, -k), extent=(0, W, H, 0))
        if "outline_px" in st:
            o = self.to_display(st["outline_px"], k)
            c = np.vstack([o, o[:1]])
            ax.plot(c[:, 0], c[:, 1], "r-", lw=1)
        for w, lab in zip(st.get("windows", []), group_labels(st)):
            sg = self.to_display(w["px"], k)
            panel = w.get("kind", "window") == "panel"
            ax.plot(sg[:, 0], sg[:, 1], "-", color="orange" if panel else "c",
                    lw=8 if panel else 3, zorder=2 if panel else 3,
                    solid_capstyle="butt")
            m = sg.mean(axis=0) + np.array([0, 14 if panel else -14])
            ax.text(m[0], m[1], lab, color="darkorange" if panel else "blue",
                    fontsize=11, weight="bold", ha="center", zorder=4)
        ok = "outline_px" in st and "scale_m_per_px" in st
        sc = st.get("scale_m_per_px")
        rm = make_roof(st)[0] if ok else None
        roof = st.get("roof") or {}
        if roof.get("type") in ("single", "double"):        # roof definition (magenta)
            pp = self.to_display(roof["px"], k)
            ax.plot(pp[:, 0], pp[:, 1], "o", color="m", ms=6, zorder=4)
            for p_, z_ in zip(pp, roof["z"]):
                ax.text(p_[0] + 6, p_[1] - 6, f"{z_:g} m", color="m", fontsize=9,
                        zorder=4)
            if roof["type"] == "double":
                rg = self.to_display(roof["ridge_px"], k)
                ax.plot(rg[:, 0], rg[:, 1], "--", color="m", lw=2, zorder=3)
                mid = rg.mean(axis=0)
                ax.text(mid[0], mid[1] - 8, f"ridge {roof['ridge_z']:g} m", color="m",
                        fontsize=9, ha="center", zorder=4)
                if rm is not None:
                    for i_, part in enumerate(rm.parts):
                        cc = poly_centroid(part["poly"])
                        pc = self.to_display([[cc[0] / sc, -cc[1] / sc]], k)[0]
                        ax.text(pc[0], pc[1], f"R{i_}", color="m", fontsize=11,
                                weight="bold", ha="center", va="center", zorder=4)
        for j, sk in enumerate(st.get("skylights", [])):    # skylights (blue)
            c = sky_corners(sk, rm.pts, sc, rm)[0] if rm is not None else None
            if c is None:                       # not valid: show the clicks only
                sg = self.to_display(sk["px"], k)
                ax.plot(sg[:, 0], sg[:, 1], ":", color="c", lw=2, zorder=3)
                continue
            d = self.to_display(np.c_[c[:, 0] / sc, -c[:, 1] / sc], k)
            ax.fill(d[:, 0], d[:, 1], color="c", alpha=0.35, zorder=3)
            cl = np.vstack([d, d[:1]])
            ax.plot(cl[:, 0], cl[:, 1], "-", color="c", lw=2, zorder=3)
            m = d.mean(axis=0)
            ax.text(m[0], m[1], f"S{j}", color="blue", fontsize=11,
                    weight="bold", ha="center", va="center", zorder=4)
        for pts, col in marks:
            pm = self.to_display(pts, k)
            ax.plot(pm[:, 0], pm[:, 1], col + "-", lw=4)
        if "north_deg" in st:                               # North arrow
            th = math.radians(st["north_deg"] + 90 * k)     # as displayed
            L = 0.06 * min(W, H)
            p0 = np.array([0.12 * W, 0.12 * H])
            p1 = p0 + L * np.array([math.sin(th), -math.cos(th)])
            ax.annotate("", xy=p1, xytext=p0,
                        arrowprops=dict(arrowstyle="->", lw=2, color="green"))
            ax.text(*(p0 + 1.4 * (p1 - p0)), "N", color="green", fontsize=12,
                    weight="bold", ha="center", va="center")
        ax.set_title(title)
        self.fig.canvas.draw_idle()
        plt.pause(0.2)                                      # make sure it is painted

    def pick(self, st, title, n=-1, marks=()):
        """Left click = add point, right click = undo, Enter = finish.
        Returns points in the ORIGINAL image frame."""
        self.show(st, title, marks)
        pts = self.fig.ginput(n, timeout=0, show_clicks=True)
        if len(pts) == 0:
            return np.empty((0, 2))
        return self.to_original(pts, st.get("rotation", 0))

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
        self.fig = None


# ---------------------------------------------------------------- geometry helpers
def outline_m(st):
    """Outline in metres (y up), counter-clockwise."""
    s = st["scale_m_per_px"]
    px = np.array(st["outline_px"])
    pts = np.c_[px[:, 0] * s, -px[:, 1] * s]
    return pts[::-1] if shoelace(pts) < 0 else pts


def locate(pts, s, q):
    """For a zone given by 2 clicked pixel points q: index of the wall that fits
    best, its length, the zone limits (distances along the wall) and the largest
    distance (m) between a clicked point and that wall."""
    qm = np.c_[q[:, 0] * s, -q[:, 1] * s]
    n = len(pts)

    def dists(k):
        a, b = pts[k], pts[(k + 1) % n]
        return [seg_proj(x, a, b)[1] for x in qm]

    i = min(range(n), key=lambda k: sum(dists(k)))
    a, b = pts[i], pts[(i + 1) % n]
    L = float(np.linalg.norm(b - a))
    u0, u1 = sorted(seg_proj(x, a, b)[0] * L for x in qm)
    return i, L, u0, u1, max(dists(i))


def place_group(w, L, u0, u1):
    """Start positions (along the wall) of the windows of a group.
    The group (n*width + (n-1)*gap) is centred on the zone. Returns
    (starts, None), or (None, reason) if the group exceeds the wall."""
    cnt, wd = int(w["n"]), w["width"]
    gap = w["gap"] if cnt > 1 else 0.0
    if cnt < 1 or wd <= 0 or gap < 0:
        return None, "invalid count, width or gap"
    total = cnt * wd + (cnt - 1) * gap
    start = (u0 + u1) / 2 - total / 2
    if start < -1e-9 or start + total > L + 1e-9:
        return None, (f"the group ({total:.2f} m wide, centred on the zone) would "
                      f"span {start:.2f} to {start + total:.2f} m along a wall "
                      f"{L:.2f} m long, i.e. it exceeds the wall")
    start = min(max(start, 0.0), L - total)
    return [start + k * (wd + gap) for k in range(cnt)], None


# ---------------------------------------------------------------- roof
# The roof is flat, single pitched (one plane through 3 outline points) or double
# pitched (horizontal ridge, one plane on each side). Stored in st["roof"]:
#   dict(type="flat")                          wall height: st["height_m"]
#   dict(type="single", px=[3 clicks], z=[3 heights])
#   dict(type="double", ridge_px=[2 clicks], ridge_z=h, px=[2 clicks], z=[2 heights])
# Clicks are pixels of the original image; outline points are projected onto the
# outline. Projects saved before the roof types existed have no "roof" key: they
# are flat roofs of height st["height_m"].
SNAP_M = 0.15          # a clicked outline point this close to a corner snaps to it


def vec_azimuth(v, north):
    """Azimuth (deg, clockwise from South) of a horizontal direction v (metres
    frame, y up) for a plan whose North is `north` deg clockwise from the top."""
    return (math.degrees(math.atan2(v[0], v[1])) - north - 180) % 360


def snap_to_outline(pts, p):
    """Point of the outline polygon nearest to p (a corner if within SNAP_M)."""
    n = len(pts)
    k = min(range(n), key=lambda k: np.linalg.norm(pts[k] - p))
    if np.linalg.norm(pts[k] - p) <= SNAP_M:
        return pts[k].copy()
    best = None
    for k in range(n):
        a, b = pts[k], pts[(k + 1) % n]
        t, d = seg_proj(p, a, b)
        if best is None or d < best[0]:
            best = (d, a + t * (b - a))
    return best[1]


def clip_half(poly, q0, nl, sgn):
    """Part of a polygon where sgn*(p - q0).nl >= 0 (Sutherland-Hodgman)."""
    out, n = [], len(poly)
    f = lambda p: sgn * float((p - q0) @ nl)
    for i in range(n):
        p, q = poly[i], poly[(i + 1) % n]
        fp, fq = f(p), f(q)
        if fp >= 0:
            out.append(p)
        if fp * fq < 0:
            out.append(p + fp / (fp - fq) * (q - p))
    return np.array(out)


def poly_centroid(poly):
    x, y = poly[:, 0], poly[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y1 - x1 * y
    A = cr.sum() / 2
    if abs(A) < 1e-12:
        return poly.mean(axis=0)
    return np.array([((x + x1) * cr).sum(), ((y + y1) * cr).sum()]) / (6 * A)


class Roof:
    """Roof surface(s) above the outline `pts` (metres, y up, counter-clockwise).
    parts[i] = dict(id, plane=(a, b, c) with z = a*x + b*y + c, poly=plan polygon).
    Flat / single pitched: one part "roof". Double pitched: "roof_0" (left of the
    ridge going from its 1st to its 2nd point) and "roof_1"; ridge=(q0, nl, hr)
    with q0 a ridge point, nl the unit normal pointing to the left and hr its height."""

    def __init__(self, kind, pts, planes, ridge=None):
        self.kind, self.pts, self.ridge = kind, pts, ridge
        if ridge is None:
            polys, ids = [pts], ["roof"]
        else:
            polys = [clip_half(pts, ridge[0], ridge[1], 1),
                     clip_half(pts, ridge[0], ridge[1], -1)]
            ids = ["roof_0", "roof_1"]
        self.parts = [dict(id=i, plane=pl, poly=po)
                      for i, pl, po in zip(ids, planes, polys)]

    def part_of(self, p):
        if self.ridge is None:
            return 0
        return 0 if float((np.asarray(p) - self.ridge[0]) @ self.ridge[1]) >= 0 else 1

    def z(self, p, i=None):
        a, b, c = self.parts[self.part_of(p) if i is None else i]["plane"]
        return a * p[0] + b * p[1] + c

    def facing(self, i):
        """Horizontal direction the part faces (downslope), None if horizontal."""
        a, b, _ = self.parts[i]["plane"]
        return None if math.hypot(a, b) < 1e-9 else np.array([-a, -b])

    def factor(self, i):
        """Area of the part divided by the area of its plan projection."""
        a, b, _ = self.parts[i]["plane"]
        return math.sqrt(1 + a * a + b * b)

    def pitch_deg(self, i):
        a, b, _ = self.parts[i]["plane"]
        return math.degrees(math.atan(math.hypot(a, b)))

    def profile(self, a, b):
        """Breakpoints (u, z) of the top of the wall a->b, u = distance from a."""
        L = float(np.linalg.norm(b - a))
        prof = [(0.0, self.z(a))]
        if self.ridge is not None:                 # the ridge may cross the wall
            q0, nl, hr = self.ridge
            da, db = float((a - q0) @ nl), float((b - q0) @ nl)
            if da * db < -1e-12:
                prof.append((da / (da - db) * L, hr))
        prof.append((L, self.z(b)))
        return prof

    def wall_area(self, a, b):
        prof = self.profile(a, b)
        return sum((u1 - u0) * (z0 + z1) / 2 for (u0, z0), (u1, z1) in zip(prof, prof[1:]))

    def top_min(self, a, b, u0, u1):
        """Lowest wall top between u0 and u1 along the wall a->b."""
        prof = self.profile(a, b)
        xs, ys = [p[0] for p in prof], [p[1] for p in prof]
        us = [u0, u1] + [u for u in xs if u0 < u < u1]
        return float(min(np.interp(u, xs, ys) for u in us))

    def zmax(self):
        return max(self.z(v, i) for i, p in enumerate(self.parts) for v in p["poly"])

    def volume(self):
        return sum(shoelace(p["poly"]) * self.z(poly_centroid(p["poly"]), i)
                   for i, p in enumerate(self.parts))


def make_roof(st):
    """(Roof, None) from the project, or (None, reason) if the roof is missing
    or invalid."""
    if "outline_px" not in st or "scale_m_per_px" not in st:
        return None, "the scale and the outline are required first"
    s, pts = st["scale_m_per_px"], outline_m(st)
    r = st.get("roof") or dict(type="flat")
    kind = r.get("type", "flat")

    def mm(px):
        px = np.asarray(px, float)
        return np.c_[px[:, 0] * s, -px[:, 1] * s]

    if kind == "flat":
        H = st.get("height_m")
        if H is None or H <= 0:
            return None, "the wall height is not defined (option 2)"
        return Roof("flat", pts, [(0.0, 0.0, float(H))]), None
    try:
        if kind == "single":
            P = np.array([snap_to_outline(pts, p) for p in mm(r["px"])[:3]])
            M = np.c_[P, np.ones(3)]
            if abs(np.linalg.det(M)) < 1e-6:
                return None, "the 3 roof points are aligned"
            roof = Roof("single", pts,
                        [tuple(np.linalg.solve(M, np.array(r["z"], float)[:3]))])
        elif kind == "double":
            q0, q1 = mm(r["ridge_px"])[:2]
            e = q1 - q0
            if np.linalg.norm(e) < SKY_MIN_M:
                return None, "the 2 ridge points are identical"
            e = e / np.linalg.norm(e)
            nl = np.array([-e[1], e[0]])                  # normal to the left
            hr = float(r["ridge_z"])
            P = [snap_to_outline(pts, p) for p in mm(r["px"])[:2]]
            d = [float((p - q0) @ nl) for p in P]
            if min(abs(x) for x in d) < 1e-6 or d[0] * d[1] > 0:
                return None, "the 2 outline points must be on each side of the ridge"
            h = [float(x) for x in r["z"]][:2]
            if max(h) > hr + 1e-9:
                return None, "the outline heights cannot exceed the ridge height"
            left = 0 if d[0] > 0 else 1                  # index of the left point
            planes = [None, None]
            for part, j in ((0, left), (1, 1 - left)):
                k = (h[j] - hr) / d[j]                   # z = hr + k * (p - q0).nl
                planes[part] = (k * nl[0], k * nl[1], hr - k * float(q0 @ nl))
            roof = Roof("double", pts, planes, (q0, nl, hr))
        else:
            return None, f"unknown roof type {kind!r}"
    except (KeyError, ValueError, IndexError, TypeError, np.linalg.LinAlgError):
        return None, "the roof definition is incomplete (option 2)"
    for p in roof.parts:
        if len(p["poly"]) < 3 or shoelace(p["poly"]) < 1e-6:
            return None, "the ridge does not cross the outline"
    low = min(roof.z(v, i) for i, p in enumerate(roof.parts) for v in p["poly"])
    if low < -1e-9:
        return None, "the roof would be below the ground at an outline corner"
    return roof, None


# Zone version, flat roof: the outline is a rectangle and the skylight edges are
# parallel to it, so a skylight is given by its diagonal (2 opposite corners).
SKY_FLAT_CLICKS = 2
SKY_FLAT_PROMPT = ("Skylight: click 2 opposite corners (diagonal; edges are "
                   "parallel to the outline) (right=undo)")
SKY_INVALID = "it will be removed at the next build"


def flat_corners(q, pts, s):
    """Plan corners (4x2, metres) of a skylight of a flat roof (diagonal). An
    excess over the outline up to SKY_TOL_M is clipped to the outline."""
    if len(q) < 2:
        return None, "it needs 2 points"
    qm = np.c_[q[:2, 0] * s, -q[:2, 1] * s]
    o = pts[0]
    u, v = pts[1] - o, pts[3] - o                # the two sides at corner 0
    lu, lv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    u, v = u / lu, v / lv
    xs, ys = sorted((qm - o) @ u), sorted((qm - o) @ v)
    if (xs[0] < -SKY_TOL_M or xs[1] > lu + SKY_TOL_M
            or ys[0] < -SKY_TOL_M or ys[1] > lv + SKY_TOL_M):
        return None, "it is not entirely inside the outline"
    x0, x1 = max(xs[0], 0.0), min(xs[1], lu)
    y0, y1 = max(ys[0], 0.0), min(ys[1], lv)
    if x1 - x0 < SKY_MIN_M or y1 - y0 < SKY_MIN_M:
        return None, f"it is smaller than {SKY_MIN_M} m"
    return np.array([o + x0 * u + y0 * v, o + x1 * u + y0 * v,
                     o + x1 * u + y1 * v, o + x0 * u + y1 * v]), None


def prune_skylights(st, rm):
    """Permanently remove the skylights that cannot be kept (outside the outline,
    too small, straddling the ridge, overlapping)."""
    acc, dropped = check_skylights(st, rm)
    if dropped:
        for _, lab, why in dropped:
            print(f"  -> skylight {lab} removed from the project: {why}")
        gone = {k for k, _, _ in dropped}
        st["skylights"] = [w for k, w in enumerate(st["skylights"]) if k not in gone]
        print(f"  {len(gone)} skylight(s) removed. The remaining ones are renumbered "
              f"(see option 6).")
    return acc


# ---------------------------------------------------------------- skylights
# A skylight is ONE rectangle lying in the roof (a roof part). Only the clicked
# pixels are stored (st["skylights"] = [dict(px=...)], original image frame), so
# the scale, the outline, the roof or the display rotation can be changed afterwards.
# Projects saved before skylights existed simply have no "skylights" key.
#   flat roof    : see the file-specific definition (SKY_FLAT_*)
#   pitched roof : 2 opposite corners; one edge is parallel to the roof contour
#                  lines (horizontal), the other one follows the slope. The
#                  skylight must lie on one side of the ridge.
SKY_TOL_M = 0.02     # a skylight may exceed the outline by this much (m)
SKY_MIN_M = 0.05     # smallest accepted skylight dimension (m)
SKY_PITCHED_PROMPT = ("Skylight: click 2 opposite corners (one edge is parallel "
                      "to the roof contour lines) (right=undo)")


def edge_azimuth(a, b, north):
    """Azimuth (deg, clockwise from South) of the outward normal of wall a->b
    (counter-clockwise outline)."""
    return vec_azimuth((b[1] - a[1], -(b[0] - a[0])), north)


def ref_edge(pts, north):
    """Reference wall of a horizontal roof: the outline edge whose azimuth is the
    nearest to 0 (South); the first one if several are equally near.
    Returns (index, a, b): a is the end of the wall on its positive-azimuth side
    (the same corner as for windows/panels), b the other end."""
    n = len(pts)

    def off(i):
        az = edge_azimuth(pts[i], pts[(i + 1) % n], north)
        return min(az, 360 - az)

    i = min(range(n), key=off)
    return i, pts[i], pts[(i + 1) % n]


def rects_overlap(c1, c2, tol=1e-6):
    """True if two rectangles (4x2 arrays) overlap by more than tol (separating
    axis test). Touching edges are not an overlap."""
    for c in (c1, c2):
        for i in range(2):
            e = c[i + 1] - c[i]
            ax = np.array([-e[1], e[0]]) / np.linalg.norm(e)
            p1, p2 = c1 @ ax, c2 @ ax
            if p1.max() <= p2.min() + tol or p2.max() <= p1.min() + tol:
                return False
    return True


def point_in_poly(p, poly):
    """True if p is inside the polygon or on its boundary."""
    inside, n = False, len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        if seg_proj(p, a, b)[1] < 1e-9:
            return True
        if (a[1] > p[1]) != (b[1] > p[1]) and \
                p[0] < a[0] + (p[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1]):
            inside = not inside
    return inside


def seg_cross(p, q, a, b):
    """True if segments pq and ab cross each other properly."""
    def orient(u, v, w):
        return (v[0] - u[0]) * (w[1] - u[1]) - (v[1] - u[1]) * (w[0] - u[0])
    return (orient(p, q, a) * orient(p, q, b) < -1e-12
            and orient(a, b, p) * orient(a, b, q) < -1e-12)


def shrink_rect(c, t):
    """Rectangle c (4x2) with each side moved inwards by t."""
    u, v = c[1] - c[0], c[3] - c[0]
    lu, lv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
    if lu <= 2 * t or lv <= 2 * t:
        return c
    u, v = u / lu * t, v / lv * t
    return np.array([c[0] + u + v, c[1] - u + v, c[2] - u - v, c[3] + u - v])


def rect_in_outline(c, pts):
    """True if the rectangle c (plan, 4x2) is inside the outline polygon
    (up to SKY_TOL_M)."""
    r, n = shrink_rect(c, SKY_TOL_M), len(pts)
    return (all(point_in_poly(p, pts) for p in r)
            and not any(seg_cross(r[i], r[(i + 1) % 4], pts[k], pts[(k + 1) % n])
                        for i in range(4) for k in range(n))
            and not any(point_in_poly(v, r) for v in pts))   # no outline corner inside


def part_axes(rm, i):
    """Unit vectors (downslope, contour) in plan for roof part i. The contour
    vector points to the positive-azimuth side. For a horizontal part the axes
    follow the outline edge 0."""
    g = rm.facing(i)
    if g is None:
        d = rm.pts[1] - rm.pts[0]
        g = np.array([d[1], -d[0]])
    gd = g / np.linalg.norm(g)
    return gd, np.array([gd[1], -gd[0]])


def ref_point(rm, i, north):
    """Reference point of roof part i: its lowest point on the positive-azimuth
    side. For a horizontal part: the positive-azimuth end of the wall whose
    azimuth is the nearest to South."""
    g = rm.facing(i)
    if g is None:
        return ref_edge(rm.pts, north)[1]
    V = rm.parts[i]["poly"]
    z = np.array([rm.z(v, i) for v in V])
    low = [k for k in range(len(V)) if z[k] <= z.min() + 1e-6]
    p = np.array([g[1], -g[0]])                       # towards positive azimuth
    return V[max(low, key=lambda k: float(V[k] @ p))]


def sky_corners(sk, pts, s, rm):
    """Corners (4x2, plan, metres) of a skylight, or (None, reason) if it is not
    valid (degenerate, too small, not inside the outline, straddling the ridge)."""
    q = np.asarray(sk["px"], float)
    if rm.kind == "flat":
        c, why = flat_corners(q, pts, s)
        if c is None:
            return None, why
    else:
        if len(q) != 2:
            return None, "it was defined for a flat roof - redefine it"
        qm = np.c_[q[:, 0] * s, -q[:, 1] * s]
        gd, cd = part_axes(rm, rm.part_of(qm.mean(axis=0)))
        sx, tx = sorted(qm @ cd), sorted(qm @ gd)
        c = np.array([sx[0] * cd + tx[0] * gd, sx[1] * cd + tx[0] * gd,
                      sx[1] * cd + tx[1] * gd, sx[0] * cd + tx[1] * gd])
    if min(np.linalg.norm(c[1] - c[0]), np.linalg.norm(c[3] - c[0])) < SKY_MIN_M:
        return None, f"it is smaller than {SKY_MIN_M} m"
    if not rect_in_outline(c, pts):
        return None, "it is not entirely inside the outline"
    if rm.ridge is not None:
        q0, nl, _ = rm.ridge
        sgn = 1 if rm.part_of(c.mean(axis=0)) == 0 else -1
        if any(sgn * float((p - q0) @ nl) < -SKY_TOL_M for p in c):
            return None, "it straddles the roof ridge"
    return c, None


def sky_params(c, rm, i, north):
    """(width, height, sill_height, dist_corner) of the skylight with plan
    corners c on roof part i; P is the reference point of the part.
    Pitched part: width = length of the edge parallel to the contour lines,
      height = the other dimension (along the slope), sill_height = distance
      between P and the lowest (horizontal) skylight edge, dist_corner = distance
      between P and the nearest sloped skylight edge (distances to the edge lines,
      measured in the roof plane).
    Horizontal part: sill_height = distance between P and the nearest skylight
      edge (the first one in case of a tie: the edge the most parallel to the
      reference wall), width = its length, height = the other dimension,
      dist_corner = distance between P and the nearest of the 2 edges
      perpendicular to it."""
    P = ref_point(rm, i, north)
    if rm.facing(i) is not None:
        gd, cd = part_axes(rm, i)
        f = rm.factor(i)
        sc, tc = c @ cd, c @ gd
        sP, tP = float(P @ cd), float(P @ gd)
        return (float(sc.max() - sc.min()), float((tc.max() - tc.min()) * f),
                abs(tP - float(tc.max())) * f,
                min(abs(sP - float(sc.min())), abs(sP - float(sc.max()))))
    _, ra, rb = ref_edge(rm.pts, north)
    dref = (rb - ra) / np.linalg.norm(rb - ra)
    ln, dist = [], []
    for k in range(4):
        e = c[(k + 1) % 4] - c[k]
        ln.append(float(np.linalg.norm(e)))
        nrm = np.array([-e[1], e[0]]) / ln[-1]
        dist.append(abs(float((P - c[k]) @ nrm)))
    key = lambda k: (round(dist[k], 6),
                     -abs(float((c[(k + 1) % 4] - c[k]) @ dref)) / ln[k])
    k = min(range(4), key=key)
    return ln[k], ln[(k + 1) % 4], dist[k], min(dist[(k + 1) % 4], dist[(k + 3) % 4])


def check_skylights(st, rm):
    """Skylights are taken in creation order. Returns (acc, dropped): the valid,
    non-overlapping ones as (index, corners), and a list of (index, label, reason)
    for those that must be skipped."""
    s = st["scale_m_per_px"]
    acc, dropped = [], []
    for k, sk in enumerate(st.get("skylights", [])):
        c, why = sky_corners(sk, rm.pts, s, rm)
        if c is None:
            dropped.append((k, f"S{k}", why))
            continue
        bad = next((j for j, o in acc if rects_overlap(c, o)), None)
        if bad is not None:
            dropped.append((k, f"S{k}", f"it overlaps skylight S{bad}"))
            continue
        acc.append((k, c))
    return acc, dropped


def accepted_skylights(st, rm, verbose=True):
    """All valid, non-overlapping skylights as (index, corners)."""
    acc, dropped = check_skylights(st, rm)
    if verbose:
        for _, lab, why in dropped:
            print(f"  -> skylight {lab} ignored: {why}")
    return acc


def add_skylight(plan, st):
    """One single rectangular skylight on the roof."""
    rm, why = make_roof(st)
    if rm is None:
        print(f"Roof or outline not ready: {why}.")
        return
    flat = rm.kind == "flat"
    n = SKY_FLAT_CLICKS if flat else 2
    q = plan.pick(st, SKY_FLAT_PROMPT if flat else SKY_PITCHED_PROMPT, n)
    if len(q) < n:
        print(f"Need exactly {n} points, skylight not added.")
        return
    new = dict(px=q.tolist())
    c, why = sky_corners(new, rm.pts, st["scale_m_per_px"], rm)
    if c is None:
        print(f"  SKIPPED: skylight not added - {why}.")
        return
    for k, o in accepted_skylights(st, rm, verbose=False):
        if rects_overlap(c, o):
            print(f"  SKIPPED: skylight not added - it overlaps skylight S{k}.")
            return
    sky = st.setdefault("skylights", [])
    sky.append(new)
    msg = f"Skylight S{len(sky) - 1} added"
    if "north_deg" in st:
        i = rm.part_of(c.mean(axis=0))
        w, h, sill, dc = sky_params(c, rm, i, st["north_deg"])
        msg += (f" on {rm.parts[i]['id']}: width {w:.2f} m, height {h:.2f} m, "
                f"sill height {sill:.2f} m, dist_corner {dc:.2f} m")
    print(msg + ".")


# ---------------------------------------------------------------- project inputs
# Everything is stored in image pixels (original frame), so changing the scale,
# the outline or the display rotation never breaks the other items.
def rotate(st):
    st["rotation"] = (st.get("rotation", 0) + 1) % 4
    print(f"  Plan rotated: {90 * st['rotation']} deg clockwise from the original.")


def set_scale(plan, st):
    p = plan.pick(st, "Click 2 points of a known distance", 2)
    if len(p) < 2:
        return
    plan.show(st, "Answer in the terminal", marks=[(p, "m")])
    d = ask("Real distance between them (m)", 1)
    st["scale_m_per_px"] = d / float(np.linalg.norm(p[0] - p[1]))


def set_north(plan, st):
    k = st.get("rotation", 0) % 4
    plan.show(st, "Answer in the terminal")
    shown = (st.get("north_deg", 0) + 90 * k) % 360
    ans = ask("Angle from the top of the DISPLAYED plan, clockwise, to North (deg)",
              round(shown, 2))
    st["north_deg"] = (ans - 90 * k) % 360              # stored in the original frame
    plan.show(st, "North direction updated (green arrow)")


def ask_height(prompt, default, hi=None, positive=False):
    """A height (m): >= 0 (> 0 if positive), and <= hi if given."""
    while True:
        v = ask(prompt, default)
        if v < 0 or (positive and v == 0):
            print("  The height must be positive." if positive
                  else "  The height cannot be negative.")
        elif hi is not None and v > hi + 1e-9:
            print(f"  The height cannot exceed {hi:g} m (the ridge).")
        else:
            return v


def set_roof(plan, st):
    """Roof type and heights. Flat: the wall height. Single pitched: the heights of
    3 points of the outline. Double pitched (horizontal ridge): the ridge height
    and 2 points locating it, then the height of one outline point on each side of
    the ridge."""
    old = st.get("roof") or {}
    cur = old.get("type", "flat")
    plan.show(st, "Answer in the terminal")
    while True:
        a = get_input("Roof: [f]lat, [s]ingle pitched or [d]ouble pitched? "
                      f"(Enter = {cur}): ").strip().lower() or cur[0]
        if a[0] in "fsd":
            break
        print("  Please answer f, s or d.")
    kind = dict(f="flat", s="single", d="double")[a[0]]
    h0 = st.get("height_m", 2.5)
    if kind == "flat":
        st["roof"] = dict(type="flat")
        st["height_m"] = ask_height("Wall height (m)", h0, positive=True)
        return
    if "outline_px" not in st or "scale_m_per_px" not in st:
        print("  A pitched roof needs the scale and the outline first (options 1 "
              "and 3), then choose option 2 again. Roof unchanged.")
        return
    keep = old if old.get("type") == kind else {}
    if kind == "single":
        q = plan.pick(st, "Single pitched roof: click 3 points of the outline "
                          "(not aligned) (right=undo)", 3)
        if len(q) < 3:
            print("Need exactly 3 points, roof unchanged.")
            return
        plan.show(st, "Answer in the terminal", marks=[(q, "m")])
        dz = keep.get("z", [h0] * 3)
        z = [ask_height(f"Height of outline point {i + 1} (m)", dz[i])
             for i in range(3)]
        new = dict(type="single", px=q.tolist(), z=z)
    else:
        hr = ask_height("Height of the roof ridge (m)", keep.get("ridge_z", h0 + 1.5),
                        positive=True)
        r = plan.pick(st, "Double pitched roof: click the 2 points locating the "
                          "ridge line (right=undo)", 2)
        if len(r) < 2:
            print("Need exactly 2 points, roof unchanged.")
            return
        q = plan.pick(st, "Click 1 point of the outline on EACH side of the ridge "
                          "(right=undo)", 2, marks=[(r, "m")])
        if len(q) < 2:
            print("Need exactly 2 points, roof unchanged.")
            return
        plan.show(st, "Answer in the terminal", marks=[(r, "m"), (q, "g")])
        dz = keep.get("z", [h0] * 2)
        z = [ask_height(f"Height of outline point {i + 1} (m)", min(dz[i], hr), hi=hr)
             for i in range(2)]
        new = dict(type="double", ridge_px=r.tolist(), ridge_z=hr, px=q.tolist(), z=z)
    roof, why = make_roof(dict(st, roof=new))
    if roof is None:
        print(f"  Roof refused: {why}. Roof unchanged.")
        return
    st["roof"] = new
    for i_, p_ in enumerate(roof.parts):
        print(f"  {p_['id']}: pitch {roof.pitch_deg(i_):.1f} deg")


def set_params(plan, st):
    set_north(plan, st)
    set_roof(plan, st)


def rect_from_3(p):
    """Rectangle from 3 clicked points: p[0]-p[1] is one side, and p[2] only
    gives the distance (and the side) of the opposite parallel side. The 4
    corners are exactly perpendicular. Returns None if degenerate."""
    a, b, c = (np.asarray(x, float) for x in p[:3])
    ab = b - a
    length = float(np.linalg.norm(ab))
    if length < 1.0:
        return None
    u = ab / length
    nrm = np.array([-u[1], u[0]])
    d = float(np.dot(c - a, nrm))
    if abs(d) < 1.0:                       # third point lies on the first side
        return None
    return np.array([a, b, b + d * nrm, a + d * nrm])


def set_outline(plan, st):
    """The outline is a rectangle: click the 2 ends of one side, then one point
    on the opposite side (right-click = undo the last point)."""
    px = plan.pick(st, "Rectangle: click the 2 ends of one side, then a point "
                       "on the opposite side (right=undo)", 3)
    if len(px) < 3:
        print("Need exactly 3 points, outline unchanged.")
        return
    rect = rect_from_3(px)
    if rect is None:
        print("Degenerate rectangle (2 identical points, or the 3rd point lies "
              "on the first side), outline unchanged.")
        return
    st["outline_px"] = rect.tolist()
    s = st.get("scale_m_per_px")
    if s:
        w = np.linalg.norm(rect[1] - rect[0]) * s
        h = np.linalg.norm(rect[3] - rect[0]) * s
        print(f"  Rectangle: {w:.2f} m x {h:.2f} m")


def group_labels(st):
    """Labels W0, W1... for window groups and P0, P1... for panel groups
    (each kind has its own numbering order)."""
    cnt, out = {"window": 0, "panel": 0}, []
    for w in st.get("windows", []):
        kd = w.get("kind", "window")
        out.append(("P" if kd == "panel" else "W") + str(cnt[kd]))
        cnt[kd] += 1
    return out


def group_rects(w, pts, s, rm):
    """Individual rectangles (wall, u0, u1, z0, z1, kind) of a group of
    windows/panels, or (None, reason) if the group must be skipped."""
    i, L, u0, u1, dist = locate(pts, s, np.array(w["px"]))
    if dist > MAX_DIST_M:
        return None, (f"the zone is too far from the outline ({dist:.2f} m from "
                      f"the nearest wall, limit {MAX_DIST_M} m)")
    starts, why = place_group(w, L, u0, u1)
    if starts is None:
        return None, why
    if w["sill"] < 0 or w["h"] <= 0:
        return None, "invalid sill or height"
    a, b = pts[i], pts[(i + 1) % len(pts)]
    for x in starts:
        top = rm.top_min(a, b, x, x + w["width"])
        if w["sill"] + w["h"] > top + 1e-9:
            return None, (f"sill + height ({w['sill'] + w['h']:.2f} m) exceeds the "
                          f"wall height ({top:.2f} m) where the "
                          f"{w.get('kind', 'window')} is")
    kd = w.get("kind", "window")
    return [dict(wall=i, u0=x, u1=x + w["width"], z0=w["sill"],
                 z1=w["sill"] + w["h"], kind=kd) for x in starts], None


def find_conflict(r, others, tol=1e-6):
    """First rectangle of `others` overlapping r (same wall, overlapping both
    along the wall AND in height). Touching edges are not a conflict."""
    for o in others:
        if (o["wall"] == r["wall"]
                and min(r["u1"], o["u1"]) - max(r["u0"], o["u0"]) > tol
                and min(r["z1"], o["z1"]) - max(r["z0"], o["z0"]) > tol):
            return o
    return None


def check_groups(st, pts):
    """Groups are taken in creation order. Returns (rects, dropped): the valid,
    non-conflicting rectangles, and a list of (index, label, reason) for the
    groups that must be skipped."""
    s = st["scale_m_per_px"]
    rm = make_roof(st)[0]
    acc, dropped = [], []
    if rm is None:
        return acc, dropped
    for idx, (lab, w) in enumerate(zip(group_labels(st), st["windows"])):
        rects, why = group_rects(w, pts, s, rm)
        if rects is None:
            dropped.append((idx, lab, why))
            continue
        bad = None
        for r in rects:
            bad = find_conflict(r, acc)
            if bad:
                break
        if bad:
            dropped.append((idx, lab, f"it overlaps a {bad['kind']} on wall "
                                      f"{bad['wall']}"))
            continue
        acc += rects
    return acc, dropped


def accepted_rects(st, pts, verbose=True):
    """All valid, non-conflicting rectangles (the project is not modified)."""
    acc, dropped = check_groups(st, pts)
    if verbose:
        for _, lab, why in dropped:
            print(f"  -> group {lab} ignored: {why}")
    return acc


def prune_groups(st, pts):
    """Permanently remove from the project the groups that cannot be kept
    (exceeding their wall, too far from the outline, too tall, overlapping)."""
    acc, dropped = check_groups(st, pts)
    if dropped:
        for _, lab, why in dropped:
            print(f"  -> group {lab} removed from the project: {why}")
        gone = {i for i, _, _ in dropped}
        st["windows"] = [w for i, w in enumerate(st["windows"]) if i not in gone]
        print(f"  {len(gone)} group(s) removed. The remaining groups are renumbered "
              f"(see option 6).")
    return acc


def add_window(plan, st):
    """A group of n identical windows OR panels, centred on a zone clicked
    along one wall. A group that exceeds the wall, is too far from the outline
    or overlaps an existing element is skipped with a message."""
    rm, why = make_roof(st)
    if rm is None:
        print(f"Roof or outline not ready: {why}.")
        return
    plan.show(st, "Answer in the terminal")
    while True:
        a = get_input("Window or panel? [w/p] (Enter = window): ").strip().lower() or "w"
        if a[0] in "wp":
            break
        print("  Please answer w or p.")
    kind = "panel" if a[0] == "p" else "window"
    q = plan.pick(st, f"Click the 2 ends of the zone (along ONE wall) "
                      f"on which the {kind}s are centred", 2)
    if len(q) < 2:
        return
    plan.show(st, "Zone selected - answer in the terminal", marks=[(q, "m")])
    s = st["scale_m_per_px"]
    pts = outline_m(st)
    i, L, u0, u1, dist = locate(pts, s, q)
    if dist > MAX_DIST_M:
        print(f"  SKIPPED: the zone is too far from the outline ({dist:.2f} m from "
              f"the nearest wall, limit {MAX_DIST_M} m).")
        return
    zl = u1 - u0
    qm = np.c_[q[:, 0] * s, -q[:, 1] * s]
    drawn = float(np.linalg.norm(qm[1] - qm[0]))
    if drawn > zl + 0.10:
        print(f"  Note: the line drawn is {drawn:.2f} m long but only {zl:.2f} m of it "
              f"lies along this wall (the rest goes beyond the wall ends, or the "
              f"line is not parallel to the wall).")
    print(f"Wall {i}: length {L:.2f} m   Zone length: {zl:.2f} m")
    n = int(ask(f"Number of identical {kind}s in this group", 1))
    if n < 1:
        print("  SKIPPED: the number must be at least 1.")
        return
    dw = zl / (2 * n - 1)
    width = ask(f"Width of each {kind} (m)", round(dw if dw >= 0.05 else 1.0, 3))
    if n > 1:
        dg = max((zl - n * width) / (n - 1), 0.0)
        gap = ask(f"Gap between {kind}s (m)", round(dg, 3))
    else:
        gap = 0.0
    d_sill, d_h = (0.9, 1.2) if kind == "window" else (0.0, 0.9)
    sill = ask("Sill height (m)", d_sill)
    h = ask(f"{kind.capitalize()} height (m)", d_h)

    new = dict(kind=kind, px=q.tolist(), n=n, width=width, gap=gap, sill=sill, h=h)
    rects, why = group_rects(new, pts, s, rm)
    if rects is None:
        print(f"  SKIPPED: {kind} group not added - {why}.")
        return
    existing = accepted_rects(st, pts, verbose=False)
    for r in rects:
        o = find_conflict(r, existing)
        if o:
            print(f"  SKIPPED: {kind} group not added - it overlaps a {o['kind']} on "
                  f"wall {o['wall']} ({o['u0']:.2f}-{o['u1']:.2f} m along the wall, "
                  f"{o['z0']:.2f}-{o['z1']:.2f} m high).")
            return
    total = n * width + (n - 1) * gap
    print(f"Group width {total:.2f} m, centred on the zone.")
    print(f"  Group: {n} x {kind}, {width:.2f} m wide, gap {gap:.2f} m, sill {sill:.2f} m, "
          f"height {h:.2f} m, on wall {i}.")
    if get_input("Add this group? [Y/n] ").strip().lower().startswith("n"):
        print("  Group not added.")
        return
    st["windows"].append(new)


def list_windows(st):
    r = st.get("roof") or {}
    print(f"  Roof: {r.get('type', 'flat')}"
          + (f", wall height {st['height_m']} m"
             if r.get("type", "flat") == "flat" and "height_m" in st else ""))
    sky = st.get("skylights", [])
    if not st["windows"] and not sky:
        print("  (no windows/panels/skylights)")
    for lab, w in zip(group_labels(st), st["windows"]):
        print(f"  [{lab}] {w['n']} x {w.get('kind', 'window')}, "
              f"{w['width']:.2f} m wide, gap {w['gap']:.2f} m, "
              f"sill={w['sill']} m, height={w['h']} m")
    rm = make_roof(st)[0]
    for k, sk in enumerate(sky):
        info = ""
        if rm is not None:
            c, why = sky_corners(sk, rm.pts, st["scale_m_per_px"], rm)
            if c is None:
                info = f" - INVALID ({why}), {SKY_INVALID}"
            elif "north_deg" in st:
                i = rm.part_of(c.mean(axis=0))
                w_, h_, _, _ = sky_params(c, rm, i, st["north_deg"])
                info = f", {w_:.2f} m x {h_:.2f} m on {rm.parts[i]['id']}"
        print(f"  [S{k}] skylight{info}")


def delete_window(st):
    list_windows(st)
    sky = st.get("skylights", [])
    if st["windows"] or sky:
        k = get_input("Label of the group/skylight to delete, e.g. W0, P1 or S0 "
                      "(Enter = cancel): ").strip().upper()
        labs = group_labels(st)
        if k in labs:
            st["windows"].pop(labs.index(k))
        elif k[:1] == "S" and k[1:].isdigit() and int(k[1:]) < len(sky):
            sky.pop(int(k[1:]))
        elif k:
            print("  Unknown label.")


# ---------------------------------------------------------------- geometry
def build(st):
    """Sections of the building: walls (their top follows the roof), windows and
    panels, floor, roof part(s) and skylights."""
    north = st["north_deg"]
    pts = outline_m(st)
    n = len(pts)
    rm, why = make_roof(st)
    if rm is None:
        raise ValueError(f"invalid roof: {why}")

    rects = accepted_rects(st, pts)          # windows + panels, conflicts removed

    comps, polys3d = [], []
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        d = b - a
        L = np.linalg.norm(d)
        d_hat = d / L
        nrm = np.array([d[1], -d[0]]) / L              # outward normal (CCW polygon)
        # Azimuth grows clockwise (seen from above), which for the outward normal
        # means moving towards the START of the wall (point a). So the corner on the
        # positive-azimuth side is a, and u0 (distance a -> window edge) is the
        # requested dist_corner_m.
        az = edge_azimuth(a, b, north)                 # clockwise from South
        prof = rm.profile(a, b)                        # top of the wall: (u, z)
        polys3d.append(([(*a, 0), (*b, 0)]
                        + [(*(a + u * d_hat), z) for u, z in prof[::-1]], "lightgrey"))
        w_area = 0.0
        count = defaultdict(int)                       # own numbering per kind
        for r in [r for r in rects if r["wall"] == i]:
            kd = r["kind"]
            j = count[kd]
            count[kd] += 1
            wa, wb = a + r["u0"] * d_hat, a + r["u1"] * d_hat
            off = 0.02 * nrm
            z0, z1 = r["z0"], r["z1"]
            polys3d.append(([(*(wa + off), z0), (*(wb + off), z0),
                             (*(wb + off), z1), (*(wa + off), z1)],
                            "deepskyblue" if kd == "window" else "orange"))
            wid, hh = r["u1"] - r["u0"], z1 - z0
            comps.append(dict(id=f"{kd}_{i}_{j}", type=kd, parent=f"wall_{i}",
                              azimuth_deg=round(az, 2), slope_deg=90,
                              width_m=round(wid, 3), height_m=round(hh, 3),
                              sill_height_m=round(z0, 3),
                              dist_corner_m=round(r["u0"], 3),
                              area_m2=round(wid * hh, 3), net_area_m2=None))
            w_area += wid * hh
        wall_area = rm.wall_area(a, b)                 # height_m = mean height
        comps.append(dict(id=f"wall_{i}", type="wall", parent=None,
                          azimuth_deg=round(az, 2), slope_deg=90,
                          width_m=round(L, 3), height_m=round(wall_area / L, 3),
                          sill_height_m=0, dist_corner_m=None,
                          area_m2=round(wall_area, 3),
                          net_area_m2=round(wall_area - w_area, 3)))

    floor_area = shoelace(pts)
    polys3d.append(([(*p_, 0) for p_ in pts], "tan"))
    comps.append(dict(id="floor", type="floor", parent=None, azimuth_deg=None,
                      slope_deg=0, width_m=None, height_m=None, sill_height_m=None,
                      dist_corner_m=None, area_m2=round(floor_area, 3),
                      net_area_m2=None))

    by_part = defaultdict(list)                        # skylights, per roof part
    for j, (_, c) in enumerate(accepted_skylights(st, rm)):
        by_part[rm.part_of(c.mean(axis=0))].append((j, c))
    for i, part in enumerate(rm.parts):
        poly, f = part["poly"], rm.factor(i)
        polys3d.append(([(*p_, rm.z(p_, i)) for p_ in poly], "salmon"))
        g = rm.facing(i)
        az = None if g is None else round(vec_azimuth(g, north), 2)
        slope = 180 if g is None else round(180 - rm.pitch_deg(i), 2)
        area, sky_area = shoelace(poly) * f, 0.0
        for j, c in by_part[i]:
            wid, hh, sill, dc = sky_params(c, rm, i, north)
            polys3d.append(([(*p_, rm.z(p_, i) + 0.02) for p_ in c], "deepskyblue"))
            comps.append(dict(id=f"skylight_{j}", type="skylight", parent=part["id"],
                              azimuth_deg=az, slope_deg=slope,
                              width_m=round(wid, 3), height_m=round(hh, 3),
                              sill_height_m=round(sill, 3), dist_corner_m=round(dc, 3),
                              area_m2=round(wid * hh, 3), net_area_m2=None))
            sky_area += wid * hh
        comps.append(dict(id=part["id"], type="roof", parent=None, azimuth_deg=az,
                          slope_deg=slope, width_m=None, height_m=None,
                          sill_height_m=None, dist_corner_m=None,
                          area_m2=round(area, 3),
                          net_area_m2=round(area - sky_area, 3)))
    return comps, polys3d, pts, rm.zmax(), floor_area, rm.volume()


def export(comps, floor_area, volume, stem):
    fname = stem + "_components.csv"
    with open(fname, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(comps[0].keys()))
        wr.writeheader()
        wr.writerows(comps)
    print(f"Saved {fname}")
    print(f"Floor area: {floor_area:.2f} m2   Volume: {volume:.2f} m3")
    summ = defaultdict(lambda: dict(area_m2=0.0, net_area_m2=0.0))
    for c in comps:
        key = f'{c["type"]}|az={c["azimuth_deg"]}|slope={c["slope_deg"]}'
        summ[key]["area_m2"] += c["area_m2"]
        summ[key]["net_area_m2"] += c["net_area_m2"] or c["area_m2"]
    for k, v in summ.items():
        print(f"  {k:40s} area={v['area_m2']:.2f}  net={v['net_area_m2']:.2f}")


def view(polys3d, pts, H):
    """3D view, non-blocking: the menu stays usable while it is displayed."""
    H = max(H, 0.1)                 # highest point of the roof
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    for verts, col in polys3d:
        ax.add_collection3d(Poly3DCollection([verts], facecolor=col,
                                             edgecolor="k", alpha=0.85))
    ax.set_xlim(pts[:, 0].min(), pts[:, 0].max())
    ax.set_ylim(pts[:, 1].min(), pts[:, 1].max())
    ax.set_zlim(0, H)
    ax.set_box_aspect((np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), H))
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_zlabel("z (m)")
    plt.show(block=False)
    plt.pause(0.2)
    return fig


# ---------------------------------------------------------------- main
MENU = """
--- Menu ---
 1  Redo scale calibration
 2  Change North angle / roof (flat, single or double pitched)
 3  Redo outline (rectangle, 3 clicks; windows/panels re-attach to the nearest wall)
 4  Add a roof skylight (2 clicks: opposite corners)
 5  Add a group of identical windows/panels
 6  List window/panel groups and skylights
 7  Delete a window/panel group or a skylight
 8  Rotate the plan 90 deg clockwise
 9  Build: 3D view + export CSV (groups that do not fit are removed from the project)
 q  Save and quit
"""


def main(path):
    stem = os.path.splitext(path)[0]
    pfile = stem + ".project.json"
    plan = Plan(load_image(path))
    save = lambda st: json.dump(st, open(pfile, "w"), indent=2)

    if os.path.exists(pfile):
        st = json.load(open(pfile))
        st.setdefault("skylights", [])         # projects saved before skylights
        print(f"Project loaded: {pfile}")
    else:
        st = dict(image=os.path.basename(path), rotation=0, windows=[],
                  skylights=[])
        plan.show(st, "Rotate the plan if needed (answer in the terminal)")
        while get_input("Rotate the plan 90 deg clockwise? [y/N] ").lower().startswith("y"):
            rotate(st)
            plan.show(st, "Rotate the plan if needed (answer in the terminal)")
        set_scale(plan, st)
        set_north(plan, st)
        set_outline(plan, st)
        set_roof(plan, st)
        plan.show(st)
        while get_input("Add a group of windows/panels? [y/N] ").lower().startswith("y"):
            add_window(plan, st)
            plan.show(st)
        save(st)
        print(f"Project saved: {pfile}")

    actions = {"1": lambda: set_scale(plan, st), "2": lambda: set_params(plan, st),
               "3": lambda: set_outline(plan, st), "4": lambda: add_skylight(plan, st),
               "5": lambda: add_window(plan, st), "6": lambda: list_windows(st),
               "7": lambda: delete_window(st), "8": lambda: rotate(st)}
    view3d = None
    while True:
        if view3d is not None and not plt.fignum_exists(view3d.number):
            view3d = None
        if view3d is None:
            plan.show(st)                   # plan visible, except with the 3D view
        print(MENU)
        c = get_input("Choice: ").strip().lower()
        if c == "q":
            break
        if c == "9":
            if "outline_px" not in st or "scale_m_per_px" not in st \
                    or "north_deg" not in st:
                print("Scale, North angle and outline are required first.")
                continue
            rm, why = make_roof(st)
            if rm is None:
                print(f"The roof is not valid: {why}.")
                continue
            plan.close()                    # the plan is hidden with the 3D view
            if view3d is not None:
                plt.close(view3d)
            prune_groups(st, outline_m(st))  # definitively drop unusable groups
            prune_skylights(st, rm)         # ... and unusable skylights
            save(st)                        # ... and save right away
            comps, polys3d, pts, zmax, floor_area, volume = build(st)
            export(comps, floor_area, volume, stem)
            view3d = view(polys3d, pts, zmax)
        elif c in actions:
            if view3d is not None:          # back to editing: close the 3D view
                plt.close(view3d)
                view3d = None
            plan.show(st)
            actions[c]()
        save(st)                            # autosave after every step
    plt.close("all")
    save(st)
    print(f"Project saved: {pfile}")


if __name__ == "__main__":
    main(sys.argv[1])
