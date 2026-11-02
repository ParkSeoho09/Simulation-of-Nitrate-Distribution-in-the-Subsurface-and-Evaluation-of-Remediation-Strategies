"""시각화(plotly): 구조물 3D 미리보기, 설치 과정 애니메이션, 결과 3D(레이어), 다중 비교, 그래프.

좌표: x, y = 흙수조 평면(m), z = −깊이(m) (지표 0, 아래로 음수) → 화면에서 위가 지표.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from params import ROLE_DEFAULTS, ROLES
from voxel import to_tank

CAMERAS = {
    "등각": dict(eye=dict(x=1.6, y=1.6, z=1.1), up=dict(x=0, y=0, z=1)),
    "정면": dict(eye=dict(x=0.0, y=-2.3, z=0.15), up=dict(x=0, y=0, z=1)),
    "평면": dict(eye=dict(x=0.0, y=0.0, z=2.6), up=dict(x=0, y=1, z=0)),
    "측면": dict(eye=dict(x=2.3, y=0.0, z=0.15), up=dict(x=0, y=0, z=1)),
}
SCEN_COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b"]
CTRL_COLOR = "#555555"
FONT = dict(family="Malgun Gothic, Apple SD Gothic Neo, Noto Sans KR, sans-serif")
REAC_CS = [[0, "#f7c948"], [0.5, "#e09f1f"], [1, "#9a5b00"]]      # 반응재(황·석회석) = 호박색
SLURRY_CS = [[0, "#9b7fe0"], [1, "#3b1d7a"]]                       # 부유 슬러리 = 보라


def tlabel(t_days):
    m = t_days * 1440
    return f"{m:.0f} 분" if m < 1440 else f"{t_days:.1f} 일"


# ───────────────────────────── 공통 ─────────────────────────────────────
def _scene(g, title=None, ztop=0.02):
    zspan = g["H"] + ztop
    m = max(g["Lx"], g["Ly"], zspan)
    return dict(
        xaxis=dict(title="x (m)", range=[0, g["Lx"]], backgroundcolor="#f7f5f0"),
        yaxis=dict(title="y (m)", range=[0, g["Ly"]], backgroundcolor="#f7f5f0"),
        zaxis=dict(title="깊이 (m, 지표=0)", range=[-g["H"], ztop], backgroundcolor="#eef3f7"),
        aspectmode="manual",
        aspectratio=dict(x=g["Lx"] / m * 1.4, y=g["Ly"] / m * 1.4, z=zspan / m * 1.4),
        camera=CAMERAS["등각"],
    )


def _gdict(g):
    if isinstance(g, dict):
        return g
    return dict(nx=g.nx, ny=g.ny, nz=g.nz, Lx=g.Lx, Ly=g.Ly, H=g.H, H_soil=g.H_soil,
                k_interface=g.k_interface, dx=g.dx, dy=g.dy, dz=g.dz)


def tank_traces(g, wt_depth=None):
    """흙수조 외곽선 + 토양/지하수층 경계면 + 지하수면."""
    Lx, Ly, H = g["Lx"], g["Ly"], g["H"]
    xs, ys, zs = [], [], []
    for (x0, y0, x1, y1) in ((0, 0, Lx, 0), (Lx, 0, Lx, Ly), (Lx, Ly, 0, Ly), (0, Ly, 0, 0)):
        for z in (0, -H):
            xs += [x0, x1, None]; ys += [y0, y1, None]; zs += [z, z, None]
    for (x, y) in ((0, 0), (Lx, 0), (Lx, Ly), (0, Ly)):
        xs += [x, x, None]; ys += [y, y, None]; zs += [0, -H, None]
    tr = [go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color="#8a7f70", width=3),
                       name="흙수조", hoverinfo="skip", showlegend=False)]
    hi = g["k_interface"] * g["dz"]
    if hi < H - 1e-9:
        tr.append(go.Surface(x=[[0, Lx], [0, Lx]], y=[[0, 0], [Ly, Ly]], z=[[-hi, -hi], [-hi, -hi]],
                             showscale=False, opacity=0.12, colorscale=[[0, "#2b7bb9"], [1, "#2b7bb9"]],
                             name="지하수층 상단", hoverinfo="name", showlegend=False))
    if wt_depth is not None and abs(wt_depth - hi) > 1e-6:
        tr.append(go.Surface(x=[[0, Lx], [0, Lx]], y=[[0, 0], [Ly, Ly]], z=[[-wt_depth] * 2] * 2,
                             showscale=False, opacity=0.18, colorscale=[[0, "#1f5fa0"], [1, "#1f5fa0"]],
                             name="지하수면", hoverinfo="name", showlegend=False))
    return tr


def mesh_traces(objects, roles, placement, offset_z=0.0, highlight=None, opacity=0.55, show_legend=True):
    out = []
    for o in objects:
        r = roles.get(o.name, {})
        role = r.get("role", "불투수")
        V = to_tank(o.vertices_m, placement)
        col = ROLE_DEFAULTS.get(role, {}).get("color", "#888888") if r.get("color_by_role", True) else o.color
        if highlight and o.name in highlight:
            col = "#ff2d55"
        T = o.triangles
        out.append(go.Mesh3d(x=V[:, 0], y=V[:, 1], z=-(V[:, 2]) + offset_z, i=T[:, 0], j=T[:, 1], k=T[:, 2],
                             color=col, opacity=opacity, name=f"{o.name} · {role}", showlegend=show_legend,
                             hovertext=f"{o.name}<br>역할: {role}", hoverinfo="text", flatshading=True))
    return out


def camera_buttons(n_scenes=1, y=1.12, x=0.0):
    btns = []
    for name, cam in CAMERAS.items():
        args = {}
        for i in range(n_scenes):
            key = "scene" if i == 0 else f"scene{i + 1}"
            args[f"{key}.camera"] = cam
        btns.append(dict(label=name, method="relayout", args=[args]))
    return dict(type="buttons", direction="right", buttons=btns, x=x, y=y, xanchor="left",
                yanchor="top", pad=dict(r=4, t=4), showactive=False, bgcolor="#ffffff")


def _cell_xyz(g, stride=1):
    xs = (np.arange(0, g["nx"], stride) + 0.5) * g["dx"]
    ys = (np.arange(0, g["ny"], stride) + 0.5) * g["dy"]
    zs = -(np.arange(0, g["nz"], stride) + 0.5) * g["dz"]
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    return X.ravel(), Y.ravel(), Z.ravel()


def stride_for(g, target):
    n = g["nx"] * g["ny"] * g["nz"]
    s = 1
    while n / s ** 3 > target:
        s += 1
    return s


def _iso(g, arr, s, lo, hi, cscale, name, opacity=0.35, count=3, cut_x=None, fill=None, showscale=False,
         cbar_title="", cbar_x=1.02, visible=True):
    a = np.asarray(arr, float)[::s, ::s, ::s].copy()
    fillv = (lo - abs(hi - lo) - 1.0) if fill is None else fill
    a = np.where(np.isfinite(a), a, fillv)
    if cut_x is not None:
        xs = (np.arange(0, g["nx"], s) + 0.5) * g["dx"]
        a[xs > cut_x] = fillv
    X, Y, Z = _cell_xyz(g, s)
    return go.Isosurface(x=X, y=Y, z=Z, value=a.ravel(), isomin=lo, isomax=hi, surface_count=count,
                         opacity=opacity, colorscale=cscale, caps=dict(x_show=False, y_show=False, z_show=False),
                         showscale=showscale, colorbar=dict(title=cbar_title, x=cbar_x, len=0.5, thickness=12),
                         name=name, hoverinfo="name", visible=visible)


def _slice(g, arr, axis, frac, cmin, cmax, cscale, name, showscale=True, cbar_title="", cbar_x=1.02,
           opacity=0.95, visible=True):
    a = np.asarray(arr, float)
    if axis == "y":
        j = int(np.clip(frac * g["ny"], 0, g["ny"] - 1))
        C = a[:, j, :]
        xs = (np.arange(g["nx"]) + 0.5) * g["dx"]; zs = -(np.arange(g["nz"]) + 0.5) * g["dz"]
        X, Z = np.meshgrid(xs, zs, indexing="ij")
        Y = np.full_like(X, (j + 0.5) * g["dy"])
    elif axis == "x":
        i = int(np.clip(frac * g["nx"], 0, g["nx"] - 1))
        C = a[i, :, :]
        ys = (np.arange(g["ny"]) + 0.5) * g["dy"]; zs = -(np.arange(g["nz"]) + 0.5) * g["dz"]
        Y, Z = np.meshgrid(ys, zs, indexing="ij")
        X = np.full_like(Y, (i + 0.5) * g["dx"])
    else:   # z: frac = 깊이 비율
        k = int(np.clip(frac * g["nz"], 0, g["nz"] - 1))
        C = a[:, :, k]
        xs = (np.arange(g["nx"]) + 0.5) * g["dx"]; ys = (np.arange(g["ny"]) + 0.5) * g["dy"]
        X, Y = np.meshgrid(xs, ys, indexing="ij")
        Z = np.full_like(X, -(k + 0.5) * g["dz"])
    return go.Surface(x=X, y=Y, z=Z, surfacecolor=C, cmin=cmin, cmax=cmax, colorscale=cscale,
                      showscale=showscale, colorbar=dict(title=cbar_title, x=cbar_x, len=0.45, thickness=12),
                      opacity=opacity, name=name, hoverinfo="name", visible=visible)


# ───────────────────────────── 1) 구조물 미리보기 ─────────────────────────
def structure_preview(objects, roles, placement, g, wt_depth=None, annotations=None, height=560):
    g = _gdict(g)
    fig = go.Figure()
    for t in tank_traces(g, wt_depth):
        fig.add_trace(t)
    for t in mesh_traces(objects, roles, placement):
        fig.add_trace(t)
    if annotations:
        _add_param_annotations(fig, annotations, "scene")
    fig.update_layout(scene=_scene(g), height=height, margin=dict(l=0, r=0, t=40, b=0), font=FONT,
                      legend=dict(orientation="h", y=-0.02, font=dict(size=10)),
                      updatemenus=[camera_buttons(1)])
    return fig


# ───────────────────────────── 파라미터 표시 ─────────────────────────────
def param_annotations(c, sc, res, g):
    """화면 위치별 파라미터 표시 목록: [(x,y,z,짧은 글, 툴팁)]"""
    g = _gdict(g)
    out = []
    Lx, Ly = g["Lx"], g["Ly"]
    out.append((Lx * 0.15, Ly * 0.15, 0.0, f"유입 NO₃ {c.NO3_in_mgN:g} mg N/L · q {c.q_mm_day:g} mm/d",
                f"유입면(지표)<br>질산성질소 {c.NO3_in_mgN:g} mg/L<br>침투 {c.q_mm_day:g} mm/일 "
                f"(≈ {c.q_mm_day * Lx * Ly:.2f} L/일)"))
    out.append((Lx * 0.9, Ly * 0.9, -g["H_soil"] * 0.35, f"토양 {c.texture.split('(')[0]}",
                f"토양 영역<br>{c.texture}<br>초기 포화도 {c.init_saturation:g}"))
    hi = g["k_interface"] * g["dz"]
    if hi < g["H"] - 1e-9:
        out.append((Lx * 0.9, Ly * 0.1, -hi - 0.02, f"지하수층 · 유속 {c.gw_velocity:g} m/d",
                    f"포화대(지하수층)<br>상단 깊이 {hi:.2f} m<br>지하수 유속 {c.gw_velocity:g} m/일"))
    if res is not None and not res.get("is_control"):
        reac = res["reac0"]
        if reac.any():
            idx = np.argwhere(reac).mean(0)
            out.append(((idx[0] + 0.5) * g["dx"], (idx[1] + 0.5) * g["dy"], -(idx[2] + 0.5) * g["dz"],
                        f"황:CaCO₃ {c.ratio_S_CaCO3:g}:1 · {res['mass0_g']:.0f} g",
                        f"반응재 영역<br>황:CaCO₃ = {c.ratio_S_CaCO3:g}:1 (부피)<br>총 질량 {res['mass0_g']:.0f} g"))
        rc = res["role_cell"]
        op = np.argwhere(rc == ROLES.index("주입 개구부"))
        if len(op):
            m = op.mean(0)
            inj = [e for e in res.get("events_log", []) if e.get("event") == "시약 주입"]
            txt = "주입 개구부"
            tip = "주입 개구부"
            if inj:
                e = inj[0]
                txt = f"주입 {e.get('delivered_L', 0):.2f} L ({e.get('mode', '')})"
                tip = f"주입 개구부<br>주입량 {e.get('delivered_L', 0):.2f} L / 계획 {e.get('planned_L', 0):.2f} L" \
                      f"<br>방식 {e.get('mode', '')} · {e.get('duration_min', 0):.1f} 분"
            out.append(((m[0] + 0.5) * g["dx"], (m[1] + 0.5) * g["dy"], -(m[2] + 0.5) * g["dz"], txt, tip))
    return out


def _add_param_annotations(fig, ann, scene="scene"):
    xs, ys, zs, tips = [], [], [], []
    lst = []
    offs = [(30, -30), (-60, 30), (40, 40), (-50, -40), (60, 0), (-70, 0)]
    for i, (x, y, z, txt, tip) in enumerate(ann):
        ax_, ay_ = offs[i % len(offs)]
        lst.append(dict(x=x, y=y, z=z, text=txt, showarrow=True, arrowsize=0.6, arrowwidth=1, ax=ax_, ay=ay_,
                        font=dict(size=9, color="#333"), bgcolor="rgba(255,255,255,0.75)", borderpad=2))
        xs.append(x); ys.append(y); zs.append(z); tips.append(tip)
    fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="markers", marker=dict(size=3, color="#333", opacity=0.5),
                               hovertext=tips, hoverinfo="text", name="파라미터", showlegend=False,
                               scene=scene))
    key = "scene" if scene == "scene" else scene
    fig.layout[key].annotations = lst


# ───────────────────────────── 2) 설치 과정 애니메이션 ───────────────────
def install_animation(res, objects, roles, placement, wt_depth=None, light=False, annotations=None, height=620):
    g = res["grid"]
    frames = res["frames"]
    if light and len(frames) > 12:
        idx = np.unique(np.linspace(0, len(frames) - 1, 12).astype(int))
        frames = [frames[i] for i in idx]
    s = stride_for(g, 6000 if light else 20000)
    base = tank_traces(g, wt_depth)
    zmax = 0.0
    for o in objects:
        V = to_tank(o.vertices_m, placement)
        zmax = max(zmax, float(V[:, 2].max()))
    smax = max(float(np.nanmax(f["solid"])) for f in frames) if frames else 1.0
    pmax = max(float(np.nanmax(f["P"])) for f in frames) if frames else 1.0
    smax = max(smax, 1e-6); pmax = max(pmax, 1e-9)

    def dyn(f):
        prog = f.get("insert_progress", 1.0)
        inserted = set(f.get("inserted", []))
        inserting = set(f.get("inserting", []))
        tr = []
        for o in objects:
            if o.name in inserted:
                off = 0.0
            elif o.name in inserting:
                off = (1 - prog) * (zmax + 0.05)
            else:
                off = zmax + 0.05
            opened = [n for n, v in f.get("open", {}).items() if v] if f.get("kind") in ("open", "inject") else []
            tr += mesh_traces([o], roles, placement, offset_z=off, highlight=opened if f.get("kind") == "inject" else None,
                              opacity=0.5, show_legend=False)
        tr.append(_iso(g, f["solid"], s, 0.02 * smax, smax, REAC_CS, "반응재(고체)", opacity=0.6, count=3))
        tr.append(_iso(g, f["P"], s, 0.05 * pmax, pmax, SLURRY_CS, "부유 슬러리", opacity=0.25, count=2))
        return tr

    fig = go.Figure(data=base + dyn(frames[0]) if frames else base)
    nb = len(base)
    fr_objs = []
    steps = []
    for k, f in enumerate(frames):
        name = f"f{k}"
        fr_objs.append(go.Frame(data=dyn(f), traces=list(range(nb, nb + len(objects) + 2)), name=name,
                                layout=dict(title=dict(text=f"<b>{f['label']}</b> · t = {tlabel(f['t'])}"))))
        steps.append(dict(method="animate", label=f"{k + 1}",
                          args=[[name], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                             transition=dict(duration=0))]))
    fig.frames = fr_objs
    if annotations:
        _add_param_annotations(fig, annotations)
    fig.update_layout(
        scene=_scene(g, ztop=zmax + 0.08), height=height, margin=dict(l=0, r=0, t=60, b=0), font=FONT, showlegend=False,
        title=dict(text=f"<b>{frames[0]['label']}</b>" if frames else "", x=0.02, y=0.97),
        updatemenus=[_play_buttons(), camera_buttons(1, y=1.0, x=0.55)],
        sliders=[dict(steps=steps, x=0.05, len=0.9, y=0.0, currentvalue=dict(prefix="단계 "))])
    return fig


def _play_buttons():
    def a(d):
        return [None, dict(frame=dict(duration=d, redraw=True), fromcurrent=True, transition=dict(duration=0),
                           mode="immediate")]
    return dict(type="buttons", direction="right", x=0.0, y=1.0, xanchor="left", yanchor="top",
                buttons=[dict(label="▶ 재생", method="animate", args=a(500)),
                         dict(label="⏸", method="animate",
                              args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
                         dict(label="느리게", method="animate", args=a(1200)),
                         dict(label="빠르게", method="animate", args=a(150))])


# ───────────────────────────── 3) 결과 3D ────────────────────────────────
LAYER_DEF = {
    "반응재 분포": dict(key="solid", cs=REAC_CS, lo_frac=0.02, kind="iso"),
    "NO₃ 오염 플룸": dict(key="NO3", cs="Reds", kind="iso"),
    "pH 단면": dict(key="pH", cs="RdBu", kind="slice"),
    "SO₄ 분포": dict(key="SO4", cs="Greens", kind="iso"),
    "지하수층 NO₃": dict(key="NO3", cs="OrRd", kind="gwslice"),
    "투수성 K/K₀ 단면": dict(key="Kratio", cs="Viridis", kind="slice"),
}


def _field_traces(res, sn, layers, s, c, cut_x, use_slices, scene="scene", cbar_x=1.02, showscale=True):
    g = res["grid"]
    tr = []
    nin = c.NO3_in_mgN
    for name in LAYER_DEF:
        on = name in layers
        d = LAYER_DEF[name]
        arr = np.where(res["active"], sn[d["key"]], np.nan)
        if d["kind"] == "iso" and not use_slices:
            if d["key"] == "solid":
                mx = max(float(np.nanmax(res["snaps"][0]["solid"])), 1e-6)
                t = _iso(g, arr, s, 0.02 * mx, mx, d["cs"], name, opacity=0.65, count=3, cut_x=cut_x, visible=on)
            elif d["key"] == "NO3":
                t = _iso(g, arr, s, 0.2 * nin, nin, d["cs"], name, opacity=0.25, count=4, cut_x=cut_x,
                         showscale=showscale and on, cbar_title="NO₃-N<br>mg/L", cbar_x=cbar_x, visible=on)
            else:
                base = c.SO4_in_mg
                t = _iso(g, arr, s, base + 10, base + 7.54 * nin, d["cs"], name, opacity=0.25, count=3,
                         cut_x=cut_x, visible=on)
        elif d["kind"] == "iso":
            if d["key"] == "solid":
                t = _slice(g, arr, "y", 0.5, 0, max(float(np.nanmax(arr)), 1e-6), d["cs"], name, False, visible=on)
            elif d["key"] == "NO3":
                t = _slice(g, arr, "x", (cut_x or g["Lx"] * 0.5) / g["Lx"], 0, nin, d["cs"], name, showscale and on,
                           "NO₃-N", cbar_x, visible=on)
            else:
                t = _slice(g, arr, "x", 0.5, c.SO4_in_mg, c.SO4_in_mg + 7.54 * nin, d["cs"], name, False, visible=on)
        elif d["kind"] == "slice":
            if d["key"] == "pH":
                t = _slice(g, arr, "y", 0.5, c.soil_pH0 - 1.5, c.soil_pH0 + 1.5, d["cs"], name, showscale and on,
                           "pH", cbar_x + 0.08, visible=on)
            else:
                t = _slice(g, arr, "y", 0.5, 0, 1.2, d["cs"], name, showscale and on, "K/K₀", cbar_x + 0.16, visible=on)
        else:
            ki = g["k_interface"]
            if ki >= g["nz"]:
                t = go.Scatter3d(x=[None], y=[None], z=[None], showlegend=False, visible=on)
            else:
                frac = (ki + (g["nz"] - ki) / 2) / g["nz"]
                t = _slice(g, arr, "z", frac, 0, nin, d["cs"], name, False, visible=on)
        if scene != "scene":
            t.update(scene=scene)
        tr.append(t)
    return tr


def results_scene(res, c, layers, objects=None, roles=None, placement=None, cut_x=None, light=False,
                  annotations=None, height=640, wt_depth=None):
    g = res["grid"]
    snaps = res["snaps"]
    if light and len(snaps) > 10:
        idx = np.unique(np.linspace(0, len(snaps) - 1, 10).astype(int))
        snaps = [snaps[i] for i in idx]
    ncell = g["nx"] * g["ny"] * g["nz"]
    use_slices = ncell > (60000 if light else 140000)
    s = stride_for(g, 5000 if light else 16000)
    base = tank_traces(g, wt_depth)
    if objects:
        base += mesh_traces(objects, roles, placement, opacity=0.18 if cut_x is None else 0.08, show_legend=False)
    nb = len(base)
    first = _field_traces(res, snaps[0], layers, s, c, cut_x, use_slices)
    fig = go.Figure(data=base + first)
    frames, steps = [], []
    for k, sn in enumerate(snaps):
        fr = _field_traces(res, sn, layers, s, c, cut_x, use_slices)
        frames.append(go.Frame(data=fr, traces=list(range(nb, nb + len(fr))), name=f"s{k}",
                               layout=dict(title=dict(text=f"<b>{res['name']}</b> · 운영 {sn['op_day']:.0f}일"))))
        steps.append(dict(method="animate", label=f"{sn['op_day']:.0f}",
                          args=[[f"s{k}"], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                                transition=dict(duration=0))]))
    fig.frames = frames
    if annotations:
        _add_param_annotations(fig, annotations)
    fig.update_layout(scene=_scene(g), height=height, margin=dict(l=0, r=0, t=60, b=0), font=FONT,
                      title=dict(text=f"<b>{res['name']}</b> · 운영 0일", x=0.02, y=0.97), showlegend=False,
                      updatemenus=[_play_buttons(), camera_buttons(1, y=1.0, x=0.55)],
                      sliders=[dict(steps=steps, x=0.05, len=0.9, y=0.0, currentvalue=dict(prefix="운영 일: "))])
    if use_slices:
        fig.add_annotation(text="큰 격자: 등값면 대신 단면으로 표시", xref="paper", yref="paper", x=0.99, y=0.02,
                           showarrow=False, font=dict(size=9, color="#666"), xanchor="right")
    return fig


# ───────────────────────────── 4) 다중 비교(시간 연동) ───────────────────
def compare_view(results: List[dict], c, layers, metric="Cout_mgN", light=True, height=820, cut_x=None,
                 scen_objects: Optional[Dict[str, tuple]] = None):
    """위: 비교군·대조군 3D를 나란히, 아래: 지표 그래프. 하나의 시간 슬라이더로 연동."""
    n = len(results)
    specs = [[{"type": "scene"}] * n, [{"type": "xy", "colspan": n}] + [None] * (n - 1)]
    fig = make_subplots(rows=2, cols=n, specs=specs, row_heights=[0.66, 0.34], vertical_spacing=0.06,
                        subplot_titles=[r["name"] for r in results] + [""])
    g0 = results[0]["grid"]
    s = stride_for(g0, 2500 if light else 7000)
    snaps_n = min(len(r["snaps"]) for r in results)
    sel = np.unique(np.linspace(0, snaps_n - 1, min(snaps_n, 10 if light else 20)).astype(int))
    dyn_idx = []
    for ci, r in enumerate(results):
        scene = "scene" if ci == 0 else f"scene{ci + 1}"
        for t in tank_traces(r["grid"]):
            fig.add_trace(t, row=1, col=ci + 1)
        if scen_objects and r["name"] in scen_objects:
            objs, roles, pl = scen_objects[r["name"]]
            for t in mesh_traces(objs, roles, pl, opacity=0.15, show_legend=False):
                fig.add_trace(t, row=1, col=ci + 1)
        tr = _field_traces(r, r["snaps"][sel[0]], layers, s, c, cut_x, False, scene=scene,
                           showscale=(ci == n - 1))
        start = len(fig.data)
        for t in tr:
            fig.add_trace(t, row=1, col=ci + 1)
        dyn_idx += list(range(start, len(fig.data)))
        fig.layout[scene].update(_scene(r["grid"]))
    # 그래프
    ylab = {"Cout_mgN": "유출 NO₃-N (mg/L)", "eta": "제거효율 η (%)", "SO4_out_mg": "유출 SO₄ (mg/L)",
            "pH_out": "유출 pH", "N_to_gw_g": "지하수 도달 NO₃-N 누적 (g)"}.get(metric, metric)
    ymax = 0
    for ci, r in enumerate(results):
        D = r["daily"]
        y = D[metric] * (100 if metric == "eta" else 1)
        ymax = max(ymax, float(np.nanmax(y)) if np.isfinite(y).any() else 0)
        col = CTRL_COLOR if r["is_control"] else SCEN_COLORS[ci % len(SCEN_COLORS)]
        fig.add_trace(go.Scatter(x=D["day"], y=y, mode="lines", name=r["name"],
                                 line=dict(color=col, width=2, dash="dash" if r["is_control"] else "solid")),
                      row=2, col=1)
    marker_idx = len(fig.data)
    t0 = results[0]["snaps"][sel[0]]["op_day"]
    fig.add_trace(go.Scatter(x=[t0, t0], y=[0, ymax * 1.05 or 1], mode="lines",
                             line=dict(color="#111", width=1.5, dash="dot"), name="현재 시각", showlegend=False),
                  row=2, col=1)
    frames, steps = [], []
    for k in sel:
        data = []
        for ci, r in enumerate(results):
            scene = "scene" if ci == 0 else f"scene{ci + 1}"
            data += _field_traces(r, r["snaps"][k], layers, s, c, cut_x, False, scene=scene,
                                  showscale=(ci == n - 1))
        td = results[0]["snaps"][k]["op_day"]
        data.append(go.Scatter(x=[td, td], y=[0, ymax * 1.05 or 1]))
        frames.append(go.Frame(data=data, traces=dyn_idx + [marker_idx], name=f"c{k}"))
        steps.append(dict(method="animate", label=f"{td:.0f}",
                          args=[[f"c{k}"], dict(mode="immediate", frame=dict(duration=0, redraw=True),
                                                transition=dict(duration=0))]))
    fig.frames = frames
    fig.update_xaxes(title_text="운영 일", row=2, col=1)
    fig.update_yaxes(title_text=ylab, row=2, col=1)
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=70, b=10), font=FONT,
                      legend=dict(orientation="h", y=0.33, x=0.0, font=dict(size=10)),
                      updatemenus=[_play_buttons(), camera_buttons(n, y=1.0, x=0.45)],
                      sliders=[dict(steps=steps, x=0.05, len=0.9, y=-0.02, currentvalue=dict(prefix="운영 일: "))])
    return fig


SYNC_JS = """
<script>
(function(){
  var gd = document.getElementsByClassName('plotly-graph-div')[0];
  if(!gd) return;
  var busy = false;
  gd.on('plotly_relayout', function(ev){
    if(busy) return;
    var cam = null;
    for (var k in ev){ if(k.indexOf('.camera')>0){ cam = ev[k]; break; } }
    if(!cam) return;
    var upd = {};
    for (var k2 in gd._fullLayout){ if(/^scene\\d*$/.test(k2)) upd[k2 + '.camera'] = cam; }
    busy = true;
    Plotly.relayout(gd, upd).then(function(){ busy = false; });
  });
})();
</script>
"""


def to_html(fig, sync_cameras=False, title="SLAD 시뮬레이션"):
    """단일 HTML 파일(plotly.js 포함). sync_cameras=True면 여러 3D 화면의 카메라를 동기화."""
    html = fig.to_html(include_plotlyjs=True, full_html=True, auto_play=False,
                       config={"displaylogo": False, "responsive": True})
    html = html.replace("<head>", f"<head><meta charset='utf-8'><title>{title}</title>", 1)
    if sync_cameras:
        html = html.replace("</body>", SYNC_JS + "</body>")
    return html


# ───────────────────────────── 5) 그래프 ────────────────────────────────
def timeseries(results: List[dict], key, ylabel, scale=1.0, hlines=None, eval_start=None, height=330):
    fig = go.Figure()
    for i, r in enumerate(results):
        D = r["daily"]
        col = CTRL_COLOR if r["is_control"] else SCEN_COLORS[i % len(SCEN_COLORS)]
        fig.add_trace(go.Scatter(x=D["day"], y=D[key] * scale, mode="lines", name=r["name"],
                                 line=dict(color=col, width=2, dash="dash" if r["is_control"] else "solid")))
        for rf in r.get("refills", []):
            if rf.get("ok"):
                fig.add_vline(x=rf["day"], line=dict(color=col, width=1, dash="dot"))
    for (y, txt) in (hlines or []):
        fig.add_hline(y=y, line=dict(color="#999", dash="dash"), annotation_text=txt, annotation_font_size=10)
    if eval_start:
        fig.add_vrect(x0=0, x1=eval_start, fillcolor="#bbbbbb", opacity=0.12, line_width=0,
                      annotation_text="대조군 돌파 전(η 평가 제외)", annotation_font_size=9,
                      annotation_position="top left")
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), font=FONT,
                      xaxis_title="운영 일", yaxis_title=ylabel, legend=dict(orientation="h", y=1.12))
    return fig


def depth_profiles(results: List[dict], metrics_list: List[dict], key, xlabel, snap="last", height=380):
    fig = go.Figure()
    for i, (r, m) in enumerate(zip(results, metrics_list)):
        profs = m.get("profiles") or []
        if not profs:
            continue
        p = profs[-1] if snap == "last" else profs[min(int(snap), len(profs) - 1)]
        g = r["grid"]
        d = (np.arange(g["nz"]) + 0.5) * g["dz"]
        col = CTRL_COLOR if r["is_control"] else SCEN_COLORS[i % len(SCEN_COLORS)]
        fig.add_trace(go.Scatter(x=p[key], y=-d, mode="lines", name=f"{r['name']} (전체 평균)",
                                 line=dict(color=col, dash="dash" if r["is_control"] else "solid")))
        if key + "_reac" in p:
            fig.add_trace(go.Scatter(x=p[key + "_reac"], y=-d, mode="lines", name=f"{r['name']} (반응재 기둥)",
                                     line=dict(color=col, dash="dot")))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), font=FONT, xaxis_title=xlabel,
                      yaxis_title="깊이 (m)", legend=dict(font=dict(size=9)))
    return fig


def baseline_damage(ctrl: dict, c, height=360):
    """대조군(꽂지 않았을 때) 피해 기본 그래프: 유출 농도와 지하수 누적 도달량."""
    D = ctrl["daily"]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=D["day"], y=D["Cout_mgN"], name="토양 바닥 유출 NO₃-N (mg/L)",
                             line=dict(color="#c0392b", width=2)), secondary_y=False)
    fig.add_trace(go.Scatter(x=D["day"], y=D["N_to_gw_g"], name="지하수 도달 누적 (g N)",
                             line=dict(color="#2c3e50", width=2, dash="dot")), secondary_y=True)
    fig.add_hline(y=10, line=dict(color="#888", dash="dash"), annotation_text="먹는물 기준 10 mg/L",
                  annotation_font_size=10)
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), font=FONT,
                      legend=dict(orientation="h", y=1.15), xaxis_title="운영 일")
    fig.update_yaxes(title_text="mg N/L", secondary_y=False)
    fig.update_yaxes(title_text="g N", secondary_y=True)
    return fig
