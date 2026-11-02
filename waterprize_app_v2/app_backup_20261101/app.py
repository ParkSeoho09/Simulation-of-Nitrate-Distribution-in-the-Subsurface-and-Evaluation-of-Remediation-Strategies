"""SLAD 반응재 배치 방식 비교 시뮬레이터 — Streamlit 앱 (한국어 UI)

실행: run_app.bat 더블클릭  (또는  python -m streamlit run app.py)
"""
from __future__ import annotations

import base64
import copy
import io
import json
import os
import sys
import time
import traceback
from dataclasses import asdict

import numpy as np
import pandas as pd
import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import install_events as ie  # noqa: E402
import metrics as M  # noqa: E402
import texts as TX  # noqa: E402
import viz  # noqa: E402
from engine import (Scenario, StopRequested, build_geometry, control_breakthrough_day, eval_start_from,  # noqa: E402
                    planned_masses, run_scenario)
from io_3mf import UNIT_LABEL, UNIT_TO_M, load_mesh_file, rescale_objects  # noqa: E402
from params import (GRID_PRESETS, REGISTRY, ROLE_DEFAULTS, ROLES, SOIL_SOURCE, SOIL_TEXTURES,  # noqa: E402
                    Common, SoilLayer, suggest_role)
from voxel import Placement, make_grid  # noqa: E402

st.set_page_config(page_title="SLAD 배치 방식 비교 시뮬레이터", layout="wide", page_icon="💧")


def _compat(fn):
    """Streamlit 버전 호환: width="stretch"를 모르는 버전이면 use_container_width=True로 바꿔 다시 호출."""
    def w(*a, **k):
        try:
            return fn(*a, **k)
        except TypeError as e:
            if "width" in k and "width" in str(e):
                k.pop("width")
                k["use_container_width"] = True
                return fn(*a, **k)
            raise
    return w


for _n in ("button", "data_editor", "dataframe", "download_button"):
    if hasattr(st, _n):
        setattr(st, _n, _compat(getattr(st, _n)))


# ───────────────────────────── 공통 도우미 ───────────────────────────────
def show_fig(fig, key=None):
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=key)


def show_html(html, height):
    import streamlit.components.v1 as components
    components.html(html, height=height, scrolling=False)


def num(label, value, key, help=None, fmt="%.4g", step=None):
    """최솟값·최댓값 제한 없는 숫자 입력."""
    kw = dict(value=value, key=key, help=help, format=fmt)
    if step is not None:
        kw["step"] = step
    return st.number_input(label, **kw)


UNIT_KEYS = {"mm": "millimeter", "cm": "centimeter", "m": "meter"}
UNIT_FROM = {v: k for k, v in UNIT_KEYS.items()}

SAMPLES = {
    "영구 주입공 (injection_well)": ("samples/injection_well_scale0.05.3mf",
        "주입관 스크린을 통해 황:석회석 슬러리를 토양에 주입한다. 주입된 입자가 스크린 주변 공극에 걸려 반응재 층을 만든다. "
        "재보충은 같은 주입관으로 다시 주입한다(주입관 막힘이 누적될 수 있음).",
        "1) 흙수조 중앙에 주입정을 30분 동안 지표에서 수직으로 삽입한다. 2) 개구부를 1분 열어 둔다. "
        "3) 고형분 20% 슬러리 2.84 L를 펌프로 0.2 L/min 주입한다. 4) 60분 정치한다. "
        "5) 효율이 50% 미만이면 1 L를 펌프로 재보충, 7일 간격, 최대 5회, 작업 20분.",
        [dict(ie.empty_row("구조물 삽입"), **{"소요(분)": 30.0}),
         dict(ie.empty_row("개구부 열기"), **{"소요(분)": 1.0}),
         dict(ie.empty_row("시약 주입"), **{"방식": "펌프", "부피(L)": 2.84, "고형분비(v/v)": 0.2, "유량(L/min)": 0.2}),
         dict(ie.empty_row("정치(대기)"), **{"소요(분)": 60.0}),
         dict(ie.empty_row("재보충"), **{"방식": "펌프", "부피(L)": 1.0, "고형분비(v/v)": 0.2, "유량(L/min)": 0.2,
                                       "트리거 η(%)": 50.0, "간격(일)": 7.0, "최대 회차": 5.0, "소요(분)": 20.0})]),
    "케이슨형 (caisson_gate)": ("samples/caisson_gate_scale0.05.3mf",
        "강철 케이슨 안 내부 챔버에 황:석회석 충전층을 미리 채워 넣는다. 설치 후 반응재를 꺼내거나 다시 채울 수 없다(재보충 불가).",
        "1) 반응재를 채운 케이슨을 흙수조 중앙에 30분 동안 삽입한다. 2) 효율이 50% 미만이면 재보충을 시도한다(구조상 불가).",
        [dict(ie.empty_row("구조물 삽입"), **{"소요(분)": 30.0}),
         dict(ie.empty_row("재보충"), **{"방식": "펌프", "부피(L)": 1.0, "트리거 η(%)": 50.0, "간격(일)": 7.0,
                                       "최대 회차": 5.0, "소요(분)": 20.0})]),
}


def ss():
    return st.session_state


def init_state():
    if "common" not in ss():
        ss().common = Common()
        ss().scen = []
        ss().cur = 0
        ss().results = None
        ss().running = False
        ss().ctrl_cache = {}


init_state()


def role_df_for(objects, old=None):
    rows = []
    oldmap = {} if old is None else {r["객체"]: r for r in old.to_dict("records")}
    for o in objects:
        if o.name in oldmap:
            rows.append(oldmap[o.name])
            continue
        role = suggest_role(o.name)
        d = ROLE_DEFAULTS[role]
        rows.append({"객체": o.name, "역할": role,
                     "K (m/s)": d["K"], "공극률": d["phi"], "반응": bool(d["reactive"]),
                     "우선순위": d["prio"], "교체 가능": False, "얇은 객체": "자동",
                     "벽 두께(mm)": round(o.min_thickness_m * 1000, 3) if np.isfinite(o.min_thickness_m) else None,
                     "체적(cm³)": round(o.volume_m3 * 1e6, 3), "닫힘": o.closed})
    return pd.DataFrame(rows)


def roles_from_df(df):
    out = {}
    for r in df.to_dict("records"):
        role = r["역할"] if r["역할"] in ROLES else "불투수"
        d = ROLE_DEFAULTS[role]
        def val(x, default):
            try:
                if x is None or (isinstance(x, float) and np.isnan(x)):
                    return default
                return float(x)
            except (TypeError, ValueError):
                return default
        out[r["객체"]] = {"role": role, "K": val(r.get("K (m/s)"), d["K"]), "phi": val(r.get("공극률"), d["phi"]),
                         "reactive": bool(r.get("반응", d["reactive"])), "prio": int(val(r.get("우선순위"), d["prio"])),
                         "accessible": bool(r.get("교체 가능", False)), "thin_mode": r.get("얇은 객체") or "자동",
                         "thickness_m": val(r.get("벽 두께(mm)"), 2.0) / 1000.0,
                         "alpha": d["alpha"], "n": d["n"]}
    return out


def new_scenario(name="비교군"):
    return {"name": name, "mechanism": "", "install_text": "", "file_name": "", "file_bytes": None,
            "unit_src": "millimeter", "unit_sel": "millimeter", "objects": [], "objects_raw": [],
            "title": "", "description": "", "role_df": pd.DataFrame(),
            "placement": dict(x=None, y=None, top_depth=0.0, rot_deg=0.0, tilt_deg=0.0, scale=1.0),
            "events_df": pd.DataFrame([ie.empty_row("구조물 삽입")]), "allow_refill_wo_opening": False,
            "mass_correction": True, "geom_report": None}


def load_file_into(sc, data, fname):
    d = load_mesh_file(data, fname)
    sc["file_bytes"] = data
    sc["file_name"] = fname
    sc["unit_src"] = d["unit"]
    sc["unit_sel"] = d["unit"]
    sc["objects_raw"] = d["objects"]
    sc["objects"] = d["objects"]
    sc["title"] = d["title"]
    sc["description"] = d["description"]
    sc["role_df"] = role_df_for(d["objects"])
    sc["geom_report"] = None


def scenario_obj(sc, c: Common) -> Scenario:
    pl = dict(sc["placement"])
    pl["x"] = c.Lx / 2 if pl.get("x") is None else pl["x"]
    pl["y"] = c.Ly / 2 if pl.get("y") is None else pl["y"]
    return Scenario(name=sc["name"], mechanism=sc["mechanism"], install_text=sc["install_text"],
                    file_name=sc["file_name"], unit=sc["unit_sel"], objects=sc["objects"],
                    roles=roles_from_df(sc["role_df"]) if len(sc["role_df"]) else {},
                    placement=Placement(**pl), events=sc["events_df"].to_dict("records"),
                    allow_refill_wo_opening=sc["allow_refill_wo_opening"],
                    mass_correction=sc["mass_correction"])


def validate_common(c: Common):
    errs, warns = [], []
    def E(cond, msg):
        if cond:
            errs.append(msg)
    def W(cond, msg):
        if cond:
            warns.append(msg)
    E(c.Lx <= 0 or c.Ly <= 0 or c.H_soil <= 0, "흙수조 크기는 0보다 커야 합니다 — 이 값은 계산이 불가능합니다.")
    E(c.H_aq < 0, "지하수층 두께가 음수입니다 — 이 값은 계산이 불가능합니다.")
    E(min(c.nx, c.ny, c.nz) < 3, "격자 수는 각 방향 3 이상이어야 합니다 — 이 값은 계산이 불가능합니다.")
    E(c.q_mm_day < 0, "침투 플럭스가 음수입니다 — 이 값은 계산이 불가능합니다.")
    E(min(c.NO3_in_mgN, c.SO4_in_mg, c.Ca_in_mg, c.gw_NO3_mgN, c.soil_NO3_mgN) < 0, "농도에 음수가 있습니다 — 이 값은 계산이 불가능합니다.")
    E(c.days <= 0, "운영 기간은 0보다 커야 합니다 — 이 값은 계산이 불가능합니다.")
    E(c.ratio_S_CaCO3 <= 0, "황:CaCO₃ 비는 0보다 커야 합니다 — 이 값은 계산이 불가능합니다.")
    E(not (0 < c.init_saturation <= 1), "초기 포화도는 0~1 사이여야 합니다 — 이 값은 계산이 불가능합니다.")
    E(c.flow_update_days <= 0, "흐름 재계산 주기는 0보다 커야 합니다 — 이 값은 계산이 불가능합니다.")
    if c.wt_depth is not None:
        E(c.wt_depth < 0 or c.wt_depth > c.H_soil + c.H_aq, "지하수면 깊이가 흙수조 밖입니다 — 이 값은 계산이 불가능합니다.")
    if c.Ks_override is not None:
        E(c.Ks_override <= 0, "포화 투수계수는 0보다 커야 합니다 — 이 값은 계산이 불가능합니다.")
    for L in c.layers:
        E(L.theta_s <= L.theta_r or L.n <= 1 or L.alpha <= 0 or L.Ks <= 0,
          f"층 {L.top}~{L.bottom} m: θs>θr, n>1, α>0, Ks>0 이어야 합니다 — 이 값은 계산이 불가능합니다.")
    W(c.q_mm_day > 100, f"침투 {c.q_mm_day:g} mm/일은 매우 큰 값입니다(일반 관개 1~20 mm/일).")
    W(c.NO3_in_mgN > 300, f"유입 NO₃-N {c.NO3_in_mgN:g} mg/L는 비현실적으로 높습니다.")
    W(not (5 <= c.temperature <= 40), f"온도 {c.temperature:g}°C는 미생물 탈질 범위를 벗어납니다.")
    W(not (0.5 <= c.ratio_S_CaCO3 <= 6), f"황:CaCO₃ = {c.ratio_S_CaCO3:g}:1은 선행연구 범위(1:1~3:1)를 벗어납니다.")
    W(c.nx * c.ny * c.nz > 250000, f"격자 {c.nx * c.ny * c.nz:,}칸: 계산이 매우 오래 걸립니다.")
    W(c.days > 1500, "운영 기간이 깁니다: 계산이 오래 걸립니다.")
    W(c.init_saturation > 0.95, "초기 포화도가 거의 포화입니다.")
    W(max(c.Lx, c.Ly, c.H_soil) > 5, "흙수조가 5 m를 넘습니다(현장 규모).")
    for k, v in c.params.items():
        if k in REGISTRY and v is not None and v < 0 and k not in ("pCO2",):
            errs.append(f"고급 파라미터 {k} = {v}: 이 값은 계산이 불가능합니다(음수).")
    return errs, warns


def preview_fig(sc, c, annotations=None):
    s = scenario_obj(sc, c)
    g = make_grid(c)
    return viz.structure_preview(s.objects, s.roles, s.placement, g, wt_depth=c.wt_depth, annotations=annotations)


# ───────────────────────────── 머리말 ─────────────────────────────────────
st.title("💧 SLAD 반응재 배치 방식 비교 시뮬레이터")
st.caption("농경지 불포화대 질산염 → 지하수 오염 방지 · 황:CaCO₃(3:1) 반응재 · 흙수조 비교실험 · "
           "비교군은 3MF와 설치 방법으로 직접 입력합니다(코드에 고정된 비교군 없음).")

tabs = st.tabs(["① 비교군 입력", "② 공통 조건", "③ 실행", "④ 결과·애니메이션", "⑤ 지표·메커니즘 설명",
                "⑥ 검증", "⑦ 내보내기"])
c: Common = ss().common

# ═════════════════════════════ ① 비교군 입력 ══════════════════════════════
with tabs[0]:
    top = st.columns([3, 2, 2, 2])
    with top[0]:
        names = [s["name"] for s in ss().scen]
        if names:
            ss().cur = min(ss().cur, len(names) - 1)
            ss().cur = st.radio("비교군 선택", range(len(names)), format_func=lambda i: names[i],
                                index=ss().cur, horizontal=True, key="scen_radio")
        else:
            st.info("비교군이 없습니다. 오른쪽에서 추가하거나 예시를 불러오세요. 비교군이 없어도 ④ 결과 탭에서 기본 대조군(꽂지 않았을 때)은 볼 수 있습니다.")
    with top[1]:
        if st.button("➕ 빈 비교군 추가", width="stretch"):
            ss().scen.append(new_scenario(f"비교군 {len(ss().scen) + 1}"))
            ss().cur = len(ss().scen) - 1
            st.rerun()
    with top[2]:
        ex = st.selectbox("예시 불러오기", ["—"] + list(SAMPLES.keys()), key="ex_sel")
        if ex != "—" and st.button("예시 추가", width="stretch"):
            path, mech, text, rows = SAMPLES[ex]
            sc_new = new_scenario(ex.split(" (")[0])
            with open(os.path.join(HERE, path), "rb") as fh:
                load_file_into(sc_new, fh.read(), os.path.basename(path))
            sc_new["mechanism"] = mech
            sc_new["install_text"] = text
            sc_new["events_df"] = pd.DataFrame(rows, columns=ie.COLUMNS)
            ss().scen.append(sc_new)
            ss().cur = len(ss().scen) - 1
            st.rerun()
    with top[3]:
        if ss().scen and st.button("🗑 선택 비교군 삭제", width="stretch"):
            ss().scen.pop(ss().cur)
            ss().cur = max(0, ss().cur - 1)
            st.rerun()

    if ss().scen:
        sc = ss().scen[ss().cur]
        k = f"s{id(sc)}"
        left, right = st.columns([1.05, 1])
        with left:
            sc["name"] = st.text_input("비교군 이름", sc["name"], key=k + "name")
            up = st.file_uploader("구조물 3MF (또는 STL)", type=["3mf", "stl"], key=k + "up")
            if up is not None and up.name != sc["file_name"]:
                try:
                    load_file_into(sc, up.getvalue(), up.name)
                    st.success(f"{up.name}: 객체 {len(sc['objects'])}개, 단위 {UNIT_LABEL.get(sc['unit_src'])}")
                except Exception as e:
                    st.error(f"3MF를 읽지 못했습니다: {e}")
            if sc["file_name"]:
                st.caption(f"파일: **{sc['file_name']}** · 3MF 단위 속성: {UNIT_LABEL.get(sc['unit_src'], sc['unit_src'])}"
                           + (f" · 제목: {sc['title']}" if sc["title"] else ""))
                if sc["description"]:
                    with st.expander("3MF에 적힌 설명"):
                        st.write(sc["description"])
            sc["mechanism"] = st.text_area("작용 메커니즘 설명 (결과·내보내기에 그대로 표시)", sc["mechanism"], height=90,
                                           key=k + "mech")
            u1, u2 = st.columns(2)
            with u1:
                cur_u = UNIT_FROM.get(sc["unit_sel"], "mm")
                newu = st.selectbox("좌표 단위", ["mm", "cm", "m"], index=["mm", "cm", "m"].index(cur_u),
                                    help="기본은 3MF의 unit 속성입니다. 파일 단위가 틀렸을 때만 바꾸세요. 자동 축소는 하지 않습니다.",
                                    key=k + "unit")
                if UNIT_KEYS[newu] != sc["unit_sel"] and sc["objects_raw"]:
                    sc["objects"] = rescale_objects(sc["objects_raw"], sc["unit_src"], UNIT_KEYS[newu])
                    sc["unit_sel"] = UNIT_KEYS[newu]
                    sc["geom_report"] = None
            with u2:
                sc["placement"]["scale"] = num("배율(사용자가 정할 때만)", float(sc["placement"]["scale"]), k + "scale",
                                               help="1이면 원래 크기. 축소가 필요하면 직접 비율을 적으세요.")
            p1, p2, p3 = st.columns(3)
            with p1:
                sc["placement"]["x"] = num("위치 x (m)", sc["placement"]["x"] if sc["placement"]["x"] is not None else c.Lx / 2,
                                           k + "px", help="3MF 원점(보통 구조물 축)이 놓일 흙수조 x 좌표")
                sc["placement"]["rot_deg"] = num("회전 (°)", float(sc["placement"]["rot_deg"]), k + "rot")
            with p2:
                sc["placement"]["y"] = num("위치 y (m)", sc["placement"]["y"] if sc["placement"]["y"] is not None else c.Ly / 2,
                                           k + "py")
                sc["placement"]["tilt_deg"] = num("기울기 (°)", float(sc["placement"]["tilt_deg"]), k + "tilt")
            with p3:
                sc["placement"]["top_depth"] = num("3MF z=0 깊이 (m)", float(sc["placement"]["top_depth"]), k + "pd",
                                                   help="0이면 3MF의 z=0 면이 지표에 놓입니다.")
            if len(sc["role_df"]):
                st.markdown("**객체별 역할표** — 이름 키워드로 제안된 값입니다. 바꿀 수 있습니다.")
                cfg = {
                    "객체": st.column_config.TextColumn(disabled=True),
                    "역할": st.column_config.SelectboxColumn(options=ROLES, required=True),
                    "K (m/s)": st.column_config.NumberColumn(format="%.2e", help="빈칸 = 토양 값"),
                    "공극률": st.column_config.NumberColumn(format="%.3f", help="빈칸 = 토양 값"),
                    "반응": st.column_config.CheckboxColumn(help="반응재(고체)일 때 SLAD 반응 여부"),
                    "우선순위": st.column_config.NumberColumn(format="%d", help="겹칠 때 큰 값이 이깁니다"),
                    "교체 가능": st.column_config.CheckboxColumn(help="카트리지처럼 꺼내 새것으로 바꿀 수 있는 반응재"),
                    "얇은 객체": st.column_config.SelectboxColumn(options=["자동", "면 래스터화", "체적만"]),
                    "벽 두께(mm)": st.column_config.NumberColumn(format="%.3f", help="자동 추정 최소 두께. 실링 장벽 저항 계산에 씀"),
                    "체적(cm³)": st.column_config.NumberColumn(disabled=True, format="%.2f"),
                    "닫힘": st.column_config.CheckboxColumn(disabled=True, help="닫힌 메시인지(아니면 내부 판정 부정확)"),
                }
                sc["role_df"] = st.data_editor(sc["role_df"], column_config=cfg, hide_index=True, key=k + "roles",
                                               width="stretch")
                st.caption("역할 기본값: " + "; ".join(f"{r} K={ROLE_DEFAULTS[r]['K']} φ={ROLE_DEFAULTS[r]['phi']}" for r in ROLES[:6])
                           + " — 모두 '가정' 값입니다.")
                if st.button("복셀 검사(격자에 올려 보기)", key=k + "vox"):
                    g = make_grid(c)
                    try:
                        geom = build_geometry(scenario_obj(sc, c), c, g)
                        pm = planned_masses(scenario_obj(sc, c), c, g, geom)
                        sc["geom_report"] = {"counts": geom["comp"]["counts"], "warnings": geom["warnings"],
                                             "errors": geom["errors"], "masses": pm,
                                             "faces": {ov.name: ov.n_faces for ov in geom["ov"]}}
                    except Exception as e:
                        sc["geom_report"] = {"errors": [str(e)], "warnings": [], "counts": {}, "masses": {}, "faces": {}}
                rep = sc["geom_report"]
                if rep:
                    for e in rep["errors"]:
                        st.error(e)
                    df = pd.DataFrame([{"객체": n, "셀 수": v, "장벽 면 수": rep["faces"].get(n, 0)} for n, v in rep["counts"].items()])
                    st.dataframe(df, hide_index=True, width="stretch")
                    for w in rep["warnings"]:
                        st.warning(w)
                    if rep["masses"]:
                        m = rep["masses"]
                        st.caption(f"반응재 질량: 사전 배치 {m['pre_mesh_g']:.0f} g(메시 체적) / {m['pre_voxel_g']:.0f} g(복셀) · "
                                   f"설치 주입 계획 {m['injection_g']:.0f} g · 합계 {m['total_g']:.0f} g")
            sc["mass_correction"] = st.checkbox("반응재 질량을 메시 체적 기준으로 보정(복셀 오차 제거)", sc["mass_correction"], key=k + "mc")
            sc["allow_refill_wo_opening"] = st.checkbox("개구부 없이도 재보충 허용(가정 — 반응재 객체를 교체하는 것으로 봄)",
                                                        sc["allow_refill_wo_opening"], key=k + "arw")
        with right:
            if sc["objects"]:
                show_fig(preview_fig(sc, c), key=k + "prev")
            else:
                st.info("3MF를 올리면 객체별 색으로 3D 미리보기가 나타납니다.")

        st.markdown("---")
        st.subheader("A1. 설치(꽂는) 방법 — 필수")
        st.caption("어떻게 꽂고, 어떻게 시약을 넣는지 적으세요. 시뮬레이터는 이 표대로 구조물을 넣고 시약을 주입하는 과정을 재현합니다.")
        a, b = st.columns([1, 1.4])
        with a:
            sc["install_text"] = st.text_area("설치 과정 설명(자유 텍스트)", sc["install_text"], height=180, key=k + "itxt",
                                              placeholder="예) 1) 케이싱을 30 cm 깊이까지 20분 동안 삽입한다. 2) 포트를 연다. "
                                                          "3) 고형분 20% 슬러리 2 L를 펌프로 0.2 L/min 주입한다. 4) 60분 정치. "
                                                          "5) 효율 50% 미만이면 카트리지 교체, 최대 8회, 15분.")
            b1, b2 = st.columns(2)
            with b1:
                if st.button("텍스트 → 표 (규칙 기반)", key=k + "parse"):
                    rows = ie.parse_text(sc["install_text"], [o.name for o in sc["objects"]])
                    if rows:
                        sc["events_df"] = pd.DataFrame(rows, columns=ie.COLUMNS)
                        st.success(f"{len(rows)}개 단계를 제안했습니다. 표에서 확인·수정하세요.")
                    else:
                        st.warning("텍스트에서 단계를 찾지 못했습니다. 표에 직접 입력하세요.")
            with b2:
                with st.popover("LLM으로 변환(선택)") if hasattr(st, "popover") else st.expander("LLM으로 변환(선택)"):
                    key_in = st.text_input("Anthropic API 키(저장 안 함)", type="password", key=k + "apikey")
                    model = st.text_input("모델", "claude-sonnet-5-5", key=k + "model")
                    if st.button("변환", key=k + "llm") and key_in:
                        try:
                            rows = ie.parse_text_llm(sc["install_text"], [o.name for o in sc["objects"]], key_in, model)
                            sc["events_df"] = pd.DataFrame(rows, columns=ie.COLUMNS)
                            st.success(f"{len(rows)}개 단계를 받았습니다.")
                        except Exception as e:
                            st.error(f"LLM 변환 실패: {e} — 규칙 기반 변환이나 직접 입력을 쓰세요.")
        with b:
            ecfg = {
                "이벤트": st.column_config.SelectboxColumn(options=ie.EVENT_TYPES, required=True),
                "방식": st.column_config.SelectboxColumn(options=ie.METHODS),
                "대상 객체": st.column_config.TextColumn(help="쉼표로 여러 개, '전체' = 해당 역할 객체 모두"),
            }
            for col in ie.NUMERIC:
                ecfg[col] = st.column_config.NumberColumn(format="%.4g")
            df = sc["events_df"]
            for col in ie.COLUMNS:
                if col not in df.columns:
                    df[col] = None
            sc["events_df"] = st.data_editor(df[ie.COLUMNS], num_rows="dynamic", column_config=ecfg, hide_index=True,
                                             key=k + "events", width="stretch")
            roles_of = {r["객체"]: r["역할"] for r in sc["role_df"].to_dict("records")} if len(sc["role_df"]) else {}
            errs, warns = ie.validate(sc["events_df"].to_dict("records"), roles_of)
            for e in errs:
                st.error(e)
            for w in warns:
                st.warning(w)
            sched = ie.schedule(sc["events_df"].to_dict("records"))
            tr = [r for r in sched if r["이벤트"] == "재보충"]
            if sched:
                st.caption("타임라인: " + " → ".join(
                    f"{r['이벤트']}({r['t_start'] * 1440:.0f}분)" if r["t_start"] is not None else f"{r['이벤트']}(운영 중)"
                    for r in sched))
            # 질량 맞추기 도우미
            with st.expander("반응재 총 질량 맞추기 도우미"):
                ref = st.number_input("목표 반응재 질량 (g)", value=1265.0, key=k + "mtgt")
                fs = st.number_input("슬러리 고형분비 (v/v)", value=0.2, key=k + "mfs")
                r_ = c.ratio_S_CaCO3
                rho = (r_ * 2.07 + 2.71) / (r_ + 1)
                st.write(f"필요 슬러리 부피 ≈ **{ref / (max(fs, 1e-9) * rho * 1000):.3f} L** (고체 혼합 밀도 {rho:.2f} g/cm³)")

# ═════════════════════════════ ② 공통 조건 ════════════════════════════════
with tabs[1]:
    st.caption("아래 조건은 모든 비교군과 대조군에 똑같이 적용됩니다(지표 문서의 통제변수). 숫자 입력에는 최솟값·최댓값 제한이 없습니다.")
    H = TX.PARAM_HELP
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**흙수조·지하수층**")
        c.Lx = num("가로 Lx (m)", c.Lx, "Lx", H["Lx"])
        c.Ly = num("세로 Ly (m)", c.Ly, "Ly", H["Ly"])
        c.H_soil = num("토양층 깊이 (m)", c.H_soil, "Hs", H["H_soil"])
        c.H_aq = num("지하수층 두께 (m)", c.H_aq, "Ha", H["H_aq"])
        c.wt_depth = st.number_input("지하수면 깊이 (m, 비우면 토양층 바닥)", value=c.wt_depth, key="wt", help=H["wt_depth"],
                                     format="%.4g")
        c.gw_velocity = num("지하수 유속 (m/일)", c.gw_velocity, "gwv", H["gw_velocity"])
        c.bottom_bc = st.selectbox("바닥 조건", ["지하수 수위", "자유배수"], index=["지하수 수위", "자유배수"].index(c.bottom_bc),
                                   help=H["bottom_bc"])
        c.aq_texture = st.selectbox("지하수층 토성", list(SOIL_TEXTURES), index=list(SOIL_TEXTURES).index(c.aq_texture))
    with c2:
        st.markdown("**토양(불포화대)**")
        c.soil_mode = st.radio("입력 방식", ["간단", "상세"], index=["간단", "상세"].index(c.soil_mode), horizontal=True)
        c.texture = st.selectbox("토성", list(SOIL_TEXTURES), index=list(SOIL_TEXTURES).index(c.texture), help=H["texture"])
        tx = SOIL_TEXTURES[c.texture]
        c.Ks_override = st.number_input("포화 투수계수 Ks (m/s, 비우면 토성 값)", value=c.Ks_override, key="Ks",
                                        help=H["Ks"], format="%.3e")
        st.caption(f"토성 기본값: θr {tx['theta_r']}, θs {tx['theta_s']}, α {tx['alpha']:.1f} 1/m, n {tx['n']}, "
                   f"Ks {tx['Ks']:.2e} m/s — {SOIL_SOURCE}")
        c.init_saturation = num("초기 포화도 (0~1)", c.init_saturation, "isat", H["init_saturation"])
        if c.soil_mode == "상세":
            c.aniso = num("이방성 Kz/Kx", c.aniso, "aniso", H["aniso"])
            st.markdown("층상 구조(위에서부터, 빈 구간은 위 토성 사용)")
            ldf = pd.DataFrame([asdict(L) for L in c.layers]) if c.layers else pd.DataFrame(
                [dict(top=0.0, bottom=c.H_soil, theta_r=tx["theta_r"], theta_s=tx["theta_s"], alpha=tx["alpha"],
                      n=tx["n"], Ks=tx["Ks"], aniso=1.0)])
            ldf = st.data_editor(ldf, num_rows="dynamic", key="layers", width="stretch",
                                 column_config={"top": st.column_config.NumberColumn("상단(m)", help=None),
                                                "bottom": st.column_config.NumberColumn("하단(m)"),
                                                "theta_r": st.column_config.NumberColumn("θr", help=H["theta_r"]),
                                                "theta_s": st.column_config.NumberColumn("θs", help=H["theta_s"]),
                                                "alpha": st.column_config.NumberColumn("α (1/m)", help=H["alpha"]),
                                                "n": st.column_config.NumberColumn("n", help=H["n"]),
                                                "Ks": st.column_config.NumberColumn("Ks (m/s)", format="%.2e", help=H["Ks"]),
                                                "aniso": st.column_config.NumberColumn("Kz/Kx", help=H["aniso"])})
            c.layers = [SoilLayer(**{k: float(v) for k, v in r.items()}) for r in ldf.dropna().to_dict("records")]
        st.markdown("**유입수·초기 상태**")
        c.q_mm_day = num("침투 플럭스 q (mm/일)", c.q_mm_day, "q", H["q"])
        st.caption(f"= 흙수조 전체 {c.q_mm_day * c.Lx * c.Ly:.3f} L/일")
        c.NO3_in_mgN = num("유입 NO₃-N (mg/L)", c.NO3_in_mgN, "no3", H["NO3_in"])
        c.SO4_in_mg = num("유입 SO₄ (mg/L)", c.SO4_in_mg, "so4", H["SO4_in"])
        c.Ca_in_mg = num("유입 Ca (mg/L)", c.Ca_in_mg, "ca", H["Ca_in"])
        c.alk_in_mgCaCO3 = num("유입 알칼리도 (mg/L as CaCO₃)", c.alk_in_mgCaCO3, "alk", H["alk_in"])
        c.soil_pH0 = num("초기 토양 pH", c.soil_pH0, "ph0", H["soil_pH0"])
        c.soil_NO3_mgN = num("초기 토양수 NO₃-N (mg/L)", c.soil_NO3_mgN, "sno3")
        c.gw_NO3_mgN = num("지하수 배경 NO₃-N (mg/L)", c.gw_NO3_mgN, "gno3")
    with c3:
        st.markdown("**반응재(모든 비교군 동일)**")
        c.ratio_S_CaCO3 = num("황 : CaCO₃ 부피비 (x:1)", c.ratio_S_CaCO3, "ratio", H["ratio"])
        c.temperature = num("온도 (°C)", c.temperature, "temp", H["temperature"])
        st.markdown("**기간·계산**")
        c.days = num("운영 기간 (일)", c.days, "days", H["days"])
        pre = st.selectbox("격자 프리셋", list(GRID_PRESETS) + ["직접 입력"],
                           index=(list(GRID_PRESETS) + ["직접 입력"]).index(c.grid_preset), help=H["grid"])
        c.grid_preset = pre
        if pre != "직접 입력":
            c.nx, c.ny, c.nz = GRID_PRESETS[pre]
        g1, g2, g3 = st.columns(3)
        c.nx = int(g1.number_input("nx", value=int(c.nx), key="nx", step=1))
        c.ny = int(g2.number_input("ny", value=int(c.ny), key="ny", step=1))
        c.nz = int(g3.number_input("nz", value=int(c.nz), key="nz", step=1))
        g = make_grid(c)
        st.caption(f"셀 크기 {g.dx * 1000:.1f} × {g.dy * 1000:.1f} × {g.dz * 1000:.1f} mm · {g.N:,}칸")
        c.flow_mode = st.radio("흐름 해석", ["준정상", "과도"], index=["준정상", "과도"].index(c.flow_mode), horizontal=True,
                               help=H["flow_mode"])
        c.flow_update_days = num("흐름 재계산 주기 (일)", c.flow_update_days, "fud", H["flow_update_days"])
        c.calcite_mode = st.radio("석회석 용해", ["속도론", "평형"], index=["속도론", "평형"].index(c.calcite_mode),
                                  horizontal=True, help=H["calcite_mode"])
        c.n_snapshots = int(num("저장할 3D 장면 수", float(c.n_snapshots), "nsnap"))
        c.light = st.toggle("가벼운 모드(프레임·해상도 축소)", c.light)
        c.redistribute_runoff = st.checkbox("지표 유출 재분배(가정, 기본 켬)", c.redistribute_runoff,
                                            help="구조물 덮개나 포화된 지표에서 못 들어간 물을 열린 흙 표면으로 고르게 보냅니다. "
                                                 "끄면 그 물(과 질산염)은 흙수조 밖으로 사라진 것으로 봅니다.")
        st.markdown("**삽입 교란(가정, 기본 없음)**")
        c.disturb_thick = num("교란대 두께 (m)", c.disturb_thick, "dth", H["disturb"])
        c.disturb_Kmult = num("교란대 K 배율", c.disturb_Kmult, "dkm", H["disturb"])
    with st.expander("고급 파라미터(반응·수송) — 출처/가정 라벨"):
        rows = [{"키": k, "값": c.params.get(k, v[0]), "단위": v[1], "라벨": TX.STATUS_LABELS.get(v[2], v[2]), "설명": v[3]}
                for k, v in REGISTRY.items()]
        pdf = st.data_editor(pd.DataFrame(rows), disabled=["키", "단위", "라벨", "설명"], hide_index=True, key="adv",
                             width="stretch", column_config={"값": st.column_config.NumberColumn(format="%.4g")})
        for r in pdf.to_dict("records"):
            if r["값"] is not None and r["값"] != REGISTRY[r["키"]][0]:
                c.params[r["키"]] = float(r["값"])
            elif r["키"] in c.params and r["값"] == REGISTRY[r["키"]][0]:
                c.params.pop(r["키"])
    errs, warns = validate_common(c)
    for e in errs:
        st.error(e)
    for w in warns:
        st.warning(w)
    if ss().scen:
        st.markdown("**비교군별 반응재 총 질량(통제변수: 모두 같아야 함)**")
        rows = []
        for s_ in ss().scen:
            try:
                so = scenario_obj(s_, c)
                gm = build_geometry(so, c, g) if so.objects else None
                pm = planned_masses(so, c, g, gm) if gm else {"total_g": 0, "pre_mesh_g": 0, "injection_g": 0}
                rows.append({"비교군": s_["name"], "사전 배치(g)": round(pm["pre_mesh_g"], 1), "주입 계획(g)": round(pm["injection_g"], 1),
                             "합계(g)": round(pm["total_g"], 1)})
            except Exception as e:
                rows.append({"비교군": s_["name"], "사전 배치(g)": None, "주입 계획(g)": None, "합계(g)": None})
        mdf = pd.DataFrame(rows)
        st.dataframe(mdf, hide_index=True, width="stretch")
        tot = [r["합계(g)"] for r in rows if r["합계(g)"]]
        if len(tot) > 1 and (max(tot) - min(tot)) / max(max(tot), 1e-9) > 0.05:
            st.warning("비교군별 반응재 총 질량이 5% 넘게 다릅니다. 지표 문서: 질량이 다르면 '구조의 차이'가 아니라 '양의 차이'를 재게 됩니다. "
                       "① 탭의 '질량 맞추기 도우미'로 주입량을 조정하세요.")

# ═════════════════════════════ ③ 실행 ═════════════════════════════════════
def run_all(common: Common, scen_list, area):
    errs, warns = validate_common(common)
    if errs:
        area.error("공통 조건에 계산할 수 없는 값이 있어 실행하지 않습니다.")
        return None
    out = {"control": None, "scen": [], "skipped": [], "common": common.to_dict(), "t": time.time()}
    bar = area.progress(0.0, text="대조군(구조물 없음) 계산 중…")
    t0 = time.time()

    def prog_ctrl(f, msg):
        bar.progress(min(0.99, 0.05 + 0.25 * f), text=f"대조군 {msg}")

    ctrl = run_scenario(common, None, prog_ctrl)
    out["control"] = ctrl
    bt = eval_start_from(ctrl, common.NO3_in_mgN)
    out["eval_start"] = bt
    n = max(1, len(scen_list))
    for i, sc in enumerate(scen_list):
        roles_of = {r["객체"]: r["역할"] for r in sc["role_df"].to_dict("records")} if len(sc["role_df"]) else {}
        e2, _ = ie.validate(sc["events_df"].to_dict("records"), roles_of)
        if not sc["objects"]:
            out["skipped"].append((sc["name"], "3MF가 없습니다."))
            continue
        if e2:
            out["skipped"].append((sc["name"], "설치 방법을 입력하세요. " + " / ".join(e2)))
            continue

        def prog(f, msg, i=i):
            bar.progress(min(0.99, 0.3 + 0.7 * (i + f) / n), text=f"{sc['name']}: {msg}")
        try:
            r = run_scenario(common, scenario_obj(sc, common), prog, eval_start_day=bt)
            r["color"] = viz.SCEN_COLORS[i % len(viz.SCEN_COLORS)]
            out["scen"].append(r)
        except StopRequested:
            raise
        except ValueError as e:
            out["skipped"].append((sc["name"], str(e)))
        except Exception as e:
            out["skipped"].append((sc["name"], f"계산 오류: {e}\n{traceback.format_exc()[-800:]}"))
    out["metrics"] = [M.compute(r, ctrl, common, bt) for r in out["scen"]] + [M.compute(ctrl, None, common, bt)]
    bar.progress(1.0, text=f"완료 · {time.time() - t0:.0f}초")
    return out


with tabs[2]:
    st.markdown("실행하면 **기본 대조군(반응재를 꽂지 않은 같은 흙수조)** 을 먼저 계산하고, 각 비교군을 같은 조건·같은 격자로 계산합니다.")
    errs, warns = validate_common(c)
    for e in errs:
        st.error(e)
    for w in warns:
        st.warning(w)
    plan = []
    for s_ in ss().scen:
        roles_of = {r["객체"]: r["역할"] for r in s_["role_df"].to_dict("records")} if len(s_["role_df"]) else {}
        e2, w2 = ie.validate(s_["events_df"].to_dict("records"), roles_of)
        status = "준비됨" if (s_["objects"] and not e2) else ("3MF 없음" if not s_["objects"] else "설치 방법을 입력하세요")
        plan.append({"비교군": s_["name"], "파일": s_["file_name"], "객체 수": len(s_["objects"]),
                     "설치 단계": len(ie.clean_rows(s_["events_df"].to_dict("records"))), "상태": status})
    if plan:
        st.dataframe(pd.DataFrame(plan), hide_index=True, width="stretch")
    st.caption(f"흐름: {c.flow_mode} ({'Richards 정상해를 ' + str(c.flow_update_days) + '일마다·투수성 5% 변화 시 재계산 — 준정상 근사' if c.flow_mode == '준정상' else '매 단계 과도 Richards'}) · "
               f"석회석: {c.calcite_mode} · 격자 {c.nx}×{c.ny}×{c.nz}")
    r1, r2 = st.columns([1, 1])
    go_run = r1.button("▶ 실행 (대조군 + 비교군 전체)", type="primary", width="stretch", disabled=bool(errs))
    stop = r2.button("⏹ 중단", width="stretch")
    if stop and ss().running:
        ss().running = False
        st.warning("계산을 중단했습니다.")
    elif ss().running and not go_run:
        ss().running = False
        st.warning("이전 계산이 중단되었습니다.")
    if go_run:
        ss().running = True
        area = st.container()
        try:
            res = run_all(copy.deepcopy(c), ss().scen, area)
            if res is not None:
                ss().results = res
                st.success("계산을 마쳤습니다. ④ 결과 탭을 보세요.")
                for nme, why in res["skipped"]:
                    st.error(f"'{nme}' 실행 안 함: {why}")
        except StopRequested:
            st.warning("중단했습니다.")
        finally:
            ss().running = False
    if ss().results:
        R = ss().results
        st.caption(f"마지막 실행: 대조군 + {len(R['scen'])}개 비교군 · 대조군 돌파일(η 평가 시작) "
                   f"{R['eval_start'] if R['eval_start'] is not None else '없음(모사 기간 안에 도달 안 함)'}일")

# ═════════════════════════════ ④ 결과·애니메이션 ══════════════════════════
@st.cache_data(show_spinner="기본 대조군 계산 중(꽂지 않았을 때)…", max_entries=4)
def baseline_control(cdict):
    cc = Common.from_dict(cdict)
    cc.nx, cc.ny, cc.nz = GRID_PRESETS["가벼움"]
    cc.light = True
    cc.n_snapshots = min(cc.n_snapshots, 12)
    r = run_scenario(cc, None)
    bt = eval_start_from(r, cc.NO3_in_mgN)
    return r, M.compute(r, None, cc, bt), bt


with tabs[3]:
    st.info("**모델 한계(항상 표시)** — " + " · ".join(f"{a}: {b}" for a, b in TX.LIMITS[:3]) +
            " … 전체는 ⑤ 탭. 🟡 가정 값은 실험으로 보정 전입니다.")
    R = ss().results
    if R is None:
        st.subheader("기본 대조군 — 반응재를 꽂지 않았을 때의 피해")
        st.caption("업로드한 파일이 없어도 볼 수 있는 기본 그래프·애니메이션입니다(가벼운 격자, 현재 공통 조건).")
        ok_c = not validate_common(c)[0]
        if ok_c and st.toggle("기본 대조군 보기", value=True):
            cr, cm, bt = baseline_control(c.to_dict())
            k1, k2, k3 = st.columns(3)
            k1.metric("지하수 도달 NO₃-N 누적", f"{cm['N_to_gw_g']:.2f} g")
            k2.metric("먹는물 기준(10 mg/L) 도달", f"{cm['arrival_10mg_day']:.0f}일" if np.isfinite(cm['arrival_10mg_day']) else "모사 기간 내 없음")
            k3.metric("유출 농도 50% 도달(돌파)", f"{bt:.0f}일" if bt else "없음")
            show_fig(viz.baseline_damage(cr, c), key="base_dmg")
            ann = viz.param_annotations(c, None, cr, cr["grid"])
            show_fig(viz.results_scene(cr, c, ["NO₃ 오염 플룸", "지하수층 NO₃"], light=True, annotations=ann, height=560),
                     key="base_anim")
    else:
        C_ = Common.from_dict(R["common"])
        all_res = R["scen"] + [R["control"]]
        mets = R["metrics"]
        sc_by_name = {s_["name"]: s_ for s_ in ss().scen}
        st.subheader("비교군 × 지표")
        head, rows = M.summary_table(mets)
        st.dataframe(pd.DataFrame(rows, columns=head), hide_index=True, width="stretch", height=min(900, 36 * (len(rows) + 1)))
        st.caption(f"η 평가 구간: 대조군 유출 농도가 유입의 90%에 도달한 {R['eval_start']}일 이후(도달 지연 착시 제외). '대조군 보정 제거율'은 같은 날 대조군 유출과 비교한 값. 재보충 전·후 구간 η는 아래 표.")
        with st.expander("재보충 전·후 구간별 η, 비교군 메커니즘 설명, 경고"):
            for r, m in zip(R["scen"], mets):
                st.markdown(f"**{r['name']}** — {r['mechanism'] or '(메커니즘 설명 없음)'}")
                st.dataframe(pd.DataFrame([{"구간 시작(일)": s_["from"], "구간 끝(일)": s_["to"], "η (%)": round(s_["eta"], 1)}
                                           for s_ in m["eta_segments"]]), hide_index=True)
                d = r["diag"]
                if np.isfinite(d.get("through_ratio", np.nan)) and d["through_ratio"] < 0.05:
                    st.warning(f"{r['name']}: 반응재 영역을 통과하는 물이 투영면적 강수의 {d['through_ratio'] * 100:.1f}%뿐입니다. "
                               "막힌 바닥·벽 때문에 물이 반응재를 돌아 흐르고 있습니다(역할표의 바닥/벽 역할을 확인하세요).")
                for w in r["warnings"][:12]:
                    st.caption("⚠ " + w)
                mb = r["mass_balance"]
                st.caption("질량수지 오차: " + ", ".join(f"{k.split('_')[0]} {v:.1e}" for k, v in mb.items()))
        view = st.radio("보기", ["설치 과정 애니메이션", "결과 3D(레이어)", "다중 비교(시간 연동)", "그래프", "재보충 이벤트"],
                        horizontal=True)
        show_ann = st.toggle("파라미터 표시(구석 캡션·마우스 툴팁)", value=True)
        light = st.toggle("가벼운 모드", value=C_.light, key="light_view")
        if view == "설치 과정 애니메이션":
            nm = st.selectbox("비교군", [r["name"] for r in R["scen"]])
            r = next(x for x in R["scen"] if x["name"] == nm)
            so = scenario_obj(sc_by_name[nm], C_) if nm in sc_by_name else None
            if so is None:
                st.warning("이 결과의 비교군 입력이 지금 목록에 없어 구조물 모양을 그릴 수 없습니다.")
            else:
                from voxel import Placement as _P
                pl = _P(**r["placement"]) if r.get("placement") else so.placement
                ann = viz.param_annotations(C_, so, r, r["grid"]) if show_ann else None
                show_fig(viz.install_animation(r, so.objects, so.roles, pl, wt_depth=C_.wt_depth, light=light, annotations=ann),
                         key="inst")
                st.caption("단계: " + " → ".join(dict.fromkeys(f["label"].split(" (")[0] for f in r["frames"])))
        elif view == "결과 3D(레이어)":
            a1, a2 = st.columns([2, 1])
            nm = a2.selectbox("비교군/대조군", [r["name"] for r in all_res])
            r = next(x for x in all_res if x["name"] == nm)
            layers = a1.multiselect("레이어", list(viz.LAYER_DEF), default=["반응재 분포", "NO₃ 오염 플룸", "pH 단면", "지하수층 NO₃"])
            cut = a2.slider("단면 자르기(x 위치, 1 = 자르지 않음)", 0.05, 1.0, 1.0, 0.05)
            so = scenario_obj(sc_by_name[nm], C_) if nm in sc_by_name else None
            from voxel import Placement as _P
            pl = _P(**r["placement"]) if r.get("placement") else None
            ann = viz.param_annotations(C_, so, r, r["grid"]) if show_ann else None
            show_fig(viz.results_scene(r, C_, layers, objects=so.objects if so else None, roles=so.roles if so else None,
                                       placement=pl, cut_x=None if cut >= 0.999 else cut * C_.Lx, light=light,
                                       annotations=ann, wt_depth=C_.wt_depth), key="res3d")
        elif view == "다중 비교(시간 연동)":
            b1, b2, b3 = st.columns([2, 1, 1])
            layers = b1.multiselect("레이어", list(viz.LAYER_DEF), default=["반응재 분포", "NO₃ 오염 플룸"])
            metric = b2.selectbox("그래프", ["Cout_mgN", "eta", "SO4_out_mg", "pH_out", "N_to_gw_g"],
                                  format_func=lambda x: {"Cout_mgN": "유출 NO₃-N", "eta": "η", "SO4_out_mg": "유출 SO₄",
                                                         "pH_out": "유출 pH", "N_to_gw_g": "지하수 도달 누적"}[x])
            sync = b3.checkbox("카메라 동기화", value=True, help="한 화면을 돌리면 다른 화면도 같이 돕니다.")
            objs = {}
            for r in R["scen"]:
                if r["name"] in sc_by_name:
                    so = scenario_obj(sc_by_name[r["name"]], C_)
                    from voxel import Placement as _P
                    objs[r["name"]] = (so.objects, so.roles, _P(**r["placement"]))
            fig = viz.compare_view(all_res, C_, layers, metric=metric, light=light, scen_objects=objs)
            if sync:
                show_html(viz.to_html(fig, sync_cameras=True), height=860)
            else:
                show_fig(fig, key="cmp")
        elif view == "그래프":
            bt = R["eval_start"]
            show_fig(viz.timeseries(all_res, "eta", "제거효율 η (%)", 100, eval_start=bt), key="g_eta")
            show_fig(viz.timeseries(all_res, "Cout_mgN", "유출 NO₃-N (mg/L)", hlines=[(10, "먹는물 기준 10")], eval_start=bt), key="g_c")
            show_fig(viz.timeseries(all_res, "N_to_gw_g", "지하수 도달 NO₃-N 누적 (g)"), key="g_gw")
            g1, g2 = st.columns(2)
            with g1:
                show_fig(viz.timeseries(all_res, "pH_below", "pH (반응재 아래 0~10 cm)", hlines=[(C_.soil_pH0 - 0.5, "−0.5"), (C_.soil_pH0 + 0.5, "+0.5")]), key="g_ph")
                show_fig(viz.timeseries(all_res, "SO4_out_mg", "유출 SO₄ (mg/L)", hlines=[(250, "250 mg/L")]), key="g_so4")
            with g2:
                show_fig(viz.timeseries(all_res, "Kreac", "반응재 영역 K (m/s)"), key="g_k")
                show_fig(viz.timeseries(all_res, "Keff", "도메인 유효 K (m/s, 정수두 계산)"), key="g_keff")
            st.markdown("**깊이 프로파일(마지막 시점)**")
            p1, p2, p3 = st.columns(3)
            with p1:
                show_fig(viz.depth_profiles(all_res, mets[:-1] + [mets[-1]], "pH", "pH"), key="dp_ph")
            with p2:
                show_fig(viz.depth_profiles(all_res, mets, "NO3", "NO₃-N (mg/L)"), key="dp_n")
            with p3:
                show_fig(viz.depth_profiles(all_res, mets, "SO4", "SO₄ (mg/L)"), key="dp_s")
        else:
            for r, m in zip(R["scen"], mets):
                st.markdown(f"**{r['name']}** — {m['refill_note']}")
                if m["refill_table"]:
                    st.dataframe(pd.DataFrame([{"회차": x.get("n"), "운영 일": round(x["day"], 1), "방식": x.get("method"),
                                                "성공": x.get("ok"), "η 직전(%)": round((x.get("eta_before") or 0) * 100, 1),
                                                "R 24h(%)": round(x.get("R_24h", 0), 1), "R 이동시간 보정(%)": round(x.get("R_travel", 0), 1),
                                                "주입량(L)": round(x.get("delivered_L", 0) or 0, 2), "T_r(분)": x.get("duration_min"),
                                                "사유": x.get("why", "")} for x in m["refill_table"]]), hide_index=True)

# ═════════════════════════════ ⑤ 지표·메커니즘 설명 ═══════════════════════
with tabs[4]:
    st.subheader("평가지표 (지표 문서 정의)")
    st.caption("지표 이름을 누르면 자세한 설명이 펼쳐집니다.")
    for name, d in TX.METRICS.items():
        with st.expander(f"**{name}** — {d['one']}"):
            st.markdown(f"**무엇을 재나요?** {d['what']}")
            st.markdown(f"**읽는 법** {d['read']}")
            st.markdown(f"**좋은 값의 기준** {d['good']}")
            st.markdown(f"**이 시뮬레이션에서 계산하는 방법** {d['how']}")
            if st.toggle("계산 방법 보기", key="f_" + name):
                st.code(d["formula"], language=None)
    st.subheader("메커니즘 설명")
    for name, d in TX.MECHANISMS.items():
        with st.expander(f"**{name}** — {d['one']}"):
            st.markdown(d["detail"])
            if d["formula"] and st.toggle("계산 방법 보기", key="m_" + name):
                st.code(d["formula"], language=None)
    st.subheader("모델 한계")
    for a, b in TX.LIMITS:
        with st.expander(f"**{a}**"):
            st.markdown(b)

# ═════════════════════════════ ⑥ 검증 ═════════════════════════════════════
with tabs[5]:
    st.markdown("앱을 만들 때 실행한 검증 결과입니다. 통과하지 못한 항목은 **미검증**으로 표시합니다.")
    vp = os.path.join(HERE, "verify", "results.json")
    if os.path.exists(vp):
        with open(vp, encoding="utf-8") as fh:
            V = json.load(fh)
        st.dataframe(pd.DataFrame(V["table"]), hide_index=True, width="stretch")
        st.caption(f"실행 환경: {V.get('env', '')} · {V.get('when', '')}")
        for k_, img in V.get("figures", {}).items():
            p_ = os.path.join(HERE, "verify", img)
            if os.path.exists(p_):
                st.image(p_, caption=k_)
    else:
        st.warning("verify/results.json 이 없습니다. 아래 버튼으로 검증을 실행하세요.")
    st.markdown("**OpenGeoSys 교차검증**")
    import shutil
    ogs_path = st.text_input("ogs 실행 파일 경로(비우면 PATH에서 찾음)", "")
    exe = ogs_path or shutil.which("ogs")
    st.write("OGS 실행 파일: " + (f"`{exe}`" if exe else "**없음 → OGS 교차검증 미수행**"))
    if st.button("검증 다시 실행(가벼움, 수 분)"):
        with st.spinner("검증 실행 중…"):
            sys.path.insert(0, os.path.join(HERE, "verify"))
            import run_verification as RV
            V = RV.run_all(quick=True, ogs_exe=exe, out_dir=os.path.join(HERE, "verify"), part="main")  # V5(격자 세분, 약 40분)는 저장된 V5.json 사용
        st.dataframe(pd.DataFrame(V["table"]), hide_index=True, width="stretch")

# ═════════════════════════════ ⑦ 내보내기 ═════════════════════════════════
with tabs[6]:
    import export_xlsx as X
    st.markdown("**프로젝트 저장·불러오기** (입력 전체 + 3MF 포함, JSON)")
    proj = X.project_to_json(c, ss().scen)
    st.download_button("💾 프로젝트 저장(.json)", proj, file_name="slad_project.json", mime="application/json")
    upj = st.file_uploader("프로젝트 불러오기(.json)", type=["json"], key="projup")
    if upj is not None and st.button("불러오기 적용"):
        try:
            cc, scs = X.project_from_json(upj.getvalue().decode("utf-8"), new_scenario, load_file_into)
            ss().common = cc
            ss().scen = scs
            ss().cur = 0
            st.success("불러왔습니다.")
            st.rerun()
        except Exception as e:
            st.error(f"불러오기 실패: {e}")
    R = ss().results
    if R is None:
        st.info("결과 내보내기는 ③ 실행 후에 쓸 수 있습니다.")
    else:
        C_ = Common.from_dict(R["common"])
        st.markdown("**결과 내보내기**")
        e1, e2, e3 = st.columns(3)
        with e1:
            if st.button("Excel 만들기(.xlsx)"):
                ss().xlsx = X.build_xlsx(R, C_, ss().scen)
            if ss().get("xlsx"):
                st.download_button("📊 Excel 내려받기", ss().xlsx, file_name="slad_결과.xlsx",
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            head, rows = M.summary_table(R["metrics"])
            st.download_button("지표표 CSV", pd.DataFrame(rows, columns=head).to_csv(index=False).encode("utf-8-sig"),
                               file_name="slad_지표.csv", mime="text/csv")
        with e2:
            st.download_button("결과 저장(.npz)", X.results_npz(R), file_name="slad_결과.npz")
            st.download_button("결과 요약(.json)", X.results_json(R), file_name="slad_결과.json", mime="application/json")
            if st.button("그림 PNG 묶음 만들기(.zip)"):
                ss().pngzip = X.png_zip(R, C_)
            if ss().get("pngzip"):
                st.download_button("🖼 PNG 묶음", ss().pngzip, file_name="slad_그림.zip")
        with e3:
            if st.button("애니메이션 HTML 만들기"):
                all_res = R["scen"] + [R["control"]]
                objs = {}
                for r in R["scen"]:
                    s_ = next((x for x in ss().scen if x["name"] == r["name"]), None)
                    if s_:
                        so = scenario_obj(s_, C_)
                        objs[r["name"]] = (so.objects, so.roles, Placement(**r["placement"]))
                fig = viz.compare_view(all_res, C_, ["반응재 분포", "NO₃ 오염 플룸"], light=True, scen_objects=objs)
                ss().html_cmp = viz.to_html(fig, sync_cameras=True).encode("utf-8")
                parts = []
                for r in R["scen"]:
                    if r["name"] in objs:
                        o, rl, pl = objs[r["name"]]
                        parts.append(viz.install_animation(r, o, rl, pl, light=True))
                ss().html_inst = [viz.to_html(f).encode("utf-8") for f in parts]
            if ss().get("html_cmp"):
                st.download_button("🎬 다중 비교 애니메이션.html", ss().html_cmp, file_name="slad_비교_애니메이션.html", mime="text/html")
            for i, h in enumerate(ss().get("html_inst", []) or []):
                st.download_button(f"🎬 설치 과정 {i + 1}.html", h, file_name=f"slad_설치과정_{i + 1}.html", mime="text/html",
                                   key=f"hi{i}")
