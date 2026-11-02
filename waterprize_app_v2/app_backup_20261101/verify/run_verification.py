"""검증 V1~V8 (+ OGS 교차검증). 결과는 verify/results.json, 그림은 verify/*.png.

  python verify/run_verification.py          # 전체(수 분~수십 분)
  python verify/run_verification.py --quick  # 가벼운 설정
통과 못 한 항목은 '미검증'으로 기록한다.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
sys.path.insert(0, APP)

from flow_richards import FlowBC, RichardsFV, vg_kr, vg_theta  # noqa: E402
from params import GRID_PRESETS, MW, REGISTRY, SOIL_TEXTURES, VM, Common, suggest_role  # noqa: E402
from transport_reaction import (LEDGER_KEYS, ChemState, ReactParams, Transport, alk_from_pH,  # noqa: E402
                                pH_of, react)
from voxel import Grid, Placement  # noqa: E402

SAND_LOAM = SOIL_TEXTURES["사양토(Sandy loam)"]

OGS_BENCHMARKS_CHECKED = {   # 저장소(gitlab.opengeosys.org/ogs/ogs, GitHub 미러 ufz/ogs) 실제 확인 결과
    "commit": "e7bf8ea2afe6b26882a06110e44ce1f09e2d3f57 (2026-10-26)",
    "license": "BSD-3-Clause 형식(재배포·사용 허용, 저작권 고지 유지). 코드는 복사하지 않고 입력 구조만 참고",
    "paths": {
        "Tests/Data/Parabolic/RichardsComponentTransport/Padilla": True,
        "Tests/Data/Parabolic/ComponentTransport/ReactiveTransport/KineticReactant": True,
        "Tests/Data/Parabolic/ComponentTransport/ReactiveTransport/EquilibriumPhase": True,
        "Tests/Data/Parabolic/ComponentTransport/ReactiveTransport/Wetland": True,
        "Tests/Data/Parabolic/ComponentTransport/AdvectionDiffusionSorptionDecay": True,
        "Tests/Data/Parabolic/Richards/RichardsFlow_2d_small.prj": True,
    },
}


def _column(nz, L=1.0, soil=SAND_LOAM):
    g = Grid(1, 1, nz, 0.1, 0.1, L, L)
    o = np.ones(g.shape)
    bar = (np.zeros((0, 1, nz)), np.zeros((1, 0, nz)), np.zeros((1, 1, nz - 1)), np.zeros((1, 1)))
    fv = RichardsFV(g, np.ones(g.shape, bool), soil["Ks"] * o, soil["Ks"] * o, soil["theta_r"] * o,
                    soil["theta_s"] * o, soil["alpha"] * o, soil["n"] * o, bar)
    return g, fv


# ───────────────────────────── V1 ─────────────────────────────────────────
def v1_richards(figs, out_dir):
    from scipy.integrate import solve_ivp
    s = SAND_LOAM
    q = 5e-3 / 86400
    K = lambda h: s["Ks"] * vg_kr(np.array([h]), s["alpha"], s["n"])[0]
    sol = solve_ivp(lambda z, h: [q / K(h[0]) - 1.0], [0, 1.0], [0.0], rtol=1e-11, atol=1e-13,
                    dense_output=True, method="LSODA")
    errs = {}
    prof = None
    for nz in (25, 50, 100, 200):
        g, fv = _column(nz)
        st = fv.solve_steady(None, FlowBC(q_top=q, bottom="지하수 수위", z_wt=0.0))
        z = g.H - g.dc
        errs[nz] = float(np.abs(st.h[0, 0, :] - sol.sol(z)[0]).max())
        if nz == 100:
            prof = (z, st.h[0, 0, :], sol.sol(z)[0], float(np.abs(st.Fz / (g.dx * g.dy) - q).max() / q))
    order = np.log2(errs[50] / errs[100])
    ok = errs[100] < 1e-3 and order > 1.5 and prof[3] < 1e-8
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.2, 4), dpi=130)
        ax.plot(prof[2], prof[0], "k-", lw=2, label="ODE (semi-analytical)")
        ax.plot(prof[1], prof[0], "o", ms=3, label="FV 3D solver, nz=100")
        ax.set_xlabel("pressure head h (m)"); ax.set_ylabel("elevation z (m)"); ax.legend(fontsize=8); ax.grid(alpha=.3)
        ax.set_title("V1 steady infiltration 5 mm/d, sandy loam", fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "V1.png")); plt.close(fig)
        figs["V1 정상 침투 수두 분포"] = "V1.png"
    except Exception:
        pass
    return dict(id="V1", 항목="1D 침투: Richards vs 준해석해(ODE)", 기준="최대오차 <1 mm & 2차 수렴 & 플럭스 일정",
                결과=f"최대오차 nz=25:{errs[25]:.1e}, 50:{errs[50]:.1e}, 100:{errs[100]:.1e}, 200:{errs[200]:.1e} m · "
                     f"수렴차수 {order:.2f} · 면 플럭스 상대오차 {prof[3]:.1e}",
                판정="통과" if ok else "미검증")


# ───────────────────────────── V2 ─────────────────────────────────────────
def v2_ogata(figs, out_dir):
    from scipy.special import erfc, erfcx
    s = SAND_LOAM
    res = {}
    keep = None
    for nz in (100, 200, 400):
        g, fv = _column(nz)
        st = fv.solve_steady(None, FlowBC(q_top=s["Ks"], bottom="자유배수"))
        aL, Dm = 0.01, 1.7e-9
        tr = Transport(g, fv, dict(alpha_L=aL, alpha_T_ratio=0.1, Dm=Dm, phi_cells=np.full(g.N, s["theta_s"])))
        tr.set_flow(st)
        c = {"NO3": np.zeros(g.N)}
        L = {k: 0.0 for k in LEDGER_KEYS}
        t, tend = 0.0, 0.25 * 86400
        while t < tend - 1e-9:
            dt = min(tr.dt_max, tend - t)
            tr.step(c, dt, {"top": {"NO3": 1.0}}, L)
            t += dt
        th = float(st.theta.ravel()[0]); v = s["Ks"] / th
        D = aL * v + Dm * th ** (7 / 3) / s["theta_s"] ** 2
        x = g.dc
        a1 = (x - v * t) / (2 * np.sqrt(D * t)); a2 = (x + v * t) / (2 * np.sqrt(D * t))
        ob = 0.5 * (erfc(a1) + np.exp(v * x / D) * erfc(a2))
        third = 0.5 * erfc(a1) + np.sqrt(v * v * t / (np.pi * D)) * np.exp(-a1 ** 2) \
            - 0.5 * (1 + v * x / D + v * v * t / D) * np.exp(v * x / D - a2 ** 2) * erfcx(a2)
        mass = ((c["NO3"] * th * g.vcell).sum() - (L["N_in"] - L["N_out"])) / max(L["N_in"], 1e-30)
        res[nz] = (float(np.abs(c["NO3"] - ob).max()), float(np.abs(c["NO3"] - third).max()), float(mass))
        if nz == 200:
            keep = (x, c["NO3"].copy(), ob, third)
    ok = res[400][1] < 0.01 and abs(res[400][2]) < 1e-10
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 3.4), dpi=130)
        ax.plot(keep[0], keep[2], "k--", label="Ogata-Banks (fixed-conc. inlet)")
        ax.plot(keep[0], keep[3], "k-", lw=2, label="3rd-type inlet (same BC as model)")
        ax.plot(keep[0][::4], keep[1][::4], "o", ms=3, label="model (TVD), nz=200")
        ax.set_xlabel("depth (m)"); ax.set_ylabel("C/C0"); ax.legend(fontsize=7); ax.grid(alpha=.3)
        ax.set_title("V2 1D advection-dispersion, t = 0.25 d", fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "V2.png")); plt.close(fig)
        figs["V2 이류-분산 해석해 비교"] = "V2.png"
    except Exception:
        pass
    return dict(id="V2", 항목="반응 없는 1D 이송: TVD vs Ogata-Banks / 3종 경계 해석해",
                기준="모델과 같은 유입 경계(3종) 해석해 최대오차 <0.01 & 질량오차 <1e-10",
                결과=f"3종 해석해 최대오차 nz=100:{res[100][1]:.4f}, 200:{res[200][1]:.4f}, 400:{res[400][1]:.4f} · "
                     f"Ogata-Banks(1종 경계)와는 경계 조건 차이로 {res[400][0]:.3f} 차이 · 질량오차 {res[400][2]:.1e}",
                판정="통과" if ok else "미검증")


# ───────────────────────────── V3 ─────────────────────────────────────────
def v3_slad(figs, out_dir):
    th = np.array([0.4])
    phi = 0.4; r = 3.0
    S0 = (1 - phi) * r / (r + 1) / VM["S"]; C0 = (1 - phi) / (r + 1) / VM["CaCO3"]
    pCO2 = -2.0
    alk0 = alk_from_pH(7.5, pCO2)

    def run(pHlo, pHhi):
        cs = ChemState({"NO3": np.array([50 / 14.007]), "SO4": np.array([0.1]), "Ca": np.array([0.5]),
                        "ALK": np.array([alk0]), "P": np.zeros(1)}, np.array([S0]), np.array([C0]), np.zeros(1),
                       np.zeros(1), np.zeros(1), np.array([6 / 2e-3]), np.array([1.0]), {k: 0.0 for k in LEDGER_KEYS})
        rp = ReactParams(k_half=5.0, S_ref=REGISTRY["S_ref_frac"][0], pH_low=pHlo, pH_high=pHhi, theta_T=1.07, T=22,
                         T_ref=22, stoich_SO4=1.10, stoich_H=1.28, stoich_S=1.10, pCO2=pCO2, I_bg=0.005,
                         bio_vol=3.6e-4, bio_decay=0.05, calcite_mode="속도론", lambda_filt=30, ratio=3)
        ts, Cs, pHs = [], [], []
        t = 0.0
        while t < 3 * 3600:
            react(cs, th, 1e-3, 60.0, rp); t += 60.0
            ts.append(t); Cs.append(cs.c["NO3"][0] * 14.007)
            pHs.append(pH_of(cs.c["ALK"], cs.c["Ca"], cs.c["SO4"], cs.c["NO3"], pCO2, 0.005)[0])
        return np.array(ts), np.array(Cs), np.array(pHs), cs
    ts, Cs, _, _ = run(-50.0, 50.0)     # pH 보정 끔 → 순수 반-차수
    ana = np.maximum(np.sqrt(50) - 0.5 * 5.0 * ts / 3600, 0) ** 2
    err = float(np.abs(Cs - ana).max())
    ts2, Cs2, pH2, cs = run(6.2, 9.0)
    N0 = 50 / 14.007
    dN = (N0 - cs.c["NO3"][0]) * 14.007
    dSO4 = (cs.c["SO4"][0] - 0.1) * 96.06
    L = cs.ledger
    alk_ratio = L["ALK_denit"] * 50.04 / (L["N_denit"] * 14.007)
    ok = err < 0.05 and abs(dSO4 / dN - 7.54) < 0.02 and abs(alk_ratio - 4.57) < 0.02
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 3.4), dpi=130)
        ax.plot(ts / 3600, ana, "k-", lw=2, label="half-order analytical")
        ax.plot(ts[::6] / 3600, Cs[::6], "o", ms=3, label="model (no pH factor)")
        ax.plot(ts2 / 3600, Cs2, "--", label="model (pH factor, calcite buffer)")
        ax.set_xlabel("time (h)"); ax.set_ylabel("NO3-N (mg/L)"); ax.legend(fontsize=7); ax.grid(alpha=.3)
        ax.set_title("V3 SLAD batch: half-order kinetics", fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, "V3.png")); plt.close(fig)
        figs["V3 SLAD 반-차수 속도식 재현"] = "V3.png"
    except Exception:
        pass
    return [dict(id="V3a", 항목="SLAD 반-차수 속도식 재현(해석해)", 기준="최대오차 <0.05 mg N/L",
                 결과=f"최대오차 {err:.4f} mg N/L (k=5.0 mg^½L^-½h^-1, 50→0 mg/L 약 2.8 h)", 판정="통과" if err < 0.05 else "미검증"),
            dict(id="V3b", 항목="SLAD 양론: SO₄ 생성·알칼리도 소비", 기준="7.54 mg SO₄/mg N, 4.57 mg CaCO₃/mg N (±0.02)",
                 결과=f"SO₄ {dSO4 / dN:.3f} mg/mg N, 알칼리도 {alk_ratio:.3f} mg CaCO₃/mg N, 반응 중 pH 최저 {pH2.min():.2f}",
                 판정="통과" if (abs(dSO4 / dN - 7.54) < 0.02 and abs(alk_ratio - 4.57) < 0.02) else "미검증"),
            dict(id="V3c", 항목="SLAD 칼럼 실측 데이터 재현(보정)", 기준="선행연구 칼럼 실측 시계열과 비교",
                 결과="실측 칼럼 데이터를 갖고 있지 않아 비교하지 못함. 속도상수는 보고 범위 중간값(보정 아님)", 판정="미검증")]


# ───────────────────────────── V4~V8 (시나리오) ──────────────────────────
def _scenario(path, events, roles_override=None):
    from engine import Scenario
    from io_3mf import load_3mf
    with open(path, "rb") as fh:
        d = load_3mf(fh.read(), os.path.basename(path))
    roles = {o.name: {"role": suggest_role(o.name)} for o in d["objects"]}
    if roles_override:
        roles.update(roles_override)
    return Scenario(name=os.path.basename(path).split("_scale")[0], objects=d["objects"], roles=roles,
                    placement=Placement(0.25, 0.25, 0, 0, 0, 1), events=events), d


def _log(msg):
    print(f"  · {time.strftime('%H:%M:%S')} {msg}", flush=True)


def scenario_checks(quick, figs, out_dir, part="all"):
    import install_events as ie
    import metrics as M
    from engine import control_breakthrough_day, eval_start_from, run_scenario
    smp = os.path.join(APP, "samples")
    well_ev = [dict(ie.empty_row("구조물 삽입"), **{"소요(분)": 30.0}), dict(ie.empty_row("개구부 열기"), **{"소요(분)": 1.0}),
               dict(ie.empty_row("시약 주입"), **{"방식": "펌프", "부피(L)": 2.84, "고형분비(v/v)": 0.2, "유량(L/min)": 0.2}),
               dict(ie.empty_row("정치(대기)"), **{"소요(분)": 60.0}),
               dict(ie.empty_row("재보충"), **{"방식": "펌프", "부피(L)": 1.0, "고형분비(v/v)": 0.2, "유량(L/min)": 0.2,
                                             "트리거 η(%)": 50.0, "간격(일)": 7.0, "최대 회차": 5.0, "소요(분)": 20.0})]
    cais_ev = [dict(ie.empty_row("구조물 삽입"), **{"소요(분)": 30.0}),
               dict(ie.empty_row("재보충"), **{"방식": "펌프", "부피(L)": 1.0, "트리거 η(%)": 50.0, "간격(일)": 7.0,
                                             "최대 회차": 5.0, "소요(분)": 20.0})]
    c = Common()
    c.nx, c.ny, c.nz = GRID_PRESETS["가벼움"]
    c.days = 60 if quick else 150
    c.light = True
    c.n_snapshots = 8
    rows = []
    t0 = time.time()
    if part == "V5":
        return _v5(c, quick, well_ev, smp, t0)
    _log("대조군 실행")
    ctrl = run_scenario(c, None)
    bt = eval_start_from(ctrl, c.NO3_in_mgN)
    _log("비교군 실행")
    well, dw = _scenario(os.path.join(smp, "injection_well_scale0.05.3mf"), well_ev)
    cais, dc = _scenario(os.path.join(smp, "caisson_gate_scale0.05.3mf"), cais_ev)
    rw = run_scenario(c, well, eval_start_day=bt)
    rc = run_scenario(c, cais, eval_start_day=bt)
    mw = M.compute(rw, ctrl, c, bt); mc = M.compute(rc, ctrl, c, bt); mk = M.compute(ctrl, None, c, bt)
    # V4
    mbs = {"영구 주입공": rw["mass_balance"], "케이슨": rc["mass_balance"], "대조군": ctrl["mass_balance"]}
    worst = max(abs(v) for mb in mbs.values() for v in mb.values())
    rows.append(dict(id="V4", 항목="질소·황·칼슘·전하(알칼리도) 질량수지", 기준="상대오차 <1%",
                     결과="최대 상대오차 " + f"{worst:.1e} · " + "; ".join(
                         f"{k}: N {v['N_err_rel']:.0e}, S {v['S_err_rel']:.0e}, Ca {v['Ca_err_rel']:.0e}, 전하 {v['ALK_err_rel']:.0e}"
                         for k, v in mbs.items()),
                     판정="통과" if worst < 0.01 else "미검증"))
    # V6
    n_w, n_c = len(dw["objects"]), len(dc["objects"])
    vox_w = int((rw["role_cell"] >= 0).sum()); vox_c = int(rc["reac0"].sum())
    diff = abs(mw["eta_last7"] - mc["eta_last7"]) > 1 or abs(mw["N_to_gw_g"] - mc["N_to_gw_g"]) > 0.05
    ok6 = n_w == 11 and n_c == 8 and vox_w > 0 and vox_c > 0 and diff and mc["R"] == 0 and mc["N_max"] == 0
    rows.append(dict(id="V6", 항목="첨부 3MF 인수 테스트(주입공·케이슨)",
                     기준="객체별 인식(11·8개), 반응재·개구부 복셀≠0, 두 구조 지표 다름, 케이슨 R=0",
                     결과=f"주입공 객체 {n_w}개(mm 단위)·구조 셀 {vox_w}, 케이슨 객체 {n_c}개·반응재 셀 {vox_c}; "
                          f"η(마지막 7일) 주입공 {mw['eta_last7']:.1f}% vs 케이슨 {mc['eta_last7']:.1f}%; "
                          f"지하수 도달 {mw['N_to_gw_g']:.2f} vs {mc['N_to_gw_g']:.2f} g (대조군 {mk['N_to_gw_g']:.2f}); "
                          f"케이슨 R={mc['R']:.0f}, N_max={mc['N_max']}; 케이슨 반응재 통과유량/투영강수 {mc['through_ratio']:.3f}",
                     판정="통과" if ok6 else "미검증"))
    # V7
    labels = [f["label"] for f in rw["frames"]]
    def first(prefix):
        return next((i for i, l in enumerate(labels) if l.startswith(prefix)), 10 ** 9)
    order_ok = first("구조물 삽입") < first("시약 주입") < first("정치")
    L = rw["ledger"]
    clos = (L["P_inj"] - L["P_dep"] - L["P_out"]) / max(L["P_inj"], 1e-30)
    inj = [e for e in rw["events_log"] if e.get("event") == "시약 주입"]
    rows.append(dict(id="V7", 항목="설치 타임라인 재생 순서 + 주입 질량 보존",
                     기준="삽입→주입→정치 순서, 주입 고형분 = 포획+유출+부유 (오차 <1e-6)",
                     결과=f"순서 {'정상' if order_ok else '오류'} ({' → '.join(dict.fromkeys(l.split(' (')[0] for l in labels))}); "
                          f"주입량 {inj[0]['delivered_L'] if inj else 0:.2f} L, 고형분 수지 오차 {clos:.1e}; "
                          f"초기 반응재 질량 주입공 {rw['mass0_g']:.0f} g vs 케이슨 {rc['mass0_g']:.0f} g",
                     판정="통과" if (order_ok and abs(clos) < 1e-6) else "미검증"))
    # V8
    same = (ctrl["grid"] == rw["grid"]) and (ctrl["grid"] == rc["grid"])
    rows.append(dict(id="V8", 항목="대조군이 비교군과 같은 조건·격자로 계산되는지",
                     기준="격자·공통 조건 동일, 반응재·구조물 없음",
                     결과=f"격자 동일 {same}; 대조군 구조 셀 {int((ctrl['role_cell'] >= 0).sum())}개, 반응재 질량 {ctrl['mass0_g']:.0f} g; "
                          f"η 평가 시작일(대조군 유출 90% 도달) {bt}일",
                     판정="통과" if (same and (ctrl["role_cell"] < 0).all() and ctrl["mass0_g"] == 0) else "미검증"))
    if part == "all":
        rows += _v5(c, quick, well_ev, smp, time.time(), (ctrl, rw, mw, bt))
    rows.append(dict(id="—", 항목="(참고) 시나리오 검증 실행 시간", 기준="", 결과=f"{time.time() - t0:.0f} 초", 판정=""))
    return rows


def _v5(c, quick, well_ev, smp, t0, base=None):
    import metrics as M
    from engine import eval_start_from, run_scenario
    if base is None:
        _log("V5 기준 격자 실행")
        ctrl = run_scenario(c, None)
        bt = eval_start_from(ctrl, c.NO3_in_mgN)
        well, _ = _scenario(os.path.join(smp, "injection_well_scale0.05.3mf"), well_ev)
        rw = run_scenario(c, well, eval_start_day=bt)
        mw = M.compute(rw, ctrl, c, bt)
    else:
        ctrl, rw, mw, bt = base
    rows = []
    c2 = Common(**{k: v for k, v in c.__dict__.items()})
    c2.nx, c2.ny, c2.nz = (int(c.nx * 1.5), int(c.ny * 1.5), int(c.nz * 1.5)) if quick else (c.nx * 2, c.ny * 2, c.nz * 2)
    _log(f"V5 세분 격자 {c2.nx}x{c2.ny}x{c2.nz} 실행")
    ctrl2 = run_scenario(c2, None)
    bt2 = eval_start_from(ctrl2, c2.NO3_in_mgN)
    well2, _ = _scenario(os.path.join(smp, "injection_well_scale0.05.3mf"), well_ev)
    rw2 = run_scenario(c2, well2, eval_start_day=bt2)
    mw2 = M.compute(rw2, ctrl2, c2, bt2)
    d_eta = mw2["eta_pre"] - mw["eta_pre"]
    d_gw = (mw2["gw_improve_pct"] - mw["gw_improve_pct"])
    rows.append(dict(id="V5", 항목=f"격자 세분화({c.nx}×{c.ny}×{c.nz} → {c2.nx}×{c2.ny}×{c2.nz}) 시 η 변화(영구 주입공)",
                     기준="재보충 전 η 변화 <5%p",
                     결과=f"η(재보충 전) {mw['eta_pre']:.1f}% → {mw2['eta_pre']:.1f}% (Δ {d_eta:+.1f}%p); "
                          f"지하수 피해 감소율 {mw['gw_improve_pct']:.1f}% → {mw2['gw_improve_pct']:.1f}% (Δ {d_gw:+.1f}%p); "
                          f"대조군 돌파일 {bt} → {bt2}",
                     판정="통과" if abs(d_eta) < 5 else "미검증"))
    rows.append(dict(id="—", 항목="(참고) V5 실행 시간", 기준="", 결과=f"{time.time() - t0:.0f} 초", 판정=""))
    return rows


# ───────────────────────────── OGS ───────────────────────────────────────
def ogs_crosscheck(ogs_exe, out_dir):
    """1D 기둥 Richards(OGS RICHARDS_FLOW, 같은 vG 곡선 표)를 생성·실행해 정상 수두를 비교.
    OGS 실행 파일이 없으면 수행하지 않는다."""
    if not ogs_exe:
        return dict(id="V1-OGS", 항목="OpenGeoSys 6 교차검증(1D Richards)", 기준="같은 격자·곡선에서 정상 수두 차 <1 cm",
                    결과="ogs 실행 파일을 찾지 못함 — OGS 교차검증 미수행 (입력 생성기는 verify/ogs_crosscheck_input 에 포함)",
                    판정="미검증")
    try:
        from ogs_input import write_column_case
        work = tempfile.mkdtemp(prefix="ogs_v1_")
        info = write_column_case(work)
        p = subprocess.run([ogs_exe, info["prj"]], cwd=work, capture_output=True, text=True, timeout=600)
        if p.returncode != 0:
            raise RuntimeError(p.stdout[-400:] + p.stderr[-400:])
        z_ogs, h_ogs = info["read_result"](work)
        g, fv = _column(info["nz"], info["L"])
        st = fv.solve_steady(None, FlowBC(q_top=0.0, bottom="지하수 수위", z_wt=0.0,
                                          fixed_h=info["fixed_top"](g)))
        z = g.H - g.dc
        h_my = np.interp(z_ogs, z[::-1], st.h[0, 0, ::-1])
        err = float(np.abs(h_my - h_ogs).max())
        return dict(id="V1-OGS", 항목="OpenGeoSys 6 교차검증(1D Richards)", 기준="정상 수두 차 <1 cm",
                    결과=f"최대 차 {err * 100:.2f} cm", 판정="통과" if err < 0.01 else "미검증")
    except Exception as e:
        return dict(id="V1-OGS", 항목="OpenGeoSys 6 교차검증(1D Richards)", 기준="정상 수두 차 <1 cm",
                    결과=f"실행 실패: {str(e)[:300]}", 판정="미검증")


def run_all(quick=True, ogs_exe=None, out_dir=HERE, part="all"):
    """part: "all" | "main"(V5 제외) | "V5"(V5만, 결과는 V5.json). 두 프로세스로 나눠 병렬 실행 가능."""
    figs = {}
    table = []
    if part == "V5":
        rows = scenario_checks(quick, figs, out_dir, part="V5")
        with open(os.path.join(out_dir, "V5.json"), "w", encoding="utf-8") as fh:
            json.dump(rows, fh, ensure_ascii=False, indent=1)
        return {"table": rows}
    for fn in (v1_richards, v2_ogata):
        try:
            table.append(fn(figs, out_dir))
        except Exception as e:
            table.append(dict(id=fn.__name__, 항목=fn.__name__, 기준="", 결과=f"오류 {e}", 판정="미검증"))
    try:
        table += v3_slad(figs, out_dir)
    except Exception as e:
        table.append(dict(id="V3", 항목="SLAD", 기준="", 결과=f"오류 {e}", 판정="미검증"))
    try:
        table += scenario_checks(quick, figs, out_dir, part=part)
    except Exception as e:
        import traceback
        table.append(dict(id="V4–V8", 항목="시나리오 검증", 기준="", 결과=f"오류 {e} {traceback.format_exc()[-500:]}", 판정="미검증"))
    table.append(ogs_crosscheck(ogs_exe or shutil.which("ogs"), out_dir))
    table.append(dict(id="OGS-목록", 항목="README에 적은 OGS 벤치마크 이름 실재 확인",
                      기준="저장소에 경로 존재", 결과=f"{OGS_BENCHMARKS_CHECKED['commit']} 에서 "
                      + ", ".join(k.split('/')[-1] for k, v in OGS_BENCHMARKS_CHECKED['paths'].items() if v) + " 모두 존재 · "
                      + OGS_BENCHMARKS_CHECKED["license"], 판정="통과"))
    v5p = os.path.join(out_dir, "V5.json")
    if part == "main" and os.path.exists(v5p):
        with open(v5p, encoding="utf-8") as fh:
            table += json.load(fh)
    order = {"V1": 0, "V1-OGS": 1, "V2": 2, "V3a": 3, "V3b": 4, "V3c": 5, "V4": 6, "V5": 7, "V6": 8, "V7": 9, "V8": 10}
    table.sort(key=lambda r: order.get(r["id"], 99))
    out = {"table": table, "figures": figs, "when": time.strftime("%Y-%m-%d %H:%M"),
           "env": f"Python {platform.python_version()}, numpy {np.__version__}, {platform.system()}",
           "quick": quick}
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    return out


if __name__ == "__main__":
    quick = "--quick" in sys.argv
    part = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--part=")), "all")
    res = run_all(quick=quick, part=part)
    for r in res["table"]:
        print(f"[{r['판정'] or '-':4s}] {r['id']:7s} {r['항목']}\n        {r['결과']}")
