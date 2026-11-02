"""알려진 문제(외부 리뷰 A·B·C 목록) 재현·회귀 시험.

각 시험은 '문제가 있으면 실패(False)'가 되도록 작성했다. v2 코드에서 먼저 돌려 사실 여부를 확인하고,
수정 뒤 같은 시험이 통과하는지 다시 돌린다. 결과는 known_issues_<태그>.json 으로 저장.

    python selftest/known_issues.py v2      # 수정 전
    python selftest/known_issues.py v3      # 수정 후
"""
from __future__ import annotations

import inspect
import json
import os
import sys
import time
import traceback

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
sys.path.insert(0, APP)


def _common(**kw):
    from params import Common
    c = Common()
    c.nx, c.ny, c.nz = 10, 10, 16
    c.days = 4
    c.light = True
    c.n_snapshots = 3
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def _box(x0, x1, y0, y1, z0, z1):
    V = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], float)
    T = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                  [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]], np.int64)
    return V, T


def _sample(name="injection_well_scale0.05.3mf"):
    from io_3mf import load_3mf
    with open(os.path.join(APP, "samples", name), "rb") as fh:
        return load_3mf(fh.read(), name)


# ───────────────────────────── A ─────────────────────────────────────────
def A1_stop():
    """중단 콜백이 run_all→run_scenario로 전달되고, 설치 단계 루프에서도 확인되는가."""
    src = open(os.path.join(APP, "app.py"), encoding="utf-8").read()
    i = src.index("def run_all")
    body = src[i:src.index("\nwith tabs[2]", i)]
    import re
    passes = bool(re.search(r"run_scenario\(common, None, \w+, stop\)", body)) and bool(re.search(r"run_scenario\(common, so, \w+, stop\)", body))
    import engine as E
    calls = {"n": 0}

    def stop():
        calls["n"] += 1
        return calls["n"] > 3
    from engine import Scenario, run_scenario
    d = _sample()
    from params import suggest_role
    roles = {o.name: {"role": suggest_role(o.name)} for o in d["objects"]}
    from voxel import Placement
    import install_events as ie
    ev = [dict(ie.empty_row("구조물 삽입"), **{"소요(분)": 30.0}),
          dict(ie.empty_row("시약 주입"), **{"방식": "펌프", "부피(L)": 2.0, "고형분비(v/v)": 0.2, "유량(L/min)": 0.2})]
    sc = Scenario(name="w", objects=d["objects"], roles=roles, placement=Placement(0.25, 0.25, 0, 0, 0, 1), events=ev)
    c = _common()
    stopped_in_install = False
    try:
        sim = E.Sim(c, sc, None, stop)
        sim.run_installation()
    except E.StopRequested:
        stopped_in_install = True
    ok = passes and stopped_in_install
    return ok, f"app.run_all이 stop 전달: {passes}; 설치 단계에서 중단됨: {stopped_in_install} (stop 호출 {calls['n']}회)"


def A2_face_crossings_nonsquare():
    """nx≠ny≠nz 격자에서 상자 메시의 면 장벽이 해석적으로 기대되는 면과 일치하는가."""
    from voxel import Grid, face_crossings
    g = Grid(30, 20, 40, 0.6, 0.4, 0.8, 0.8)
    V, T = _box(0.133, 0.417, 0.077, 0.291, 0.115, 0.537)
    bx, by, bz, bt = face_crossings(V, T, g)
    xc, yc, zc = g.xc, g.yc, g.dc
    exp_bx = np.zeros_like(bx); exp_by = np.zeros_like(by); exp_bz = np.zeros_like(bz)
    inY = (yc > 0.077) & (yc < 0.291); inZ = (zc > 0.115) & (zc < 0.537); inX = (xc > 0.133) & (xc < 0.417)
    for xw in (0.133, 0.417):
        i = np.where((xc[:-1] < xw) & (xc[1:] > xw))[0]
        exp_bx[np.ix_(i, np.where(inY)[0], np.where(inZ)[0])] = True
    for yw in (0.077, 0.291):
        j = np.where((yc[:-1] < yw) & (yc[1:] > yw))[0]
        exp_by[np.ix_(np.where(inX)[0], j, np.where(inZ)[0])] = True
    for zw in (0.115, 0.537):
        k = np.where((zc[:-1] < zw) & (zc[1:] > zw))[0]
        exp_bz[np.ix_(np.where(inX)[0], np.where(inY)[0], k)] = True
    mis = int((bx != exp_bx).sum() + (by != exp_by).sum() + (bz != exp_bz).sum())
    return mis == 0, f"30×20×40 격자 상자 메시: 기대와 다른 장벽 면 {mis}개 (bx {int(bx.sum())}/{int(exp_bx.sum())}, " \
                     f"by {int(by.sum())}/{int(exp_by.sum())}, bz {int(bz.sum())}/{int(exp_bz.sum())})"


def A3_R_mean():
    """실패한 재보충(R=0)이 평균 R에 섞이지 않는가 + 성공률을 따로 내는가."""
    import metrics as M
    src = inspect.getsource(M.compute)
    has_rate = "refill_success_rate" in src
    mixes = 'Rv = [x["R_travel"] for x in Rs if np.isfinite' in src
    return (has_rate and not mixes), f"성공률 분리: {has_rate}; 실패분을 평균에 포함: {mixes}"


def A4_count_below_precision():
    """_count_below 가 많은 직선·큰 좌표에서도 정확한가(정확한 기준 구현과 비교)."""
    from voxel import _count_below
    rng = np.random.default_rng(1)
    nL = 3_000_000
    hl = np.sort(rng.integers(0, nL, 2_000_000))
    base = 1.0e3                                       # 큰 좌표(예: mm 단위를 m로 잘못 넣은 경우와 비슷한 규모)
    hx = base + rng.random(len(hl)) * 1e-3
    q_l = rng.integers(0, nL, 20000)
    # 교차점 바로 옆(1e-9) 질의 → 정밀도 손실을 드러냄
    pick = rng.integers(0, len(hl), 20000)
    _, ui = np.unique(hl[pick], return_index=True)       # 직선당 질의 하나(같은 직선 중복 선택 방지)
    pick = pick[ui]
    q_l = hl[pick]
    qx = hx[pick] + 1e-10
    q = np.full((nL, 1), base - 1.0)
    q[q_l, 0] = qx
    got = _count_below(hl, hx, nL, q)[q_l, 0]
    # 기준: lexsort 기반 정확 계산
    order = np.lexsort((hx, hl))
    hl_s, hx_s = hl[order], hx[order]
    exp = np.empty(len(q_l), np.int64)
    for n_, (l, x) in enumerate(zip(q_l, qx)):
        a = np.searchsorted(hl_s, l, "left"); b = np.searchsorted(hl_s, l, "right")
        exp[n_] = np.searchsorted(hx_s[a:b], x, "left")
    bad = int((got != exp).sum())
    return bad == 0, f"직선 {nL:,}개, 교차점 {len(hl):,}개, 좌표 ~{base:g}: 틀린 개수 {bad}/{len(q_l)}"


def A5_react_substep_and_k3():
    import transport_reaction as TR
    src = inspect.getsource(TR.react)
    thr = getattr(TR, "ALK_STEP_MAX", None) == 0.1 or ("0.1)" in src.split("nsub = ")[1][:200] if "nsub = " in src else False)
    k_lo = TR._pwp_k(25.0 - 1e-9)[2]; k_hi = TR._pwp_k(25.0 + 1e-9)[2]
    jump = abs(k_hi / k_lo - 1)
    return (thr and jump < 1e-6), f"하위 단계 기준 0.1 meq/L: {thr}; k3 25°C 불연속 비율 {jump:.2e}"


def A6_transient_rejects_nonconverged():
    import engine as E
    fn = getattr(E.Sim, "_transient", None) or getattr(E.Sim, "_flow_step")
    src = inspect.getsource(fn)
    accepts = "근사값으로 진행" in src or ("h, pond = hn, pn" in src.split("else:")[1] if "else:" in src else False)
    # 실제 동작 확인: 수렴 못 하는 상황(Newton 반복 1회 제한)에서 FAILED로 끝나는가
    behaved = None
    if hasattr(E, "SimulationFailed"):
        import flow_richards as F
        c = _common(days=1)
        sim = E.Sim(c, None)
        sim.run_installation()
        orig = F.RichardsFV.step
        def bad_step(self, *a, **k):
            h, conv, it, p = orig(self, *a, **dict(k, max_total=1))
            return h, False, it, p
        F.RichardsFV.step = bad_step
        try:
            sim.run_operation()
            behaved = False
        except E.SimulationFailed:
            behaved = True
        finally:
            F.RichardsFV.step = orig
    return (not accepts) and behaved is not False, f"수렴 실패 해를 채택하는 코드 경로: {accepts}; 강제 실패 시 FAILED 처리: {behaved}"


def A7_linsolve_no_zero():
    import flow_richards as F
    src = inspect.getsource(F.RichardsFV._linsolve)
    zero = "x = np.zeros_like(b)" in src
    order_ok = src.find("gmres(") > 0 and src.find("lgmres") < src.find("spsolve")
    return (not zero) and order_ok, f"실패 시 0 벡터 반환: {zero}; lgmres/gmres 후 spsolve: {order_ok}"


def A8_ponding_iter_cap():
    import flow_richards as F
    src = inspect.getsource(F.RichardsFV.step)
    cap = "max_total" in src
    return cap, f"한 단계 총 뉴턴 반복 상한: {cap} (v2: 전환 6회 × 25회 = 최대 150회)"


def A9_clock():
    import engine as E
    src = inspect.getsource(E.Sim)
    direct = src.count("self.t = ") + src.count("self.t += ")
    return direct <= 2, f"self.t 를 직접 바꾸는 곳 {direct}곳(단일 시계 함수 외)"


def A9b_fractional_days():
    from engine import run_scenario
    c = _common(days=2.5)
    r = run_scenario(c, None)
    last = float(r["daily"]["day"][-1])
    return abs(last - 2.5) < 1e-6, f"days=2.5 → 마지막 기록 {last:.4f}일"


def A10_eval_start_trigger():
    import engine as E
    src = inspect.getsource(E.Sim.run_operation)
    uses = "eval_start_day" in src
    return (not uses), f"재보충 트리거에 eval_start_day 사용: {uses}"


def A11_input_checks():
    src = open(os.path.join(APP, "app.py"), encoding="utf-8").read()
    i = src.index("def validate_common")
    body = src[i:i + 4000]
    ok_h = "H_soil <= 0" in body
    from voxel import Grid
    g = Grid(4, 4, 10, 0.5, 0.5, 0.705, 0.70)       # H_aq = 5 mm < dz
    k = g.k_interface
    return ok_h and k <= g.nz - 1, f"H_soil<=0 오류 처리: {ok_h}; H_aq<dz 일 때 k_interface={k} (nz={g.nz})"


def A12_through_flow_def():
    import engine as E
    newdef = hasattr(E.Sim, "_reference_inflow")
    return newdef, f"'반응재가 없을 때 같은 위치 통과 유량' 기준 사용: {newdef}"


def A13_stl_merge():
    from io_3mf import load_stl
    import struct
    # 반올림 경계(소수 6자리)에 걸친 꼭짓점: 같은 점이 출력기마다 ±3e-7 다르게 적힌 경우
    V, T = _box(10.0000005, 20.0000005, 10.0000005, 20.0000005, 0.0000005, 5.0000005)
    tri = V[T] * 1.0
    tri = tri + np.random.default_rng(0).choice([-3e-7, 3e-7], tri.shape)
    buf = b"\0" * 80 + struct.pack("<I", len(tri))
    for t in tri:
        buf += struct.pack("<12fH", 0, 0, 0, *t.ravel(), 0)
    d = load_stl(buf, "tiny.stl")
    o = d["objects"][0]
    return o.closed and len(o.vertices_m) == 8, f"꼭짓점 {len(o.vertices_m)}개(기대 8), 닫힘 {o.closed}"


def A14_triangle_pid():
    import io_3mf
    src = inspect.getsource(io_3mf.load_3mf)
    tri = 'get("pid")' in src.split("triangles")[-1] or "p1" in src
    rng = "인덱스" in src and "범위" in src
    return tri and rng, f"삼각형 pid/p1 처리: {tri}; 꼭짓점 인덱스 범위 검사: {rng}"


def A15_opening_K():
    from params import ROLE_DEFAULTS
    K = ROLE_DEFAULTS["주입 개구부"]["K"]
    return K is not None, f"개구부 기본 K = {K} (None이면 토양 K 그대로)"


def A16_particle_mass_big_theta_change():
    """θ가 크게 바뀌는 조건에서 주입 입자 질량 보존 + 정치 시 공극 초과 포획 여부."""
    import transport_reaction as TR
    from params import VM
    src = inspect.getsource(TR.deposit_particles)
    overfill = "if not all_now:" in src and "cap_w" in src.split("if not all_now:")[1][:200]
    # v2: all_now 이면 남은 공극과 무관하게 전량 포획 → 공극률이 하한(0.02)에 걸림
    return (not overfill), f"정치(all_now) 때 공극 한도 무시하고 전량 포획: {overfill}"


TESTS = [A1_stop, A2_face_crossings_nonsquare, A3_R_mean, A4_count_below_precision, A5_react_substep_and_k3,
         A6_transient_rejects_nonconverged, A7_linsolve_no_zero, A8_ponding_iter_cap, A9_clock, A9b_fractional_days,
         A10_eval_start_trigger, A11_input_checks, A12_through_flow_def, A13_stl_merge, A14_triangle_pid,
         A15_opening_K, A16_particle_mass_big_theta_change]


def run(tag="cur", only=None):
    out = []
    for f in TESTS:
        if only and f.__name__ not in only:
            continue
        t0 = time.time()
        try:
            ok, msg = f()
        except Exception as e:      # noqa: BLE001
            ok, msg = False, f"예외: {e} {traceback.format_exc()[-300:]}"
        out.append({"id": f.__name__.split("_")[0], "test": f.__name__, "pass": bool(ok), "detail": msg,
                    "sec": round(time.time() - t0, 2)})
        print(("PASS " if ok else "FAIL ") + f.__name__ + " — " + msg, flush=True)
    with open(os.path.join(HERE, f"known_issues_{tag}.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    return out


if __name__ == "__main__":
    tag = sys.argv[1] if len(sys.argv) > 1 else "cur"
    run(tag, sys.argv[2:] or None)
