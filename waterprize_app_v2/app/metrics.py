"""평가지표 6개 (지표 문서 '흙수조 비교실험 측정 지표 체계' 정의) + 지하수 피해 보조지표.

모든 값은 시뮬레이션 기록(engine.result)에서 계산한다.
η 평가 구간: 대조군 유출 농도가 유입의 90%에 도달한 날 이후(그 전에는 질산염이 아직 바닥에
다 도달하지 않아, 반응재가 없어도 η가 높게 나오는 '도달 지연' 착시가 생기므로 제외).
재보충 전·후 구간은 나눠 계산한다(재보충 시각은 분 단위 기록 사용, C4). 대조군 보정 제거율 = (대조군 유출 − 비교군 유출)/유입.

v3 추가:
- η_mass = 1 − ∫Q_out C_out dt / ∫Q_in C_in dt (경계: 지표 유입 ↔ 토양층 바닥면 유출, 평가 구간 전체). 저장량 변화(토양층
  NO3 저장 증가분)도 함께 보고한다(η_mass는 저장량을 빼지 않은 '통과 기준' 값).
- 대조군 돌파 이전 구간의 η를 '초기 η'로 따로 표시(C1). 반응재를 거치지 못하는 구조의 판정 근거(통과 비율)를 같이 낸다.
- 재보충 R: 성공한 재보충만 평균, 성공률은 따로(A3). R_travel 은 피스톤 흐름 근사(C2).
- Q_N: 초기 반응재 질량 기준과 누적 투입(초기+재보충) 질량 기준을 모두(C3).
- r: 셀 중심 거리 ± 격자 반대각선(C5).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from params import MW, REGISTRY

STD_NO3_N = 10.0      # 먹는물 질산성질소 기준 mg/L (한국 먹는물 수질기준)
SO4_LIMIT = 250.0     # 지표 문서: 음용수 기준 초과 경고


def _nanmean(a):
    a = np.asarray(a, float)
    return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")


def _window(D, d0, d1):
    # 재보충 판단은 그날 기록 뒤에 이뤄지므로 재보충 당일 η 는 '재보충 전' 값이다(경계 포함, 부동소수 허용오차 1e-6).
    m = (D["day"] >= d0 - 1e-6) & (D["day"] <= d1 + 1e-6)
    return m


def pH_verdict(dpH):
    if not np.isfinite(dpH):
        return "판정 불가"
    if dpH < -1:
        return "산성화 — 완충 실패(CaCO₃ 부족)"
    if -0.5 < dpH < 0.5:
        return "적정 완충(목표 범위)"
    if dpH > 1:
        return "CaCO₃ 과잉"
    return "경계 범위(주의)"


def K_verdict(p):
    if not np.isfinite(p):
        return "판정 불가"
    if p >= 80:
        return "양호"
    if p >= 50:
        return "부분적 막힘"
    return "심각한 막힘(clogging)"


def SO4_verdict(ratio, so4):
    msg = []
    if np.isfinite(ratio):
        if 0.8 <= ratio <= 1.2:
            msg.append("SLAD 반응이 주 메커니즘")
        elif ratio < 0.5:
            msg.append("이론보다 매우 적음(다른 경로 혼재 또는 석고 침전)")
        else:
            msg.append("이론과 다소 차이")
    if np.isfinite(so4) and so4 > SO4_LIMIT:
        msg.append("유출수 SO₄²⁻ > 250 mg/L: 2차 오염 경고")
    return " / ".join(msg) if msg else "판정 불가"


def compute(res: dict, ctrl: Optional[dict], c, eval_start: Optional[float]) -> Dict:
    D = res["daily"]
    g = res["grid"]
    out: Dict = {"name": res["name"], "is_control": res["is_control"]}
    days = D["day"]
    end = float(days[-1]) if len(days) else 0.0
    e0 = eval_start if eval_start is not None else 0.0
    out["eval_start"] = e0
    # 재보충 시각(분 → 일). 기록 구간 [t0, t1]이 재보충 시작 이전에 끝나면 '재보충 전'
    refill_days = [r.get("t_start_min", r["day"] * 1440.0) / 1440.0 for r in res.get("refills", []) if r.get("ok")]
    first_ref = refill_days[0] if refill_days else end + 1
    t1 = D["t1"] if "t1" in D and len(D["t1"]) == len(days) else days
    out["status"] = res.get("status", "OK")
    # ── 지표 1
    w_pre = (days >= e0 - 1e-6) & (t1 <= first_ref + 1e-6)
    out["eta_pre"] = _nanmean(D["eta"][w_pre]) * 100
    w_init = days < e0 - 1e-6
    out["eta_initial"] = _nanmean(D["eta"][w_init]) * 100 if w_init.any() else float("nan")
    # 누적 질량 기준 η (평가 구간)
    if "N_top_in_g" in D and len(D["N_top_in_g"]) == len(days) and len(days):
        k0 = int(np.searchsorted(days, e0 - 1e-6))
        Nin0 = D["N_top_in_g"][k0 - 1] if k0 > 0 else 0.0
        Nout0 = D["N_to_gw_g"][k0 - 1] if k0 > 0 else 0.0
        Nin = D["N_top_in_g"][-1] - Nin0
        Nout = D["N_to_gw_g"][-1] - Nout0
        out["eta_mass"] = (1 - Nout / Nin) * 100 if Nin > 0 else float("nan")
        out["eta_mass_all"] = (1 - D["N_to_gw_g"][-1] / D["N_top_in_g"][-1]) * 100 if D["N_top_in_g"][-1] > 0 else float("nan")
        out["N_in_g"] = float(Nin); out["N_out_g"] = float(Nout)
    else:
        out["eta_mass"] = out["eta_mass_all"] = float("nan")
    out["eta_all"] = _nanmean(D["eta"][days >= e0 - 1e-6]) * 100
    out["eta_last7"] = _nanmean(D["eta"][days >= end - 7]) * 100
    if ctrl is not None and not res["is_control"]:
        Dc = ctrl["daily"]
        cc = np.interp(days, Dc["day"], Dc["Cout_mgN"])
        m_ = days >= e0 - 1e-6
        out["eta_vs_ctrl"] = _nanmean((cc[m_] - D["Cout_mgN"][m_]) / c.NO3_in_mgN) * 100
    else:
        out["eta_vs_ctrl"] = float("nan")
    segs = []
    bounds = [e0] + [d for d in refill_days if d >= e0] + [end + 1]
    t0 = D["t0"] if "t0" in D and len(D["t0"]) == len(days) else days - 1
    for i in range(len(bounds) - 1):
        # 기록 구간이 [bounds_i, bounds_i+1] 안에 완전히 들어가는 날만(재보충이 걸친 날은 앞 구간)
        m = (t1 <= bounds[i + 1] + 1e-6) & (days >= e0 - 1e-6)
        if i > 0:
            m &= t0 >= bounds[i] - 1e-6
        segs.append({"from": bounds[i], "to": min(bounds[i + 1], end), "eta": _nanmean(D["eta"][m]) * 100})
    out["eta_segments"] = segs
    denit_mg = (D["denit_g"][-1] if len(D["denit_g"]) else 0.0) * 1000
    m0 = res.get("mass0_g") or 0.0
    out["Q_N"] = denit_mg / m0 if m0 > 0 else float("nan")
    # 누적 투입 질량 = 초기 + 재보충으로 들어간 고형분(장부 S_inj 증가분을 질량으로 환산)
    L = res.get("ledger", {}); L0 = res.get("ledger0", {})
    from params import VM, RHO_SOLID
    r_ = c.ratio_S_CaCO3
    dS_inj_mol = (L.get("S_inj", 0.0) - L0.get("S_inj", 0.0)) if L0 else 0.0
    added_g = dS_inj_mol * MW["S"] * (1 + 1.0 / max(r_, 1e-9) * (RHO_SOLID["CaCO3"] / RHO_SOLID["S"]))
    out["mass_added_g"] = float(added_g)
    out["Q_N_cum"] = denit_mg / (m0 + added_g) if (m0 + added_g) > 0 else float("nan")
    out["denit_g"] = denit_mg / 1000
    out["reactive_g"] = m0
    # ── 지표 2
    pH0 = c.soil_pH0
    out["dpH_below"] = _nanmean(D["pH_below"][days >= end - 7]) - pH0
    out["dpH_around"] = _nanmean(D["pH_around"][days >= end - 7]) - pH0
    out["dpH_out"] = _nanmean(D["pH_out"][days >= end - 7]) - pH0
    out["dpH_below_min"] = float(np.nanmin(D["pH_below"]) - pH0) if np.isfinite(D["pH_below"]).any() else float("nan")
    out["pH_verdict"] = pH_verdict(out["dpH_below"]) if not res["is_control"] else "해당 없음(반응재 없음)"
    # 깊이 프로파일(마지막 스냅샷)
    profs = []
    for sn in res["snaps"]:
        act = res["active"]
        reac_cols = res["reac0"].any(axis=2)
        prof = {"op_day": sn["op_day"]}
        for key in ("NO3", "SO4", "pH"):
            arr = np.where(act, sn[key], np.nan)
            prof[key] = np.nanmean(arr.reshape(-1, arr.shape[2]), axis=0) if np.isfinite(arr).any() else np.full(arr.shape[2], np.nan)
            if reac_cols.any():
                sub = arr[reac_cols]
                prof[key + "_reac"] = np.nanmean(sub, axis=0)
        profs.append(prof)
    out["profiles"] = profs
    if res["snaps"]:
        last = res["snaps"][-1]
        reac = res["reac0"]
        if reac.any():
            # 반응재 주변(2셀) ΔpH의 공간 표준편차 = 완충 균질성
            from scipy.ndimage import binary_dilation
            ring = binary_dilation(reac, iterations=2) & ~reac & res["active"]
            vals = last["pH"][ring] - pH0
            out["dpH_spatial_std"] = float(np.nanstd(vals)) if ring.any() else float("nan")
        else:
            out["dpH_spatial_std"] = float("nan")
    # ── 지표 3
    dx, dy, dz = g["dx"], g["dy"], g["dz"]
    reac0 = res["reac0"]
    out["V_reactive_cm3"] = float(reac0.sum() * dx * dy * dz * 1e6)
    supply = None
    roles = res.get("roles_of", {})
    oc = res["obj_cell"]; rc = res["role_cell"]
    from params import ROLES
    if (rc == ROLES.index("주입 개구부")).any():
        supply = np.argwhere(rc == ROLES.index("주입 개구부"))
        out["supply_kind"] = "주입 개구부"
    elif reac0.any():
        supply = np.argwhere(rc == ROLES.index("반응재(고체)"))
        out["supply_kind"] = "반응재 객체(사전 배치)"
    if supply is not None and reac0.any() and len(supply):
        cells = np.argwhere(reac0)
        sx = (supply[:, 0] + 0.5) * dx; sy = (supply[:, 1] + 0.5) * dy
        cx = (cells[:, 0] + 0.5) * dx; cy = (cells[:, 1] + 0.5) * dy
        # 각 반응재 셀에서 가장 가까운 공급 지점까지의 수평 거리
        best = np.full(len(cells), np.inf)
        for s in range(0, len(sx), 256):
            d = np.hypot(cx[:, None] - sx[None, s:s + 256], cy[:, None] - sy[None, s:s + 256])
            best = np.minimum(best, d.min(axis=1))
        half = np.hypot(dx, dy)           # 두 셀 중심 거리의 최대 양자화 오차 = 반대각선 × 2
        out["r_cm"] = float(max(0.0, best.max() - 0.0) * 100)
        out["r_err_cm"] = float(half * 100)
        out["r_cell_note"] = f"셀 중심 거리 — 오차 범위 ±{half * 100:.1f} cm(격자 반대각선 두 개)"
        out["reac_depth_cm"] = (float(cells[:, 2].min() * dz * 100), float((cells[:, 2].max() + 1) * dz * 100))
    else:
        out["r_cm"] = float("nan")
        out["reac_depth_cm"] = (float("nan"), float("nan"))
    if res["snaps"]:
        last = res["snaps"][-1]
        SO4_bg = c.SO4_in_mg
        thr_mg = REGISTRY["SO4_increase_thr"][0] * MW["SO4"]
        soil_zone = np.zeros_like(res["active"])
        soil_zone[:, :, :g["k_interface"]] = True          # 흙수조(토양층)만 셈
        eff = (np.abs(last["pH"] - pH0) < REGISTRY["pH_ok_window"][0]) & (last["SO4"] > SO4_bg + thr_mg) \
            & res["active"] & soil_zone
        out["V_eff_cm3"] = float(eff.sum() * dx * dy * dz * 1e6)
        out["V_eff_mask"] = eff
    else:
        out["V_eff_cm3"] = float("nan")
    # ── 지표 4
    if res.get("Keff0"):
        out["Keff_ret"] = float(D["Keff"][-1] / res["Keff0"] * 100) if len(D["Keff"]) else float("nan")
    else:
        out["Keff_ret"] = float("nan")
    if ctrl is not None and ctrl.get("Keff0"):
        out["Keff_vs_soil"] = float(D["Keff"][-1] / ctrl["Keff0"] * 100) if len(D["Keff"]) else float("nan")
    else:
        out["Keff_vs_soil"] = float("nan")
    if res.get("Kreac0") and np.isfinite(res["Kreac0"]):
        out["Kreac_ret"] = float(D["Kreac"][-1] / res["Kreac0"] * 100)
        out["Kreac_min"] = float(np.nanmin(D["Kreac"]) / res["Kreac0"] * 100)
    else:
        out["Kreac_ret"] = out["Kreac_min"] = float("nan")
    out["Keff_cm_s"] = float(D["Keff"][-1] * 100) if len(D["Keff"]) else float("nan")
    out["K_verdict"] = K_verdict(out["Kreac_ret"] if np.isfinite(out["Kreac_ret"]) else out["Keff_ret"])
    out["K_at_refill"] = []
    for r in res.get("refills", []):
        i = int(np.argmin(np.abs(days - r["day"])))
        out["K_at_refill"].append(float(D["Kreac"][i] / res["Kreac0"] * 100) if res.get("Kreac0") else float("nan"))
    # ── 지표 5
    can_refill = any(v == "주입 개구부" for v in roles.values()) or any(
        r.get("ok") for r in res.get("refills", []))
    out["refill_possible"] = bool(can_refill)
    init_win = _window(D, e0, e0 + 7)
    eta_init = _nanmean(D["eta"][init_win])
    out["eta_init"] = eta_init * 100
    q = c.q_mm_day / 1000.0
    theta_mean = 0.3
    if res["snaps"]:
        th = res["snaps"][0]["theta"]
        theta_mean = float(np.nanmean(np.where(res["active"], th, np.nan)[:, :, :g["k_interface"]]))
    dint = g["k_interface"] * dz
    if reac0.any():
        dc = float((np.argwhere(reac0)[:, 2].mean() + 0.5) * dz)
    else:
        dc = 0.0
    tau = max(0.0, (dint - dc) * theta_mean / q) if q > 0 else 0.0
    out["travel_days"] = tau
    Rs = []
    for r in res.get("refills", []):
        rr = dict(r)
        if r.get("ok") and eta_init > 0:
            tr0 = r.get("t_end_min", r["day"] * 1440.0) / 1440.0

            def eta_at(day):
                i = np.where(days >= day)[0]
                return float(np.nanmean(D["eta"][i[:2]])) if len(i) else float("nan")
            rr["R_24h"] = eta_at(tr0 + 1) / eta_init * 100
            rr["R_travel"] = eta_at(tr0 + tau + 1) / eta_init * 100
        else:
            rr["R_24h"] = rr["R_travel"] = float("nan")
        Rs.append(rr)
    out["refill_table"] = Rs
    ok_R = [x for x in Rs if x.get("ok")]
    out["refill_attempts"] = len(Rs)
    out["refill_success"] = len(ok_R)
    out["refill_success_rate"] = (len(ok_R) / len(Rs) * 100) if Rs else float("nan")
    out["R_note"] = "R_travel은 피스톤 흐름 이동시간 근사(우회·K 감소·재반응 미반영) — 근사값"
    if not can_refill:
        out["R"] = 0.0; out["R_24h"] = 0.0; out["N_max"] = 0; out["T_r_min"] = float("nan")
        out["refill_note"] = "개구부·교체 가능 객체가 없어 재보충 불가 → R = 0 (지표 문서 케이슨형 기준)"
    else:
        Rv = [x["R_travel"] for x in ok_R if np.isfinite(x.get("R_travel", np.nan))]
        out["R"] = float(np.mean(Rv)) if Rv else float("nan")        # 성공한 재보충만 평균(A3)
        R24 = [x["R_24h"] for x in ok_R if np.isfinite(x.get("R_24h", np.nan))]
        out["R_24h"] = float(np.mean(R24)) if R24 else float("nan")
        n = 0
        for x in Rs:
            if x.get("ok") and x.get("R_travel", 0) > 70:
                n += 1
            else:
                break
        out["N_max"] = n
        out["T_r_min"] = float(np.mean([x.get("duration_min", 0) for x in ok_R])) if ok_R else float("nan")
        out["refill_note"] = ("재보충 이벤트가 모사 기간에 일어나지 않음(η가 트리거 아래로 안 떨어짐)" if not Rs
                              else f"모사 기간 내 관측값(N_max는 하한). R은 반응재→유출부 이동시간 {tau:.1f}일 보정 — "
                                   "피스톤 흐름 근사(우회·K 감소·재반응 미반영)")
    # ── 지표 6
    w = days >= e0
    so4 = _nanmean(D["SO4_out_mg"][w])
    cout = _nanmean(D["Cout_mgN"][w])
    out["SO4_out"] = so4
    if ctrl is not None and not res["is_control"]:
        # 실제로 제거된 양 = 대조군 유출 − 비교군 유출 (도달 지연 효과 제거)
        Dc = ctrl["daily"]
        cc = np.interp(days, Dc["day"], Dc["Cout_mgN"])
        sc_ = np.interp(days, Dc["day"], Dc["SO4_out_mg"])
        removed = _nanmean(cc[w] - D["Cout_mgN"][w])
        dSO4 = _nanmean(D["SO4_out_mg"][w] - sc_[w])
        theo = 7.54 * removed
        out["SO4_theory_increase"] = theo
        out["SO4_ratio"] = dSO4 / theo if theo > 0.5 else float("nan")
    else:
        out["SO4_theory_increase"] = float("nan")
        out["SO4_ratio"] = float("nan")
    out["SO4_verdict"] = SO4_verdict(out["SO4_ratio"], so4) if not res["is_control"] else "해당 없음(반응재 없음)"
    out["SO4_max"] = float(np.nanmax(D["SO4_out_mg"])) if np.isfinite(D["SO4_out_mg"]).any() else float("nan")
    # ── 지하수 피해(보조)
    out["N_to_gw_g"] = float(D["N_to_gw_g"][-1]) if len(D["N_to_gw_g"]) else float("nan")
    above = np.where(np.isfinite(D["Cout_mgN"]) & (D["Cout_mgN"] >= STD_NO3_N))[0]
    out["arrival_10mg_day"] = float(days[above[0]]) if len(above) else float("nan")
    out["gw_NO3_end"] = float(D["gw_NO3_mgN"][-1]) if len(D["gw_NO3_mgN"]) else float("nan")
    if ctrl is not None and not res["is_control"]:
        Dc = ctrl["daily"]
        n_ctrl = float(Dc["N_to_gw_g"][-1])
        out["gw_improve_pct"] = (1 - out["N_to_gw_g"] / n_ctrl) * 100 if n_ctrl > 0 else float("nan")
        m = (Dc["day"] >= e0)
        cc = _nanmean(Dc["Cout_mgN"][m])
        out["Cout_improve_pct"] = (1 - cout / cc) * 100 if cc > 0 else float("nan")
    else:
        out["gw_improve_pct"] = float("nan")
        out["Cout_improve_pct"] = float("nan")
    out["Cout_mean"] = cout
    dg = res.get("diag", {}) or {}
    out["through_ratio"] = dg.get("through_ratio", float("nan"))
    out["through_verdict"] = dg.get("through_verdict", "")
    # C1: 반응재를 거치지 못하는 구조는 η가 낮게 나온 '이유'를 함께 표시(평가에서 빼서 좋게 보이게 하지 않음)
    if not res["is_control"]:
        tr_ = out["through_ratio"]
        if np.isfinite(tr_) and tr_ < 0.2:
            out["eval_note"] = (f"반응재 통과 유량이 반응재가 없을 때의 {tr_ * 100:.1f}% — 물이 반응재를 거의 거치지 않음. "
                                "η가 낮은 것은 구조(우회) 때문이며 평가에서 제외하지 않음")
        else:
            out["eval_note"] = ""
    out["chem_engine"] = res.get("chem_engine", "")
    return out


SUMMARY_ROWS = [
    ("계산 상태", "status", "{}"),
    ("지표1 η (재보충 전, %)", "eta_pre", "{:.1f}"),
    ("지표1 초기 η (대조군 돌파 전, %)", "eta_initial", "{:.1f}"),
    ("지표1 η_mass 누적 질량 기준 (평가 구간, %)", "eta_mass", "{:.1f}"),
    ("지표1 η 마지막 7일 (%)", "eta_last7", "{:.1f}"),
    ("지표1 대조군 보정 제거율 (%)", "eta_vs_ctrl", "{:.1f}"),
    ("지표1 Q_N 초기 질량 기준 (mg N/g)", "Q_N", "{:.2f}"),
    ("지표1 Q_N 누적 투입 질량 기준 (mg N/g)", "Q_N_cum", "{:.2f}"),
    ("지표2 ΔpH 반응재 아래 0~10 cm", "dpH_below", "{:+.2f}"),
    ("지표2 ΔpH 반응재 주변부", "dpH_around", "{:+.2f}"),
    ("지표2 ΔpH 바닥 유출부", "dpH_out", "{:+.2f}"),
    ("지표2 판정", "pH_verdict", "{}"),
    ("지표3 확산반경 r (cm)", "r_cm", "{:.1f}"),
    ("지표3 r 오차 범위 ± (cm)", "r_err_cm", "{:.1f}"),
    ("지표3 유효 반응영역 V_eff (cm³)", "V_eff_cm3", "{:.0f}"),
    ("지표4 반응재 영역 K 유지율 (%)", "Kreac_ret", "{:.1f}"),
    ("지표4 도메인 유효 K 유지율 (%)", "Keff_ret", "{:.1f}"),
    ("지표4 판정", "K_verdict", "{}"),
    ("지표5 재보충 성능 회복 R (성공분 평균, %, 근사)", "R", "{:.1f}"),
    ("지표5 재보충 성공률 (%)", "refill_success_rate", "{:.0f}"),
    ("지표5 N_max (회)", "N_max", "{}"),
    ("지표5 T_r (분/회)", "T_r_min", "{:.0f}"),
    ("지표6 유출수 SO₄²⁻ (mg/L)", "SO4_out", "{:.1f}"),
    ("지표6 측정/이론 비", "SO4_ratio", "{:.2f}"),
    ("지표6 판정", "SO4_verdict", "{}"),
    ("보조 지하수 도달 NO₃-N 누적 (g)", "N_to_gw_g", "{:.3f}"),
    ("보조 대조군 대비 지하수 피해 감소 (%)", "gw_improve_pct", "{:.1f}"),
    ("보조 10 mg/L 도달일 (운영 일)", "arrival_10mg_day", "{:.0f}"),
    ("진단 반응재 통과 유량 / 반응재 없을 때 같은 위치 유량", "through_ratio", "{:.3f}"),
    ("진단 통과 판정", "through_verdict", "{}"),
    ("화학 엔진", "chem_engine", "{}"),
    ("반응재 총 질량 (g)", "reactive_g", "{:.0f}"),
]


def fmt(v, f):
    if isinstance(v, str):
        return v
    try:
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "–"
        return f.format(v)
    except Exception:
        return str(v)


def summary_table(metrics_list: List[Dict]):
    rows = []
    for label, key, f in SUMMARY_ROWS:
        rows.append([label] + [fmt(m.get(key), f) for m in metrics_list])
    return ["지표"] + [m["name"] for m in metrics_list], rows
