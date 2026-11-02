"""시뮬레이션 엔진: 설치 타임라인 재현 → 운영(질산염 유입) → 결과 기록.

비교군은 하드코딩하지 않는다. 시나리오(3MF 객체 + 역할표 + 배치 + 설치 단계 표 + 시약 형태)만 다르고
모든 계산은 같은 물리(flow_richards, transport_reaction, phreeqc_engine)를 쓴다.
대조군 = 구조물·반응재 없는 같은 흙수조(scenario=None).

v3 변경 요약(알려진 문제 목록 번호):
- 시간은 _clock() 한 함수로만 진행한다(A9). 설치·개폐·센서·재보충 소요 시간도 흐름·이송을 계산하며 지나간다.
- 수렴하지 않은 흐름 해는 채택하지 않는다. Δt를 줄여 다시 풀고, 한계에 이르면 SimulationFailed → 결과 'FAILED'(A6, 방침 2a).
- 중단 콜백을 설치·주입·흐름 반복 안에서도 확인한다(A1).
- 구조물 삽입은 진행률만큼 깊이 방향으로 단계적으로 들어간다(A9).
- 재보충 트리거는 유입 대비 유출(그날의 η)로만 판정한다(A10). 재보충 시각은 분 단위로 기록(C4).
- 주입 흐름은 기본이 과도 Richards(원천항) 해석(B1). '중첩 근사(빠름)'은 K=Ks·kr(h)로 보정.
- 지화학은 PHREEQC(IPhreeqc) 엔진이 기본(가능할 때). 아니면 내장 화학(이온쌍 포함).
"""
from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


def _limit_malloc_fragmentation():
    """리눅스(glibc): 큰 블록을 mmap으로 받아 해제 즉시 OS에 돌려주게 한다.
    큰 격자에서 희소 분해(SuperLU/ILU)를 반복하면 메모리가 계속 늘어 ×2 격자 계산이 죽었다(약 250 MB/30 s 증가 → 3.4 GB에서 종료).
    임계 128 KB로 같은 계산이 약 380 MB에서 일정해졌지만 보통 배열까지 mmap 되어 전체가 2.2배 느려졌다(5일 대조군 110 s vs 50 s).
    임계 4 MB는 큰 분해 버퍼만 mmap 하고 속도 저하가 없다(36 s). 윈도우·맥에서는 아무것도 하지 않는다."""
    import sys
    import os as _os
    if not sys.platform.startswith("linux") or _os.environ.get("WP_NO_MALLOPT"):
        return
    thr = int(_os.environ.get("WP_MMAP_THRESHOLD", str(4 << 20)))     # 4 MB: 이보다 큰 블록(희소 분해 등)만 mmap
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.mallopt(-1, thr)    # M_TRIM_THRESHOLD
        libc.mallopt(-3, thr)    # M_MMAP_THRESHOLD
    except Exception:      # noqa: BLE001
        pass


_limit_malloc_fragmentation()

import install_events as ie
from flow_richards import (DIRECT_FIRST_MAX, FlowBC, FlowState, LinearSolveError, RichardsFV, StopRequested, new_stats, vg_h_from_Se,
                           vg_kr, vg_theta)
from params import (D50_TEXTURE, MW, REGISTRY, RHO_SOLID, ROLE_DEFAULTS, ROLES, SOIL_TEXTURES, VM, Common)
from transport_reaction import (LEDGER_KEYS, PHI_MIN_DEP, ChemState, ReactParams, Transport, alk_from_pH,
                                deposit_particles, kozeny_carman_ratio, pH_of, react)
from voxel import Grid, Placement, compose, make_grid, voxelize_objects

ENGINE_VERSION = "v3.0"


class SimulationFailed(RuntimeError):
    pass


REAGENT_FORMS = ["(a) 고체 사전 배치", "(b) 고체 카트리지 교체", "(c) 입자 슬러리(포획)",
                 "(d) 액체-용존 전자공여체", "(e) 액체-콜로이드·나노 현탁"]
LIQUID_MATERIALS = ["선택 안 함", "티오황산염(S2O3²⁻)", "콜로이드 황", "사용자 정의"]
# 티오황산염 독립영양 탈질 양론(문헌, 참고): 5 S2O3²⁻ + 8 NO3⁻ + H2O → 10 SO4²⁻ + 4 N2 + 2 H⁺
# (예: Cardoso et al. 2006 Biotechnol. Bioeng.; 원 논문 대조 필요). 속도 상수는 미정 → 사용자 입력 필수.
THIOSULFATE_STOICH = {"yD": 0.625, "ySO4": 1.25, "yH": 0.25, "sD": 2.0}


@dataclass
class Scenario:
    name: str
    mechanism: str = ""
    install_text: str = ""
    file_name: str = ""
    unit: str = "millimeter"
    objects: list = field(default_factory=list)          # io_3mf.MeshObject
    roles: Dict[str, dict] = field(default_factory=dict)
    placement: Placement = field(default_factory=Placement)
    events: List[dict] = field(default_factory=list)
    allow_refill_wo_opening: bool = False
    mass_correction: bool = True
    color: str = "#1f77b4"
    reagent: Dict = field(default_factory=dict)          # 시약 형태와 액체 시약 정의
    file_hash: str = ""


def reagent_form(sc: Optional[Scenario]) -> str:
    if sc is None:
        return ""
    return (sc.reagent or {}).get("form") or ""


def validate_reagent(sc: Scenario) -> List[str]:
    """시약 형태 검사: 미정 양론·속도는 계산을 막는다(지어내지 않음)."""
    errs = []
    rg = sc.reagent or {}
    form = rg.get("form")
    if form not in REAGENT_FORMS:
        errs.append("시약 형태를 선택하세요(필수).")
        return errs
    row_forms = {str(r.get("시약 형태") or "") for r in ie.clean_rows(sc.events) if r.get("시약 형태")}
    bad = [f for f in row_forms if f not in REAGENT_FORMS]
    if bad:
        errs.append(f"설치 단계 표의 시약 형태 {bad}를 알 수 없습니다.")
    if form.startswith("(d)") or any(f.startswith("(d)") for f in row_forms):
        mat = rg.get("material", "선택 안 함")
        if mat in (None, "", "선택 안 함"):
            errs.append("액체 시약 물질을 선택하지 않아 계산할 수 없습니다(티오황산염/콜로이드 황/사용자 정의).")
        need = ["yD", "ySO4", "yH", "sD", "k", "KN", "KD", "conc_molL", "Rf"]
        miss = [k for k in need if rg.get(k) in (None, "", "미정") or (isinstance(rg.get(k), float) and rg.get(k) != rg.get(k))]
        if miss:
            errs.append("액체 공여체 정의가 '미정'인 항목이 있어 계산할 수 없습니다: " + ", ".join(miss) +
                        " (양론·속도는 지어내지 않습니다. 문헌 값이나 실험값을 넣으세요).")
    if form.startswith("(e)") or any(f.startswith("(e)") for f in row_forms):
        for k in ("lambda_colloid", "d_p"):
            if rg.get(k) in (None, "", "미정"):
                errs.append(f"콜로이드 시약의 {k}가 미정입니다.")
    return errs


# ───────────────────────────── 토양 재료 ─────────────────────────────────
def soil_arrays(c: Common, g: Grid):
    nx, ny, nz = g.shape
    d = g.dc
    kint = g.k_interface
    tex = dict(SOIL_TEXTURES[c.texture])
    if c.Ks_override:
        tex["Ks"] = float(c.Ks_override)
    aq = SOIL_TEXTURES[c.aq_texture]
    P = {k: np.zeros(nz) for k in ("Ks", "thr", "ths", "alpha", "n", "aniso")}
    for k in range(nz):
        if k >= kint:
            src, an = aq, 1.0
        else:
            src, an = tex, c.aniso
            if c.soil_mode == "상세" and c.layers:
                for L in c.layers:
                    if L.top <= d[k] < L.bottom:
                        src = {"Ks": L.Ks, "theta_r": L.theta_r, "theta_s": L.theta_s,
                               "alpha": L.alpha, "n": L.n}
                        an = L.aniso
                        break
        P["Ks"][k] = src["Ks"]; P["thr"][k] = src["theta_r"]; P["ths"][k] = src["theta_s"]
        P["alpha"][k] = src["alpha"]; P["n"][k] = src["n"]; P["aniso"][k] = an
    full = {k: np.broadcast_to(v[None, None, :], g.shape).ravel().copy() for k, v in P.items()}
    full["Ks_h"] = full["Ks"]
    full["Ks_v"] = full["Ks"] * full["aniso"]
    full["zone"] = np.broadcast_to((np.arange(nz) >= kint)[None, None, :], g.shape).ravel().copy()
    het = getattr(c, "heterogeneity", None)
    if het and het.get("sigma_lnK", 0) > 0:
        rng = np.random.default_rng(int(het.get("seed", 0)))
        z = rng.standard_normal(g.shape)
        Lc = max(1, int(round(float(het.get("corr_len", 0.05)) / max(g.dx, 1e-9))))
        from scipy.ndimage import uniform_filter
        z = uniform_filter(z, size=Lc, mode="nearest")
        z = (z - z.mean()) / max(z.std(), 1e-12)
        f = np.exp(float(het["sigma_lnK"]) * z).ravel()
        soil_only = ~full["zone"]
        full["Ks_h"] = np.where(soil_only, full["Ks_h"] * f, full["Ks_h"])
        full["Ks_v"] = np.where(soil_only, full["Ks_v"] * f, full["Ks_v"])
    return full


def reactive_mass_g(S, Cal, V):
    return float((S * MW["S"] + Cal * MW["CaCO3"]).sum() * V)


# ───────────────────────────── 형상 준비 ─────────────────────────────────
def build_geometry(sc: Scenario, c: Common, g: Grid):
    ov, warns = voxelize_objects(sc.objects, sc.roles, sc.placement, g)
    comp = compose(ov, sc.roles, g)
    errs = []
    roles_of = {o.name: sc.roles.get(o.name, {}).get("role", "불투수") for o in sc.objects}
    n_reac = n_open = 0
    for o in sc.objects:
        role = roles_of[o.name]
        cnt = comp["counts"].get(o.name, 0)
        if role in ("반응재(고체)", "주입 개구부"):
            if not o.closed:
                errs.append(f"'{o.name}'({role})이 닫히지 않은 메시입니다(열린/비다양체 모서리 {o.open_edges}개). "
                            "반응재·개구부는 체적 판정이 필요하므로 오류로 멈춥니다(A13). 메시를 고쳐 다시 넣으세요.")
            if cnt == 0:
                errs.append(f"'{o.name}'({role})의 복셀이 0개입니다. 배치·단위·해상도를 확인하세요. "
                            "샘플로 대체하지 않고 실행을 멈춥니다.")
            if role == "반응재(고체)":
                n_reac += cnt
            else:
                n_open += cnt
    if n_reac == 0 and n_open == 0 and not errs:
        errs.append("반응재(고체) 또는 주입 개구부 역할 객체가 하나도 없습니다. 역할표를 확인하세요.")
    return {"ov": ov, "comp": comp, "warnings": warns, "errors": errs, "roles_of": roles_of}


def planned_masses(sc: Scenario, c: Common, g: Grid, geom) -> Dict[str, float]:
    """사전 배치 반응재 질량(메시 체적 기준·복셀 기준)과 설치 단계 주입 계획 질량(g), 황 당량(mol S)."""
    r = c.ratio_S_CaCO3
    rho_mix = (r * RHO_SOLID["S"] + RHO_SOLID["CaCO3"]) / (r + 1) * 1000.0   # g/m3 고체
    fS_vol = r / (r + 1)
    pre_mesh = pre_vox = 0.0
    S_mol = 0.0
    for o in sc.objects:
        rr = sc.roles.get(o.name, {})
        if rr.get("role") != "반응재(고체)":
            continue
        phi = float(rr.get("phi") or ROLE_DEFAULTS["반응재(고체)"]["phi"])
        vm = o.volume_m3 * sc.placement.scale ** 3 * (1 - phi)
        pre_mesh += vm * rho_mix
        pre_vox += geom["comp"]["counts"].get(o.name, 0) * g.vcell * (1 - phi) * rho_mix
        S_mol += vm * fS_vol / VM["S"]
    inj = 0.0
    inj_L = 0.0
    form = reagent_form(sc)
    rg = sc.reagent or {}
    for row in ie.clean_rows(sc.events):
        if row["이벤트"] == "시약 주입" and row["부피(L)"]:
            inj_L += row["부피(L)"]
            fs = row["고형분비(v/v)"] if row["고형분비(v/v)"] is not None else 0.2
            if form.startswith("(d)"):
                cm = float(rg.get("conc_molL") or 0.0)
                S_mol += row["부피(L)"] * cm * float(rg.get("sD") or 0.0)
            elif form.startswith("(e)"):
                vs = row["부피(L)"] * 1e-3 * fs
                inj += vs * RHO_SOLID["S"] * 1000.0
                S_mol += vs / VM["S"]
            else:
                inj += row["부피(L)"] * 1e-3 * fs * rho_mix
                S_mol += row["부피(L)"] * 1e-3 * fs * fS_vol / VM["S"]
    return {"pre_mesh_g": pre_mesh, "pre_voxel_g": pre_vox, "injection_g": inj,
            "total_g": (pre_mesh if sc.mass_correction else pre_vox) + inj,
            "S_equiv_mol": S_mol, "inject_volume_L": inj_L}


# ───────────────────────────── 시뮬레이터 ───────────────────────────────
class Sim:
    def __init__(self, c: Common, sc: Optional[Scenario], progress=None, stop=None, eval_start_day=None):
        self.c = c
        self.sc = sc
        self.g = make_grid(c)
        self.progress = progress or (lambda f, m: None)
        self.stop = stop or (lambda: False)
        self.eval_start_day = eval_start_day          # 지표 계산용(재보충 트리거에는 쓰지 않음, A10)
        self.warn: List[str] = []
        self.t = 0.0                       # 일 — _clock()으로만 바꾼다
        self.t_op0 = None
        self.status = "OK"
        self.fail_reason = ""
        self.solver = new_stats()
        self.solver.update({"transport_steps": 0, "chem_calls": 0, "phreeqc_calls": 0, "flow_rejected": 0,
                            "flow_accepted": 0, "dt_min_s": np.inf, "dt_max_s": 0.0, "steady_solves": 0,
                            "steady_fail": 0})
        self.flow_log: List[dict] = []
        g = self.g
        N = g.N
        self.soil = soil_arrays(c, g)
        self.V = g.vcell
        self.Ks_ref_h = self.soil["Ks_h"].copy()
        self.Ks_ref_v = self.soil["Ks_v"].copy()
        self.thr = self.soil["thr"].copy()
        self.alpha = self.soil["alpha"].copy()
        self.n = self.soil["n"].copy()
        self.phi_ref = self.soil["ths"].copy()
        self.phi_ref0 = self.phi_ref.copy()
        self.disturb = np.ones(N)
        self.active = np.ones(N, bool)
        self.role_cell = np.full(N, -1, np.int8)
        self.obj_cell = np.full(N, -1, np.int16)
        nx, ny, nz = g.shape
        self.barrier = (np.zeros((nx - 1, ny, nz)), np.zeros((nx, ny - 1, nz)),
                        np.zeros((nx, ny, nz - 1)), np.zeros((nx, ny)))
        self.inserted = set()
        self.insert_frac: Dict[str, float] = {}
        self.open_state: Dict[str, bool] = {}
        self.cake_m3 = 0.0
        # 화학 초기
        p = c.p
        self.I_bg = p("I_bg")
        self.zc = (g.H - np.broadcast_to(g.dc[None, None, :], g.shape)).ravel()
        self.pCO2_cells = self._pCO2_field()
        self.pCO2 = p("pCO2")
        alk_soil = alk_from_pH(c.soil_pH0, self.pCO2_cells, self.I_bg)
        self.alk_in = c.alk_in_mgCaCO3 / 50.04          # meq/L = eq/m3
        zone = self.soil["zone"]
        cdict = {"NO3": np.where(zone, c.gw_NO3_mgN, c.soil_NO3_mgN) / MW["N"],
                 "SO4": np.full(N, c.SO4_in_mg / MW["SO4"]),
                 "Ca": np.full(N, c.Ca_in_mg / MW["Ca"]),
                 "ALK": np.where(zone, self.alk_in, alk_soil),
                 "P": np.zeros(N)}
        form = reagent_form(sc)
        self.form = form
        forms = {form}
        if sc is not None:
            forms |= {str(r.get("시약 형태") or "") for r in ie.clean_rows(sc.events)}
        if any(f.startswith("(e)") for f in forms):
            cdict["PC"] = np.zeros(N)
        if any(f.startswith("(d)") for f in forms):
            cdict["D"] = np.zeros(N)
        if c.o2_stage:
            cdict["O2"] = np.full(N, p("DO_in") / 32.0)            # mol/m3
        self.cs = ChemState(c=cdict, S=np.zeros(N), Cal=np.zeros(N), Gyp=np.zeros(N), Bio=np.zeros(N),
                            Inert=np.zeros(N), aCal=np.full(N, 6.0 / p("cal_d_p")),
                            phi_matrix=self.soil["ths"].copy(), ledger={k: 0.0 for k in LEDGER_KEYS},
                            phi_min=np.full(N, PHI_MIN_DEP))
        donor = None
        rg = (sc.reagent if sc is not None else {}) or {}
        if "D" in cdict:
            donor = {k: float(rg[k]) for k in ("yD", "ySO4", "yH", "sD", "k", "KN", "KD")}
        bio_vol = 0.08 * 113e-3 / max(p("bio_density"), 1e-9)
        self.rp = ReactParams(k_half=p("k_half"), S_ref=p("S_ref_frac"), pH_low=p("pH_low"),
                              pH_high=p("pH_high"), theta_T=p("theta_T"), T=c.temperature, T_ref=p("T_ref"),
                              stoich_SO4=p("stoich_SO4"), stoich_H=p("stoich_H"), stoich_S=p("stoich_S"),
                              pCO2=self.pCO2_cells, I_bg=self.I_bg, bio_vol=bio_vol,
                              bio_decay=p("bio_decay"), calcite_mode=c.calcite_mode,
                              lambda_filt=p("lambda_filt"), ratio=c.ratio_S_CaCO3, bio_det=p("bio_det"),
                              o2_on=bool(c.o2_stage), k_O2=p("k_O2"), K_O2=p("K_O2"), K_I_O2=p("K_I_O2"),
                              donor=donor, chem=c.chem_mode, k1_decay=c.k1_decay_day / 86400.0)
        self.retard = {"D": float(rg.get("Rf") or 1.0)} if "D" in cdict else None
        # 화학 엔진
        self.chem_engine = "내장"
        self.phreeqc = None
        if c.chem_engine == "PHREEQC" and c.chem_mode == "SLAD":
            import phreeqc_engine as PE
            self.phreeqc = PE.get_engine()
            if self.phreeqc is not None:
                self.chem_engine = "PHREEQC"
            else:
                self.warn.append("PHREEQC 엔진을 찾지 못해 내장 화학(이온쌍 포함)으로 계산했습니다: "
                                 + PE.status().get("error", "")[:200])
        self._chem_acc = 0.0
        # 수리 경계
        self.wt_depth = c.wt_depth if c.wt_depth is not None else g.interface_depth
        self.z_wt = g.H - self.wt_depth
        self.k_wt = int(np.clip(np.floor(self.wt_depth / g.dz + 0.5), 0, nz - 1))
        Ks_aq = SOIL_TEXTURES[c.aq_texture]["Ks"]
        self.gw_grad = (c.gw_velocity / 86400.0) / Ks_aq if (c.H_aq > 0 and c.gw_velocity) else 0.0
        self.bottom = c.bottom_bc
        self.leak_R = (c.leak_thick / max(c.leak_K, 1e-30)) if c.bottom_bc.startswith("누수") else 0.0
        # 초기 수두
        Se0 = np.clip((c.init_saturation * self.soil["ths"] - self.thr) / (self.soil["ths"] - self.thr), 1e-3, 1.0)
        h = vg_h_from_Se(Se0, self.alpha, self.n)
        if self.bottom != "자유배수":
            h = np.where(self.zc < self.z_wt, self.z_wt - self.zc, h)
        self.h = h.reshape(g.shape)
        self.topstate = None
        if c.o2_stage:
            th0 = vg_theta(self.h.ravel(), self.thr, self.soil["ths"], self.alpha, self.n)
            air = np.maximum(self.soil["ths"] - th0, 0.0)
            self.cs.O2g = air * 0.21 / (8.206e-5 * (c.temperature + 273.15))       # 대기 21% O2
        # 기록
        self.frames: List[dict] = []
        self.snaps: List[dict] = []
        self.daily: Dict[str, list] = {k: [] for k in (
            "day", "t0", "t1", "eta", "Cout_mgN", "SO4_out_mg", "pH_out", "pH_around", "pH_below", "pH_min",
            "Keff", "Kreac", "N_to_gw_g", "gw_NO3_mgN", "denit_g", "water_out_L", "water_in_L", "N_top_in_g",
            "reactive_g", "q_mm_day")}
        self.events_log: List[dict] = []
        self.refills: List[dict] = []
        self.flow_solves = 0
        self.geom = None
        self.objects_by_name = {}
        if sc is not None:
            self.geom = build_geometry(sc, c, self.g)
            self.warn += self.geom["warnings"]
            self.objects_by_name = {o.name: i for i, o in enumerate(sc.objects)}
            for o in sc.objects:
                if self.geom["roles_of"][o.name] == "주입 개구부":
                    self.open_state[o.name] = True
        self.fv = None
        self.tr = None
        self.st = None

    # ── 공통 도우미 ───────────────────────────────────────────────────
    def _check_stop(self):
        if self.stop():
            raise StopRequested()

    def _clock(self, dt_s: float):
        """시뮬레이션 시계를 진행하는 유일한 함수(A9)."""
        self.t += dt_s / 86400.0

    def _pCO2_field(self):
        c = self.c
        if c.pCO2_mode != "깊이별":
            return np.full(self.g.N, c.p("pCO2"))
        depth = self.g.H - self.zc
        f = depth / max(self.g.H, 1e-9)
        return c.p("pCO2") + (c.p("pCO2_bottom") - c.p("pCO2")) * f

    def q_at(self, op_day: float) -> float:
        """운영 시작 후 op_day 일의 지표 순 플럭스 m/s(강우·관개 − 잠재 증발산)."""
        c = self.c
        q = c.q_mm_day
        if c.rain_series:
            rs = sorted(c.rain_series, key=lambda r: r[0])
            for d0, mm in rs:
                if op_day + 1e-9 >= d0:
                    q = mm
        return (q - c.et_mm_day) / 1000.0 / 86400.0

    def _next_q_change(self, op_day: float) -> float:
        if not self.c.rain_series:
            return np.inf
        nxt = [r[0] for r in self.c.rain_series if r[0] > op_day + 1e-9]
        return min(nxt) if nxt else np.inf

    # ── 구조물 적용 ───────────────────────────────────────────────────
    def _partial_member(self, v, frac):
        """삽입 진행률 frac: 객체의 가장 깊은 부분부터가 아니라 위(지표)에서부터 frac×깊이 범위만 존재."""
        if frac >= 1.0:
            return v.member, v.faces
        g = self.g
        k_has = np.where(v.member.any(axis=(0, 1)))[0]
        if len(k_has) == 0:
            return v.member, v.faces
        kmax = k_has.max()
        kcut = int(np.floor(frac * (kmax + 1)))
        m = v.member.copy()
        m[:, :, kcut:] = False
        faces = v.faces
        if faces is not None:
            bx, by, bz, bt = [f.copy() for f in faces]
            bx[:, :, kcut:] = False; by[:, :, kcut:] = False; bz[:, :, max(kcut - 1, 0):] = False
            if kcut == 0:
                bt[:] = False
            faces = (bx, by, bz, bt)
        return m, faces

    def _apply_structure(self):
        """현재 꽂혀 있는 객체(삽입 진행률 포함)로 셀 역할/장벽/재료 갱신."""
        import copy as _copy
        g = self.g
        sc = self.sc
        ov = []
        for v in self.geom["ov"]:
            if v.name not in self.inserted:
                continue
            fr = self.insert_frac.get(v.name, 1.0)
            if fr < 1.0:
                v2 = _copy.copy(v)
                v2.member, v2.faces = self._partial_member(v, fr)
                ov.append(v2)
            else:
                ov.append(v)
        comp = compose(ov, sc.roles, g) if ov else None
        N = g.N
        role = np.full(N, -1, np.int8)
        obj = np.full(N, -1, np.int16)
        nx, ny, nz = g.shape
        barrier = (np.zeros((nx - 1, ny, nz)), np.zeros((nx, ny - 1, nz)),
                   np.zeros((nx, ny, nz - 1)), np.zeros((nx, ny)))
        if comp is not None:
            role = comp["role_grid"].ravel()
            o2 = comp["obj_grid"].ravel()
            idx_map = np.array([self.objects_by_name[v.name] for v in ov] + [-1], dtype=np.int16)
            obj = np.where(o2 >= 0, idx_map[np.maximum(o2, 0)], -1).astype(np.int16)
            barrier = comp["barrier"]
        newly = (role >= 0) & (self.role_cell < 0)
        removed = (role < 0) & (self.role_cell >= 0)
        p = self.c
        rmix = p.ratio_S_CaCO3
        for ri, rname in enumerate(ROLES):
            m = newly & (role == ri)
            if not m.any() or rname == "불투수":
                continue
            for oid in np.unique(obj[m]):
                name = sc.objects[oid].name
                rr = sc.roles.get(name, {})
                cells = np.where(m & (obj == oid))[0]
                d = ROLE_DEFAULTS[rname]
                K = rr.get("K", d["K"])
                phi = rr.get("phi", d["phi"])
                al = rr.get("alpha", d["alpha"])
                nn = rr.get("n", d["n"])
                if rname == "토양" and (K in (None, "") or K != K):
                    continue
                if rname == "주입 개구부" and (K in (None, "") or K != K):
                    K = p.p("K_open")
                if rname == "반응재(고체)" and p.phi_reac_override is not None:
                    phi = p.phi_reac_override
                K = float(K if K not in (None, "") else (d["K"] or 1e-5))
                phi = float(phi if phi not in (None, "") and phi == phi else (d["phi"] or 0.4))
                self.Ks_ref_h[cells] = K
                self.Ks_ref_v[cells] = K
                self.alpha[cells] = float(al if al not in (None, "") and al == al else 14.5)
                self.n[cells] = float(nn if nn not in (None, "") and nn == nn else 2.68)
                self.thr[cells] = min(0.03, 0.1 * phi)
                if rname == "반응재(고체)" and rr.get("reactive", True):
                    fsol = 1.0 - phi
                    corr = 1.0
                    if sc.mass_correction:
                        nvox = int((self.geom["comp"]["obj_grid"].ravel() == oid).sum())
                        if nvox > 0:
                            corr = sc.objects[oid].volume_m3 * sc.placement.scale ** 3 / (nvox * g.vcell)
                            corr = float(np.clip(corr, 0.2, (1 - 0.05) / max(fsol, 1e-6)))
                    vs = fsol * corr
                    self.cs.phi_matrix[cells] = 1.0
                    self.cs.S[cells] = vs * rmix / (rmix + 1) / VM["S"]
                    self.cs.Cal[cells] = vs / (rmix + 1) / VM["CaCO3"]
                    self.cs.aCal[cells] = 6.0 / self.c.p("cal_d_p")
                    self.phi_ref[cells] = 1.0 - vs
                    self.phi_ref0[cells] = 1.0 - vs
                else:
                    self.cs.phi_matrix[cells] = phi
                    self.phi_ref[cells] = phi
                    self.phi_ref0[cells] = phi
                if rname in ("주입 개구부", "관 내부(빈 공간)"):
                    # 빈 공간에 쌓이는 입자는 충전층(필터 케이크)을 이룸: 포획 하한 = 충전 공극률
                    self.cs.phi_min[cells] = min(self.c.packed_porosity, phi)
        if removed.any():
            m = removed
            self.Ks_ref_h[m] = self.soil["Ks_h"][m]; self.Ks_ref_v[m] = self.soil["Ks_v"][m]
            self.alpha[m] = self.soil["alpha"][m]; self.n[m] = self.soil["n"][m]; self.thr[m] = self.soil["thr"][m]
            self.cs.phi_matrix[m] = self.soil["ths"][m]; self.phi_ref[m] = self.soil["ths"][m]
            self.phi_ref0[m] = self.soil["ths"][m]
            self.cs.S[m] = 0; self.cs.Cal[m] = 0; self.cs.Gyp[m] = 0; self.cs.Bio[m] = 0
            self.cs.phi_min[m] = PHI_MIN_DEP
        new_active = role != ROLES.index("불투수")
        lost = self.active & ~new_active
        if lost.any():
            for k in self.cs.c:
                self.cs.c[k][lost] = 0.0
        self.active = new_active
        self.role_cell = role
        self.obj_cell = obj
        self.barrier = barrier
        if self.c.disturb_thick > 0 and self.c.disturb_Kmult != 1.0 and newly.any():
            nd = int(np.ceil(self.c.disturb_thick / min(g.dx, g.dy, g.dz)))
            m = (role >= 0).reshape(g.shape)
            dil = m.copy()
            for _ in range(nd):
                d2 = dil.copy()
                d2[1:] |= dil[:-1]; d2[:-1] |= dil[1:]
                d2[:, 1:] |= dil[:, :-1]; d2[:, :-1] |= dil[:, 1:]
                d2[:, :, 1:] |= dil[:, :, :-1]; d2[:, :, :-1] |= dil[:, :, 1:]
                dil = d2
            ring = (dil & ~m).ravel() & (self.role_cell < 0)
            self.disturb[ring] = self.c.disturb_Kmult
        self._rebuild_models()

    def _material(self):
        phi = self.cs.porosity()
        ratio = kozeny_carman_ratio(phi, self.phi_ref)
        Ksh = self.Ks_ref_h * ratio * self.disturb
        Ksv = self.Ks_ref_v * ratio * self.disturb
        # 개구부·관 내부(빈 공간)에 입자가 쌓이면 '입자 충전층' K(Kozeny–Carman, 입경 d_p)로 바뀐다.
        # 빈 공간을 기준으로 한 공극비 보정은 의미가 없으므로 min(K_빈공간, K_충전층(d_p, φ))를 쓴다.
        vo = self.cs.phi_min > PHI_MIN_DEP + 1e-12
        if vo.any():
            d = self.c.p("inj_d_p")
            pv = np.clip(phi[vo], 1e-3, 0.999)
            Kkc = 9.81e6 * d * d * pv ** 3 / (180.0 * (1 - pv) ** 2)        # ρg/μ = 9.81e6 1/(m·s)
            Ksh[vo] = np.minimum(self.Ks_ref_h[vo], Kkc) * self.disturb[vo]
            Ksv[vo] = np.minimum(self.Ks_ref_v[vo], Kkc) * self.disturb[vo]
        ths = np.maximum(phi, self.thr + 0.01)
        return Ksh, Ksv, ths

    def _collect_stats(self):
        if self.fv is not None:
            for k, v in self.fv.stats.items():
                if k == "lin_max_relres":
                    self.solver[k] = max(self.solver[k], v)
                else:
                    self.solver[k] += v
            self.fv.stats = new_stats()

    def _rebuild_models(self):
        self._collect_stats()
        Ksh, Ksv, ths = self._material()
        self.fv = RichardsFV(self.g, self.active.reshape(self.g.shape), Ksh, Ksv, self.thr, ths,
                             self.alpha, self.n, self.barrier)
        self.fv.stop = self.stop
        p = self.c
        self.tr = Transport(self.g, self.fv, dict(alpha_L=p.p("alpha_L"), alpha_T_ratio=p.p("alpha_T_ratio"),
                                                  Dm=p.p("Dm"), phi_cells=ths))
        self.st = None

    def _bc(self, q, src=None, fixed=None):
        return FlowBC(q_top=q, bottom=self.bottom, z_wt=self.z_wt, gw_grad=self.gw_grad,
                      wt_k=self.k_wt, src=src, fixed_h=fixed, leak_R=self.leak_R, h_crit=self.c.h_crit)

    def _set_flow(self, st):
        """흐름장 교체(수분량 변화 시 질량 보존: c ← c·θ_old/θ_new). 저장량 변화는 W_swap 장부에."""
        th_new = st.theta.ravel()
        th_old = self.tr.theta if self.tr.st is not None else getattr(self, "_theta_prev", None)
        if th_old is not None and th_old.shape == th_new.shape:
            ok = th_new > 0
            for k in self.cs.c:
                self.cs.c[k] = np.where(ok, self.cs.c[k] * th_old / np.where(ok, th_new, 1.0), 0.0)
            self.cs.ledger["W_swap"] += float(((th_new - th_old) * self.V).sum())
        self.tr.p["phi_cells"] = self._material()[2]
        self.tr.set_flow(st)
        self.st = st
        self._theta_prev = th_new.copy()
        self.h = st.h
        if st.topstate is not None:
            self.topstate = st.topstate

    def _static_flow(self):
        """설치 단계(관개 전) 배경 흐름: 정지 상태(유량 0), 초기 수분 유지 — 가정."""
        g = self.g
        nx, ny, nz = g.shape
        Ksh, Ksv, ths = self._material()
        th = vg_theta(self.h.ravel(), self.thr, ths, self.alpha, self.n)
        th = np.where(self.active, th, 0.0)
        z = lambda *s_: np.zeros(s_)
        return FlowState(h=self.h.copy(), theta=th.reshape(g.shape), Fx=z(nx - 1, ny, nz), Fy=z(nx, ny - 1, nz),
                         Fz=z(nx, ny, nz - 1), Ftop=z(nx, ny), Fbot=z(nx, ny), Fleft=z(ny, nz), Fright=z(ny, nz),
                         Fsrc=z(nx, ny, nz), ponded=np.zeros((nx, ny), bool), info={"static": True})

    def _steady(self, q, src=None, fixed=None):
        bc = self._bc(q, src, fixed)
        self.fv.stop = self.stop
        st = self.fv.solve_steady(None, bc)
        self.flow_solves += 1
        self.solver["steady_solves"] += 1
        if self.c.redistribute_runoff and np.max(q) > 0 and st.info.get("final_conv"):
            for _ in range(3):
                tot = float(np.mean(q)) * self.g.Lx * self.g.Ly
                if st.runoff <= 1e-4 * tot:
                    break
                A = self.g.dx * self.g.dy
                open_ = self.fv.top_ok & ~st.ponded.ravel()
                if not open_.any():
                    break
                qarr = np.zeros(len(self.fv.top_cells))
                qarr[open_] = tot / (open_.sum() * A)
                bc = self._bc(qarr, src, fixed)
                st = self.fv.solve_steady(None, bc)
                self.flow_solves += 1
                self.solver["steady_solves"] += 1
        if not st.info.get("final_conv", False):
            self.solver["steady_fail"] += 1
            raise SimulationFailed(f"{self.t:.3f}일: 흐름 정상해가 수렴하지 않았습니다 "
                                   f"({getattr(self.fv, 'last_error', '')[:200]}).")
        return st

    def _flow_step(self, q, dt_target, src=None, fixed=None):
        """과도 Richards 한 단계(채택된 Δt ≤ dt_target). 수렴 실패 해는 버리고 Δt를 줄인다(A6).
        Δt가 flow_dt_min_s 아래로 가도 실패하면 SimulationFailed."""
        bc = self._bc(q, src, fixed)
        dt = min(dt_target, getattr(self, "_dt_flow", 60.0))
        dtmin = max(float(self.c.flow_dt_min_s), 1e-3)
        inj = src is not None or fixed is not None
        mi, mt = (30, 90) if inj else (24, 80)        # 주입(원천항)은 비선형성이 커서 반복을 더 허용
        if inj:
            dtmin = min(dtmin, 0.05)
        while True:
            self._check_stop()
            try:
                hn, conv, it, pn = self.fv.step(self.h, dt, bc, self.topstate, max_iter=mi, max_total=mt, tol=1e-6)
            except LinearSolveError as e:
                conv, it, err = False, 99, str(e)
            self.flow_solves += 1
            if conv:
                self.solver["flow_accepted"] += 1
                self.solver["dt_min_s"] = min(self.solver["dt_min_s"], dt)
                self.solver["dt_max_s"] = max(self.solver["dt_max_s"], dt)
                st = self.fv.fluxes(hn, bc, pn)
                self._dt_flow = dt * (1.5 if it <= 7 else (1.0 if it <= 12 else 0.6))
                if inj:
                    self._dt_flow = min(self._dt_flow, 120.0)
                return st, dt
            self.solver["flow_rejected"] += 1
            self.solver["dt_cuts"] += 1
            dt *= 0.3
            if dt < dtmin:
                raise SimulationFailed(f"{self.t:.4f}일: 과도 흐름이 Δt={dt / 0.3:.2g}초에서도 수렴하지 않아 중단했습니다"
                                       f"(수렴 실패 해는 채택하지 않음).")

    # ── 이송·반응 한 구간 ────────────────────────────────────────────
    def _particle_species(self):
        out = []
        if np.any(self.cs.c["P"] > 0):
            out.append(("P", self.rp.lambda_filt, None, self.c.p("inj_d_p")))
        if "PC" in self.cs.c and np.any(self.cs.c["PC"] > 0):
            rg = self.sc.reagent if self.sc else {}
            out.append(("PC", float(rg.get("lambda_colloid") or self.c.p("lambda_colloid")), 1.0,
                        float(rg.get("d_p") or self.c.p("colloid_d_p"))))
        return out

    def _transport(self, dt, bc_conc, src_conc):
        self.tr.step(self.cs.c, dt, bc_conc, self.cs.ledger, src_conc=src_conc, retard=self.retard)
        self.solver["transport_steps"] += 1

    def _deposit(self, dt):
        for sp_, lam, fS, dp in self._particle_species():
            deposit_particles(self.cs, self.tr.theta, self.tr.vmag, dt, self.rp, self.V, d_p=dp, species=sp_,
                              lam=lam, frac_S=fS)

    def _react(self, dt, force=False):
        """반응 dt 초. PHREEQC 엔진은 chem_dt_h 간격으로 모아서 호출."""
        if self.c.chem_mode != "SLAD" or self.chem_engine == "내장":
            react(self.cs, self.tr.theta, self.V, dt, self.rp, vmag=self.tr.vmag)
            self.solver["chem_calls"] += 1
            return
        self._chem_acc += dt
        if self._chem_acc < self.c.chem_dt_h * 3600.0 - 1e-9 and not force:
            return
        dtc, self._chem_acc = self._chem_acc, 0.0
        if dtc <= 0:
            return
        self._phreeqc_react(dtc)

    def _phreeqc_react(self, dt):
        import phreeqc_engine as PE
        cs = self.cs
        th = self.tr.theta
        m = ((cs.S > 0) | (cs.Cal > 0) | (cs.Gyp > 0)) & (th > 1e-6)
        if not m.any():
            if np.any(cs.Bio > 0):
                cs.Bio *= np.exp(-self.rp.bio_decay * dt / 86400.0)
            return
        ii = np.where(m)[0]
        # O2 단계·액체 공여체는 내장 코드로 먼저(연산자 분할), PHREEQC는 SLAD·탄산계·석회석·석고
        if self.rp.o2_on or self.rp.donor is not None:
            import copy as _c
            rp2 = _c.copy(self.rp); rp2.k_half = 0.0
            react(cs, th, self.V, dt, rp2, cell_mask=m, vmag=self.tr.vmag)
        t_ = th[ii]
        cells = {"NO3": cs.c["NO3"][ii] * 1e-3, "SO4": cs.c["SO4"][ii] * 1e-3, "Ca": cs.c["Ca"][ii] * 1e-3,
                 "ALK": cs.c["ALK"][ii] * 1e-3, "Ssol": cs.S[ii] / t_ * 1e-3, "Cal": cs.Cal[ii] / t_ * 1e-3,
                 "Gyp": cs.Gyp[ii] / t_ * 1e-3, "theta": t_, "aCal": cs.aCal[ii]}
        finh = 1.0
        if self.rp.o2_on and "O2" in cs.c:
            finh = self.rp.K_I_O2 / (self.rp.K_I_O2 + cs.c["O2"][ii])
        import copy as _c
        rp = _c.copy(self.rp)
        if np.ndim(finh):
            rp_list = None
        try:
            if np.ndim(finh) and np.any(finh < 0.999):
                # 억제 계수를 셀별 온도 보정 항에 곱해 넘긴다(PARM(6))
                out = PE.react_cells(cells, dt, rp, self.pCO2_cells[ii], self.c.calcite_mode, self.phreeqc,
                                     fT_cells=finh)
            else:
                out = PE.react_cells(cells, dt, rp, self.pCO2_cells[ii], self.c.calcite_mode, self.phreeqc)
        except Exception as e:      # noqa: BLE001
            raise SimulationFailed(f"{self.t:.3f}일: PHREEQC 반응 계산 실패 — {str(e)[:300]}")
        self.solver["phreeqc_calls"] += 1
        L = cs.ledger
        V = self.V
        dN = out["dN"]
        s = self.rp.stoich_SO4
        dmolN = float((dN * 1e3 * t_).sum() * V)
        L["N_denit"] += dmolN
        L["ALK_denit"] += (2 * s - 1) * dmolN
        newS = out["Ssol"] * 1e3 * t_
        newCal = out["Cal"] * 1e3 * t_
        newGyp = out["Gyp"] * 1e3 * t_
        dCal = float(((cs.Cal[ii] - newCal)).sum() * V)
        L["Cal_diss"] += dCal
        L["ALK_cal"] += 2 * dCal
        L["Gyp_prec"] += float((newGyp - cs.Gyp[ii]).sum() * V)
        cs.Bio[ii] += self.rp.bio_vol * dN * 1e3 * t_
        cs.Bio *= np.exp(-self.rp.bio_decay * dt / 86400.0)
        if self.rp.bio_det > 0:
            cs.Bio *= np.exp(-self.rp.bio_det * self.tr.vmag * dt)
        cs.c["NO3"][ii] = out["NO3"] * 1e3; cs.c["SO4"][ii] = out["SO4"] * 1e3; cs.c["Ca"][ii] = out["Ca"] * 1e3
        cs.c["ALK"][ii] = out["ALK"] * 1e3
        cs.S[ii] = newS; cs.Cal[ii] = newCal; cs.Gyp[ii] = newGyp
        self._last_pH_phreeqc = (ii, out["pH"])

    def _advance(self, dt_total_s, bc_conc, src_conc=None, particles=False, react_on=True):
        """현재 흐름장으로 이송(+입자 포획)+반응을 dt_total_s 만큼. 시계도 함께 진행."""
        t_left = dt_total_s
        fac = max(float(self.c.dt_factor), 1e-3)
        strang = self.c.splitting == "Strang" and react_on
        while t_left > 1e-9:
            self._check_stop()
            dt = min(self.tr.dt_max * fac, t_left)
            if strang:
                self._transport(0.5 * dt, bc_conc, src_conc)
                if particles:
                    self._deposit(0.5 * dt)
                self._react(dt)
                self._transport(0.5 * dt, bc_conc, src_conc)
                if particles:
                    self._deposit(0.5 * dt)
            else:
                self._transport(dt, bc_conc, src_conc)
                if particles:
                    self._deposit(dt)
                if react_on:
                    self._react(dt)
            t_left -= dt
            self._clock(dt)

    def _pass(self, dt_s, operating):
        """설치·운영 중 시간이 흐르는 구간(개폐·센서·카트리지 교체 등). 관개는 운영 중이면 계속된다(가정)."""
        if dt_s <= 0:
            return
        conc = self._inflow_conc(operating)
        relax = (not operating) and self.c.flow_mode == "과도" and self._pressurized()
        if (operating or relax) and self.c.flow_mode == "과도":
            # 운영 중: 관개 계속. 설치 중: 주입으로 생긴 압력·수분이 퍼지는 재분배를 과도 흐름으로 계산(관개 없음)
            t_end = self.t + dt_s / 86400.0
            while self.t < t_end - 1e-12:
                q = self.q_at(self.t - self.t_op0) if operating else 0.0
                self.fv.update_Ks(*self._material())
                st, dt = self._flow_step(q, (t_end - self.t) * 86400.0)
                self._set_flow(st)
                self._advance(dt, conc, particles=True, react_on=operating)
        else:
            self._advance(dt_s, conc, particles=True, react_on=operating)

    def _pressurized(self):
        """주입 뒤 아직 압력(양의 압력수두)이나 큰 수분 불균형이 남아 있는가."""
        h = np.asarray(self.h).ravel()[self.active]
        return bool(getattr(self, "_injected", False)) and len(h) > 0

    # ── 기록 ─────────────────────────────────────────────────────────
    def _frame(self, label, kind, extra=None):
        self._check_stop()
        g = self.g
        solid = (self.cs.S * VM["S"] + self.cs.Cal * VM["CaCO3"]).reshape(g.shape).astype(np.float32)
        Ptot = self.cs.c["P"] + (self.cs.c["PC"] if "PC" in self.cs.c else 0.0)
        fr = {"t": self.t, "label": label, "kind": kind, "inserted": sorted(self.inserted),
              "insert_frac": dict(self.insert_frac),
              "open": dict(self.open_state), "solid": solid,
              "P": np.asarray(Ptot).reshape(g.shape).astype(np.float32),
              "theta": (self.tr.theta.reshape(g.shape).astype(np.float32) if self.tr and self.tr.st is not None
                        else None)}
        if "D" in self.cs.c:
            fr["D"] = self.cs.c["D"].reshape(g.shape).astype(np.float32)
        if extra:
            fr.update(extra)
        self.frames.append(fr)

    def _pH_cells(self, mask=None):
        c = self.cs.c
        sel = np.where(self.active if mask is None else mask)[0]
        pH = np.full(self.g.N, np.nan)
        if len(sel):
            pH[sel] = pH_of(c["ALK"][sel], c["Ca"][sel], c["SO4"][sel], c["NO3"][sel], self.pCO2_cells[sel], self.I_bg)
        return pH

    def _snapshot(self, label="운영"):
        g = self.g
        c = self.cs.c
        pH = self._pH_cells()
        Ksh, Ksv, ths = self._material()
        kr = np.where(self.active, Ksv / np.maximum(self.Ks_v0, 1e-30), np.nan)
        sn = {
            "t": self.t, "op_day": self.t - self.t_op0, "label": label,
            "NO3": (c["NO3"] * MW["N"]).reshape(g.shape).astype(np.float32),
            "SO4": (c["SO4"] * MW["SO4"]).reshape(g.shape).astype(np.float32),
            "pH": pH.reshape(g.shape).astype(np.float32),
            "solid": (self.cs.S * VM["S"] + self.cs.Cal * VM["CaCO3"]).reshape(g.shape).astype(np.float32),
            "Kratio": kr.reshape(g.shape).astype(np.float32),
            "theta": self.tr.theta.reshape(g.shape).astype(np.float32),
            "h": np.asarray(self.h).reshape(g.shape).astype(np.float32),
        }
        if "D" in c:
            sn["D"] = c["D"].reshape(g.shape).astype(np.float32)
        self.snaps.append(sn)

    # ── 이벤트 처리 ──────────────────────────────────────────────────
    def _targets(self, row, role_name):
        names = [n.strip() for n in str(row.get("대상 객체") or "전체").split(",") if n.strip()]
        allnames = [o.name for o in self.sc.objects if self.geom["roles_of"][o.name] == role_name] \
            if role_name else [o.name for o in self.sc.objects]
        if not names or names == ["전체"]:
            return allnames
        out = [n for n in allnames if n in names or any(n.lower() == x.lower() for x in names)]
        miss = [x for x in names if x not in out and x != "전체"]
        if miss and role_name:
            self.warn.append(f"대상 객체 {miss}는 '{role_name}' 역할이 아니거나 없습니다.")
        return out

    def _insert(self, row, operating=False):
        targets = self._targets(row, None)
        pl = self.sc.placement
        moved = False
        for key, attr in (("x(m)", "x"), ("y(m)", "y"), ("깊이(m)", "top_depth"),
                          ("기울기(°)", "tilt_deg"), ("회전(°)", "rot_deg")):
            v = row.get(key)
            if v is not None and getattr(pl, attr) != v:
                setattr(pl, attr, float(v))
                moved = True
        if moved:
            self.geom = build_geometry(self.sc, self.c, self.g)
        nfr = 4 if self.c.light else 8
        dur_s = float(row.get("소요(분)") or 0) * 60.0
        self.inserted |= set(targets)
        for k in range(1, nfr + 1):
            fr = k / nfr
            for n in targets:
                self.insert_frac[n] = fr
            self._apply_structure()                  # 진행률만큼 구조가 흐름에 반영(A9)
            self._set_flow(self._static_flow() if not operating else self._steady(self.q_at(self.t - self.t_op0)))
            self._pass(dur_s / nfr, operating)
            self._frame(f"구조물 삽입 ({k}/{nfr})", "insert", {"insert_progress": fr, "inserting": targets})
        for n in targets:
            self.insert_frac[n] = 1.0
        self.events_log.append({"t": self.t, "event": "구조물 삽입", "targets": targets, "duration_min": dur_s / 60})

    def _pull(self, row):
        targets = self._targets(row, None)
        self.inserted -= set(targets)
        self._apply_structure()
        self._set_flow(self._static_flow())
        self._pass(float(row.get("소요(분)") or 0) * 60.0, False)
        self.events_log.append({"t": self.t, "event": "케이싱 인발", "targets": targets})
        self._frame("케이싱 인발", "pull")

    def _opening_cells(self, targets, connected_only=False):
        cells = np.where(np.isin(self.obj_cell, [self.objects_by_name[n] for n in targets]) &
                         (self.role_cell == ROLES.index("주입 개구부")) & self.active)[0]
        if not connected_only or len(cells) == 0 or self.fv is None:
            return cells
        # 벽(장벽 면)에 완전히 갇혀 흙과 이어지지 않은 개구부 셀은 주입 대상에서 뺀다. 넣은 물이 나갈 곳이 없어
        # 압력이 무한히 오르고 선형계가 풀리지 않는다(×1.5 격자에서 개구부 셀 1개가 갇힌 것을 확인).
        import scipy.sparse as _sp
        import scipy.sparse.csgraph as _cg
        fv = self.fv
        oc = fv.cmap[cells]
        isop = np.zeros(fv.nA, bool); isop[oc] = True
        G = _sp.csr_matrix((np.ones(len(fv.fa)), (fv.fa, fv.fb)), shape=(fv.nA, fv.nA))
        _, lab = _cg.connected_components(G, directed=False)
        has_soil = np.zeros(lab.max() + 1, bool)
        np.logical_or.at(has_soil, lab[~isop], True)
        keep = has_soil[lab[oc]]
        if not keep.all():
            self.warn.append(f"개구부 셀 {int((~keep).sum())}개가 벽에 갇혀 흙과 이어지지 않아 주입에서 뺐습니다"
                             f"(격자가 개구부 두께를 제대로 담지 못함 — 격자를 바꿔 보세요).")
        return cells[keep]

    def _straining(self, d_p) -> bool:
        if not self.c.straining:
            return False
        d50 = D50_TEXTURE.get(self.c.texture, 1e-4)
        return d_p / d50 > self.c.p("strain_ratio")

    def _inject(self, row, operating=False, label="시약 주입"):
        """개구부를 통한 시약 주입. 흐름 = 과도 Richards(원천항/고정수두, B1) 또는 중첩 근사(옵션).
        시약 형태: (c) 슬러리 입자 P, (e) 콜로이드 PC, (d) 용존 공여체 D."""
        g = self.g
        allt = self._targets(row, "주입 개구부")
        targets = [n for n in allt if self.open_state.get(n, True)]
        closed = [n for n in allt if not self.open_state.get(n, True)]
        if closed:
            self.warn.append(f"{label}: 닫힌 개구부 {closed}는 건너뜁니다.")
        cells = self._opening_cells(targets, connected_only=True)
        rec = {"t": self.t, "event": label, "targets": targets, "planned_L": row.get("부피(L)") or 0.0,
               "t_start_min": (self.t - (self.t_op0 or 0.0)) * 1440.0}
        if len(cells) == 0:
            self.warn.append(f"{label}: 주입할 개구부 셀이 없습니다(꽂히지 않았거나 닫힘).")
            rec["delivered_L"] = 0.0
            self.events_log.append(rec)
            return rec
        form = str(row.get("시약 형태") or "") or self.form or "(c) 입자 슬러리(포획)"
        rg = (self.sc.reagent or {})
        V_L = float(row.get("부피(L)") or 0.0)
        fs = row.get("고형분비(v/v)")
        fs = 0.2 if fs is None else float(fs)
        if form.startswith("(d)"):
            fs = 0.0
        mode = row.get("방식") or "펌프"
        dur_min = row.get("소요(분)")
        Vw = V_L * 1e-3 * (1 - fs)
        conc_inj = {"NO3": 0.0, "SO4": self.c.SO4_in_mg / MW["SO4"], "Ca": self.c.Ca_in_mg / MW["Ca"],
                    "ALK": self.alk_in, "P": 0.0}
        species = None
        if form.startswith("(d)"):
            conc_inj["D"] = float(rg.get("conc_molL") or 0.0) * 1e3      # mol/m3
        elif form.startswith("(e)"):
            conc_inj["PC"] = fs / max(1 - fs, 1e-6)
            species = "PC"
        else:
            conc_inj["P"] = fs / max(1 - fs, 1e-6)
            species = "P"
        if "O2" in self.cs.c:
            conc_inj["O2"] = self.c.p("DO_in") / 32.0
        d_p = float(rg.get("d_p") or self.c.p("colloid_d_p")) if species == "PC" else self.c.p("inj_d_p")
        strain = species is not None and self._straining(d_p)
        rec["straining"] = bool(strain)
        if strain:
            rec["note"] = (f"입자 지름/토양 d50 = {d_p / D50_TEXTURE.get(self.c.texture, 1e-4):.2f} > "
                           f"{self.c.p('strain_ratio')}: 입자가 토양 공극에 못 들어가 개구부에 필터 케이크로 쌓임(B7)")
        q = self.q_at(self.t - self.t_op0) if (operating and self.t_op0 is not None) else 0.0
        top_conc = self._inflow_conc(operating)
        delivered = 0.0
        t_cap = (float(dur_min) * 60.0) if dur_min else None
        elapsed = 0.0
        nfr = 4 if self.c.light else 8
        next_frame = Vw / nfr
        base = self.st
        Pin = conc_inj.get(species, 0.0) if species else 0.0
        h_open_max = -np.inf
        use_super = self.c.inject_model.startswith("중첩")
        # 펌프 최대 압력(가정): 표의 압력(kPa) 칸이 있으면 그 값, 없으면 300 kPa. 넘으면 그 압력으로 제한된 주입으로 전환.
        p_lim_kPa = float(row.get("압력(kPa)") or 300.0) if mode == "펌프" else None
        pump_limited = False
        # 큰 격자(셀 3만 개 초과)에서만: 압력 제한 주입으로 바뀔 때 개구부 수두를 한 번에 한계까지 올리지 않고
        # 짧은 단계(≤2 s)로 나눠 올린다(수치 연속법). ×1.5 격자에서 마른 흙에 30 m 수두를 한 단계에 걸면
        # 습윤 전선이 한 셀에 몰려 뉴턴이 Δt 0.1 s에서도 발산했다(시험 확인). 올리는 시간은 수십 초.
        # 기본·가벼움 격자는 예전과 같이 바로 한계 압력을 건다(자체 점검 E5 숫자 그대로).
        use_ramp = g.N > DIRECT_FIRST_MAX
        if use_ramp and mode == "펌프" and not use_super:
            # 큰 격자: 정해진 유량을 넣는 데 필요한 압력을 먼저 싸게 추정(현재 흙의 상대투수 kr(h)를 쓴 선형 섭동 해).
            # 펌프 한계를 넘으면 처음부터 압력 제한 주입으로 시작한다. ×2 격자에서 유량 주입 첫 단계는 압력이 수 km로
            # 계산되는 극단적인 계라 모든 선형 풀이가 수 분씩 실패했다(시험 확인). 작은 격자는 예전 순서 그대로.
            Qmin_ = row.get("유량(L/min)")
            Qw_ = (float(Qmin_) / 60000.0 * (1 - fs)) if Qmin_ else (Vw / (float(dur_min) * 60.0) if dur_min else 1.0 / 60000.0 * (1 - fs))
            s_ = np.zeros(g.N); s_[cells] = Qw_ / len(cells)
            try:
                kr_ = vg_kr(self.h.ravel(), self.alpha, self.n)
                pert_ = self.fv.injection_perturbation(src=s_, bottom_dirichlet=self.bottom != "자유배수", K_scale=kr_)
                h_need = float(np.max(self.h.ravel()[cells] + pert_["dH"].ravel()[cells]))
                rec["pressure_estimate_kPa"] = h_need * 9.81
                if h_need * 9.81 > p_lim_kPa:
                    pump_limited = True
                    self.warn.append(f"{label}: 정해진 유량에 필요한 압력(추정 {h_need * 9.81:.0f} kPa)이 펌프 최대 압력 "
                                     f"{p_lim_kPa:.0f} kPa(가정)를 넘어 처음부터 압력 제한 주입으로 계산합니다.")
            except LinearSolveError:
                pass
        h_fix = None
        dh_ramp = 2.0
        ramp_steps = 0
        while delivered < Vw * (1 - 1e-6):
            self._check_stop()
            self.fv.update_Ks(*self._material())
            # 원천 정의
            src = fixed = None
            ramping = False
            if mode == "펌프" and pump_limited:
                hlim = p_lim_kPa * 1000.0 / 9810.0
                h_tgt = hlim
                if use_ramp:
                    if h_fix is None:
                        h_fix = float(np.clip(np.max(self.st.h.ravel()[cells]), 0.0, hlim))
                    h_tgt = min(hlim, h_fix + dh_ramp)
                    ramping = h_fix < hlim - 1e-9
                fixed = np.full(g.N, np.nan)
                fixed[cells] = h_tgt
            elif mode == "펌프":
                Qmin = row.get("유량(L/min)")
                if Qmin:
                    Qw = float(Qmin) / 60000.0 * (1 - fs)
                elif dur_min:
                    Qw = Vw / (float(dur_min) * 60.0)
                else:
                    Qw = 1.0 / 60000.0 * (1 - fs)
                src = np.zeros(g.N); src[cells] = Qw / len(cells)
            else:
                fixed = np.full(g.N, np.nan)
                depth = g.H - self.zc[cells]
                hp = float(row.get("압력(kPa)") or 0.0) * 1000.0 / 9810.0 if mode == "가압" else 0.0
                fixed[cells] = hp + depth          # 관이 지표까지 차 있는 수두 + 가압 → 압력수두
            remain = Vw - delivered
            if use_super:
                st, dt, Qin = self._inject_superposition(base, cells, src, fixed, remain, t_cap, elapsed)
            else:
                Qguess = (src.sum() if src is not None else getattr(self, "_Q_last", 1e-7))
                dt_t = min(remain / max(Qguess, 1e-12), 120.0)
                if t_cap is not None:
                    dt_t = min(dt_t, t_cap - elapsed)
                if ramping:
                    dt_t = min(dt_t, 2.0)
                if dt_t <= 0:
                    break
                try:
                    st, dt = self._flow_step(q, dt_t, src=src, fixed=fixed)
                except SimulationFailed:
                    if mode == "펌프" and pump_limited and ramping and dh_ramp > 0.05:
                        dh_ramp *= 0.25
                        continue
                    if mode == "펌프" and not pump_limited:
                        pump_limited = True
                        self.warn.append(f"{label}: 정해진 유량을 넣는 압력이 수렴하지 않을 만큼 커져(막힘) "
                                         f"최대 압력 {p_lim_kPa:.0f} kPa(가정) 제한 주입으로 바꿨습니다.")
                        continue
                    raise
                Qin = float(np.clip(st.Fsrc.ravel()[cells], 0, None).sum())
                if mode == "펌프" and not pump_limited and float(np.max(st.h.ravel()[cells])) * 9.81 > p_lim_kPa:
                    # 정해진 유량을 넣으려면 펌프 한계보다 큰 압력이 필요 → 이 단계는 버리고 압력 제한 주입으로 다시 푼다
                    pump_limited = True
                    self.warn.append(f"{label}: 정해진 유량을 넣는 데 펌프 최대 압력 {p_lim_kPa:.0f} kPa(가정)보다 큰 압력이 필요해 "
                                     "압력 제한 주입으로 바꿨습니다(막힘).")
                    continue
                self._Q_last = max(Qin, 1e-12)
                if ramping:
                    ramp_steps += 1
                    h_fix = h_tgt
                    dh_ramp = min(dh_ramp * 1.5, 5.0)
            if dt <= 0:
                break
            if Qin <= 1e-15:
                self.warn.append(f"{label}: 주입 유량이 0입니다(압력 부족/막힘). 주입 중단.")
                break
            h_open_max = max(h_open_max, float(np.max(st.h.ravel()[cells])))
            self._set_flow(st)
            if strain:
                # 입자는 토양으로 못 들어가므로 개구부 셀에 직접 쌓는다(남은 공극까지). 남는 고형분은 '케이크 초과' 기록.
                cin = dict(conc_inj); cin[species] = 0.0
                self._advance(dt, top_conc, src_conc=cin, particles=False, react_on=operating)
                self._cake_deposit(cells, Qin * dt * Pin, species, d_p)
            else:
                self._advance(dt, top_conc, src_conc=conc_inj, particles=True, react_on=operating)
            delivered += Qin * dt
            elapsed += dt
            if delivered >= next_frame - 1e-12:
                self._frame(f"{label} ({min(100, 100 * delivered / max(Vw, 1e-12)):.0f}%)", "inject",
                            {"inject_cells": cells, "Q_Lmin": Qin / max(1 - fs, 1e-9) * 60000})
                next_frame += Vw / nfr
            if t_cap is not None and elapsed >= t_cap - 1e-6:
                break
            if strain and self._opening_full(cells):
                self.warn.append(f"{label}: 개구부가 필터 케이크로 가득 차 주입이 막혔습니다({delivered * 1e3:.2f} L 주입 후).")
                break
        rec["delivered_L"] = delivered / max(1 - fs, 1e-9) * 1e3
        rec["duration_min"] = elapsed / 60.0
        rec["t_end_min"] = (self.t - (self.t_op0 or 0.0)) * 1440.0
        rec["mode"] = mode
        rec["form"] = form
        rec["pump_limited"] = bool(pump_limited)
        rec["pressure_ramp_steps"] = ramp_steps
        rec["inject_model"] = "중첩 근사" if use_super else "과도 Richards"
        rec["max_pressure_kPa"] = float(max(h_open_max, 0.0) * 9.81) if np.isfinite(h_open_max) else 0.0
        rec["pressure_limited"] = bool(pump_limited)
        if rec["delivered_L"] < 0.95 * V_L:
            self.warn.append(f"{label}: 계획 {V_L:.2f} L 중 {rec['delivered_L']:.2f} L만 주입됨(시간 제한/압력 부족/막힘).")
        if rec.get("max_pressure_kPa", 0) > 300:
            self.warn.append(f"{label}: 개구부 압력이 {rec['max_pressure_kPa']:.0f} kPa로 매우 큽니다"
                             "(개구부 주변 막힘). 실제로는 토양 할렬·역류가 생길 수 있습니다.")
        self.events_log.append(rec)
        self.fv.update_Ks(*self._material())
        # 주입 끝: 원천 없는 흐름으로 복귀
        self._injected = True
        if operating and self.c.flow_mode == "준정상":
            self._set_flow(self._steady(q))
        elif not operating:
            if self.c.flow_mode == "과도":
                self._pass(60.0, False)          # 주입 직후 1분: 압력 해소(재분배)를 과도 흐름으로
            else:
                self._set_flow(self._static_flow())
        return rec

    def _inject_superposition(self, base, cells, src, fixed, remain, t_cap, elapsed):
        """중첩 근사(빠름): 현재 흐름 + K=Ks·kr(h) 로 푼 섭동 흐름, θ ← min(θs, θ+섭동 저류) 근사."""
        import copy
        g = self.g
        if base is None:
            base = self._static_flow()
        kr = vg_kr(self.h.ravel(), self.alpha, self.n)
        bottom_dir = self.bottom != "자유배수"
        if fixed is not None:
            fd = fixed.copy(); ok = np.isfinite(fd)
            fd[ok] = np.maximum(fd[ok] - self.h.ravel()[ok], 0.0)
            pert = self.fv.injection_perturbation(fixed_dH=fd, bottom_dirichlet=bottom_dir, K_scale=kr)
        else:
            pert = self.fv.injection_perturbation(src=src, bottom_dirichlet=bottom_dir, K_scale=kr)
        Qin = float(pert["Fsrc"].ravel()[cells].clip(min=0).sum())
        if Qin <= 0:
            return base, 0.0, 0.0
        dt = remain / Qin / 4.0
        if t_cap is not None:
            dt = min(dt, max(t_cap - elapsed, 0.0))
        st = copy.copy(base)
        st.Fx = base.Fx + pert["Fx"]; st.Fy = base.Fy + pert["Fy"]; st.Fz = base.Fz + pert["Fz"]
        st.Ftop = base.Ftop + pert["Ftop"]; st.Fbot = base.Fbot + pert["Fbot"]
        st.Fsrc = base.Fsrc + pert["Fsrc"]
        # θ 갱신: 주입수가 주입구 주변을 포화시키는 근사(B1 최소 보정)
        Ksh, Ksv, ths = self._material()
        th = base.theta.ravel().copy()
        st.theta = np.minimum(ths, th + np.clip(pert["dH"].ravel(), 0, None) * 0.0 +
                              np.where(pert["dH"].ravel() > 1e-3, ths - th, 0.0)).reshape(g.shape)
        h = self.h.ravel().copy()
        h = np.where(pert["dH"].ravel() > 1e-3, np.maximum(h, 0.0) + pert["dH"].ravel(), h)
        st.h = h.reshape(g.shape)
        return st, dt, Qin

    def _cake_deposit(self, cells, vol_solid, species, d_p):
        """필터 케이크: 고형분을 개구부 셀에 남은 공극만큼 쌓는다(흐름 비례 배분). 남으면 cake_overflow."""
        if vol_solid <= 0:
            return
        cs = self.cs
        phi = cs.porosity()[cells]
        room = np.maximum(phi - cs.phi_min[cells], 0.0) * self.V
        tot_room = room.sum()
        put = min(vol_solid, tot_room)
        if tot_room > 0:
            share = room / tot_room * put
        else:
            share = np.zeros(len(cells))
        r = self.c.ratio_S_CaCO3
        fS = 1.0 if species == "PC" else r / (r + 1.0)
        dS = share / self.V * fS / VM["S"]
        dC = share / self.V * (1 - fS) / VM["CaCO3"]
        newC = cs.Cal[cells] + dC
        cs.aCal[cells] = np.where(newC > 0, (cs.aCal[cells] * cs.Cal[cells] + 6.0 / d_p * dC) / np.maximum(newC, 1e-30),
                                  cs.aCal[cells])
        cs.S[cells] += dS
        cs.Cal[cells] += dC
        # 케이크가 절반 넘게 찬 빈 공간 셀은 수분특성도 '입자 충전층(모래 유사, 가정)'으로 바꾼다
        filled = (cs.phi_matrix[cells] - cs.porosity()[cells]) > 0.5 * (cs.phi_matrix[cells] - cs.phi_min[cells])
        if filled.any():
            sand = SOIL_TEXTURES["모래(Sand)"]
            cf = cells[filled]
            self.alpha[cf] = sand["alpha"]; self.n[cf] = sand["n"]
            self.fv.alpha[cf] = sand["alpha"]; self.fv.n[cf] = sand["n"]
        L = cs.ledger
        L["S_inj"] += float(dS.sum() * self.V)
        L[f"{species}_dep"] = L.get(f"{species}_dep", 0.0) + float(put)
        L[f"{species}_srcin"] = L.get(f"{species}_srcin", 0.0) + float(vol_solid)
        L["cake_overflow"] = L.get("cake_overflow", 0.0) + float(vol_solid - put)
        self.cake_m3 += put

    def _opening_full(self, cells):
        return bool(np.all(self.cs.porosity()[cells] - self.cs.phi_min[cells] < 1e-6))

    def _standby(self, row, operating=False):
        """정치: 부유 입자는 흐름을 따라 계속 이동·포획. 끝에 남은 공극만큼 즉시 포획(A16)."""
        dur = float(row.get("소요(분)") or 0.0) * 60.0
        if dur > 0:
            nst = 3 if self.c.light else 6
            for k in range(nst):
                self._pass(dur / nst, operating)
                self._frame(f"정치(대기) {100 * (k + 1) / nst:.0f}%", "standby")
        dep = self._capture_all()
        self.events_log.append({"t": self.t, "event": "정치(대기)", "deposited_m3": dep})
        self._frame("정치 완료 — 입자 포획", "standby")

    def _capture_all(self):
        dep = 0.0
        for sp_, lam, fS, dp in self._particle_species():
            dep += deposit_particles(self.cs, self.tr.theta, self.tr.vmag, 0.0, self.rp, self.V, all_now=True,
                                     d_p=dp, species=sp_, lam=lam, frac_S=fS)
        return dep

    def _inflow_conc(self, operating):
        c = self.c
        gw = {"NO3": c.gw_NO3_mgN / MW["N"], "SO4": c.SO4_in_mg / MW["SO4"], "Ca": c.Ca_in_mg / MW["Ca"],
              "ALK": self.alk_in}
        if "O2" in self.cs.c:
            gw["O2"] = 0.0
        if not operating:
            return {"top": {}, "gw": gw}
        top = {"NO3": c.NO3_in_mgN / MW["N"], "SO4": c.SO4_in_mg / MW["SO4"], "Ca": c.Ca_in_mg / MW["Ca"],
               "ALK": self.alk_in, "P": 0.0}
        if "O2" in self.cs.c:
            top["O2"] = c.p("DO_in") / 32.0
        return {"top": top, "gw": gw}

    # ── 설치 단계 ────────────────────────────────────────────────────
    def run_installation(self):
        self._rebuild_models()
        self._dt_flow = 30.0
        self._set_flow(self._static_flow())
        self._frame("설치 전 흙수조", "start")
        if self.sc is None:
            return
        rows = ie.schedule(self.sc.events)
        self.refill_rules = [r for r in rows if r["이벤트"] == "재보충"]
        if not any(r["이벤트"] == "구조물 삽입" for r in rows):
            self.inserted = set(o.name for o in self.sc.objects)
            self._apply_structure()
            self._set_flow(self._static_flow())
            self._frame("구조물 설치됨(0일)", "insert", {"insert_progress": 1.0, "inserting": []})
        for idx, r in enumerate(rows):
            if r["이벤트"] == "재보충":
                continue
            if r["t_start"] is not None and r["t_start"] > self.t:
                self._pass((r["t_start"] - self.t) * 86400.0, False)
            ev = r["이벤트"]
            if ev == "구조물 삽입":
                self._insert(r)
            elif ev == "케이싱 인발":
                self._pull(r)
            elif ev == "케이싱 유지":
                self.events_log.append({"t": self.t, "event": ev})
            elif ev in ("개구부 열기", "개구부 닫기"):
                for n in self._targets(r, "주입 개구부"):
                    self.open_state[n] = (ev == "개구부 열기")
                self.events_log.append({"t": self.t, "event": ev})
                self._frame(ev, "open")
                self._pass((r["소요(분)"] or 0.0) * 60.0, False)
            elif ev == "시약 주입":
                self._inject(r)
                nxt = rows[idx + 1]["이벤트"] if idx + 1 < len(rows) else None
                if nxt != "정치(대기)":
                    self._capture_all()
                    self._frame("주입 종료 — 입자 포획", "inject")
            elif ev == "정치(대기)":
                self._standby(r)
            elif ev == "센서 측정":
                self.events_log.append({"t": self.t, "event": ev})
                self._frame("센서 측정", "sensor")
                self._pass((r["소요(분)"] or 0.0) * 60.0, False)
        self._capture_all()

    # ── 운영 단계 ────────────────────────────────────────────────────
    def _masks(self):
        g = self.g
        thr = self.c.p("reactive_thr_frac") * self.c.p("S_ref_frac")
        reac = (self.cs.S * VM["S"] > thr) & self.active
        R3 = reac.reshape(g.shape)
        dil = R3.copy()
        for _ in range(2):
            d2 = dil.copy()
            d2[1:] |= dil[:-1]; d2[:-1] |= dil[1:]
            d2[:, 1:] |= dil[:, :-1]; d2[:, :-1] |= dil[:, 1:]
            d2[:, :, 1:] |= dil[:, :, :-1]; d2[:, :, :-1] |= dil[:, :, 1:]
            dil = d2
        around = (dil & ~R3).ravel() & self.active
        below = np.zeros(g.shape, bool)
        nb = max(1, int(round(0.10 / g.dz)))
        if R3.any():
            ii, jj = np.where(R3.any(axis=2))
            for i, j in zip(ii, jj):
                kmax = np.where(R3[i, j])[0].max()
                below[i, j, kmax + 1:min(g.nz, kmax + 1 + nb)] = True
        below = below.ravel() & self.active & ~reac
        return reac, around, below

    def _pH_mean(self, mask):
        if not mask.any():
            return np.nan
        return float(np.nanmean(self._pH_cells(mask)[mask]))

    def _pH_outflow(self):
        g = self.g
        k = min(g.k_interface, g.nz) - 1
        k = max(k, 0)
        idx = np.arange(g.N).reshape(g.shape)[:, :, k].ravel()
        if k < g.nz - 1:
            w = np.clip(self.st.Fz[:, :, k].ravel(), 0, None)
        else:
            w = np.clip(self.st.Fbot.ravel(), 0, None)
        ok = (w > 0) & self.active[idx]
        if not ok.any():
            return np.nan
        c = self.cs.c
        sel = idx[ok]
        wt = w[ok] / w[ok].sum()
        ALK = (c["ALK"][sel] * wt).sum(); Ca = (c["Ca"][sel] * wt).sum()
        SO4 = (c["SO4"][sel] * wt).sum(); NO3 = (c["NO3"][sel] * wt).sum()
        pc = float((self.pCO2_cells[sel] * wt).sum())
        return float(pH_of(np.array([ALK]), np.array([Ca]), np.array([SO4]), np.array([NO3]), pc, self.I_bg)[0])

    def _Kreac(self, reac_mask0):
        if reac_mask0 is None or not reac_mask0.any():
            return np.nan
        _, Ksv, _ = self._material()
        k = Ksv[reac_mask0]
        return float(len(k) / np.sum(1.0 / np.maximum(k, 1e-30)))

    def _do_refill(self, rule, eta_now, n_done):
        method = rule.get("방식") or "펌프"
        rec = {"day": self.t - self.t_op0, "t_start_min": (self.t - self.t_op0) * 1440.0, "eta_before": eta_now,
               "method": method, "n": n_done + 1, "duration_min": float(rule.get("소요(분)") or 0.0)}
        sc = self.sc
        t_start = self.t
        if method == "카트리지 교체":
            names = self._targets(rule, "반응재(고체)")
            acc = [n for n in names if sc.roles.get(n, {}).get("accessible", False) or sc.allow_refill_wo_opening]
            if not acc:
                rec["ok"] = False
                rec["why"] = "교체 가능한 반응재 객체가 없음(역할표 '교체 가능' 체크 필요)"
                return rec
            ids = [self.objects_by_name[n] for n in acc]
            cells = np.isin(self.obj_cell, ids) & (self.role_cell == ROLES.index("반응재(고체)"))
            # 교체 시간 동안 관개·이송은 계속(가정) → 끝에 새 카트리지
            self._pass(rec["duration_min"] * 60.0, True)
            rmix = self.c.ratio_S_CaCO3
            self.phi_ref[cells] = self.phi_ref0[cells]            # 기준 공극률 초기값으로(A16)
            vs = 1.0 - self.phi_ref0[cells]
            self.cs.S[cells] = vs * rmix / (rmix + 1) / VM["S"]
            self.cs.Cal[cells] = vs / (rmix + 1) / VM["CaCO3"]
            self.cs.Gyp[cells] = 0; self.cs.Bio[cells] = 0; self.cs.Inert[cells] = 0
            self.cs.aCal[cells] = 6.0 / self.c.p("cal_d_p")
            self.cs.ledger["S_inj"] += float(self.cs.S[cells].sum() * self.V)
            rec["ok"] = True
            rec["targets"] = acc
        else:
            if not any(self.geom["roles_of"][o.name] == "주입 개구부" for o in sc.objects):
                rec["ok"] = False
                rec["why"] = "개구부가 없어 재보충 불가(R=0)"
                return rec
            r = self._inject(rule, operating=True, label=f"재보충 {n_done + 1}회")
            self._capture_all()
            rec["ok"] = r.get("delivered_L", 0) > 0
            rec["delivered_L"] = r.get("delivered_L", 0)
            rec["targets"] = r.get("targets", [])
            used = (self.t - t_start) * 1440.0
            if rec["duration_min"] > used:                 # 표의 소요 시간이 더 길면 나머지도 흐르게(같은 규칙)
                self._pass((rec["duration_min"] - used) * 60.0, True)
            rec["duration_min"] = max(rec["duration_min"], used)
        rec["t_end_min"] = (self.t - self.t_op0) * 1440.0
        return rec

    def _reference_inflow(self, reac, q):
        """반응재가 없을 때(같은 흙, 같은 경계) 같은 셀 집합으로 들어갈 유량 — 통과 비율의 분모(A12)."""
        Ksh = self.soil["Ks_h"]; Ksv = self.soil["Ks_v"]
        nx, ny, nz = self.g.shape
        bar0 = (np.zeros((nx - 1, ny, nz)), np.zeros((nx, ny - 1, nz)), np.zeros((nx, ny, nz - 1)), np.zeros((nx, ny)))
        fv0 = RichardsFV(self.g, np.ones(self.g.shape, bool), Ksh, Ksv, self.soil["thr"], self.soil["ths"],
                         self.soil["alpha"], self.soil["n"], bar0)
        st0 = fv0.solve_steady(None, self._bc(max(q, 0.0)))
        if not st0.info.get("final_conv"):
            return np.nan
        return self._inflow_into(reac, st0)

    def _inflow_into(self, reac, st):
        g = self.g
        R = reac.reshape(g.shape)
        inflow = 0.0
        for ax, F in enumerate((st.Fx, st.Fy, st.Fz)):
            sa = [slice(None)] * 3; sb = [slice(None)] * 3
            sa[ax] = slice(0, -1); sb[ax] = slice(1, None)
            a = R[tuple(sa)]; b = R[tuple(sb)]
            inflow += float(np.clip(F, 0, None)[~a & b].sum() + np.clip(-F, 0, None)[a & ~b].sum())
        inflow += float(np.clip(st.Ftop, 0, None)[R[:, :, 0]].sum())
        return inflow

    def _through_flow(self, reac):
        """반응재 통과 유량 / '반응재가 없을 때 같은 위치를 통과할 유량'(A12). 1에 가까우면 흐름을 그대로 받고,
        0이면 우회(예: 바닥이 막힌 케이슨), 1보다 크면 흐름이 반응재로 모임."""
        if reac is None or not reac.any():
            return {"through_ratio": np.nan}
        q = max(self.q_at(0.0), self.c.q_mm_day / 1000.0 / 86400.0)
        if self.c.flow_mode == "과도":
            st = self._steady(q)
        else:
            st = self.st
        inflow = self._inflow_into(reac, st)
        ref = self._reference_inflow(reac, q)
        ratio = inflow / ref if (ref and np.isfinite(ref) and ref > 0) else np.nan
        verdict = ("판정 불가" if not np.isfinite(ratio) else
                   "우회: 반응재를 거의 거치지 않음" if ratio < 0.2 else
                   "부분 우회" if ratio < 0.7 else "정상 통과" if ratio <= 1.5 else "흐름 집중")
        return {"through_ratio": ratio, "inflow_m3s": inflow, "ref_inflow_m3s": ref, "through_verdict": verdict}

    def _record_day(self, t0, t1, last, Keff):
        c = self.c
        L = self.cs.ledger
        dN = L["IF_N"] - last["IF_N"]; dW = L["IF_W"] - last["IF_W"]; dS = L["IF_SO4"] - last["IF_SO4"]
        dNin = L.get("N_top_in", 0.0) - last["N_top_in"]; dWin = L.get("W_top_in", 0.0) - last["W_top_in"]
        Cout = dN / dW * MW["N"] if dW > 0 else np.nan
        SO4o = dS / dW * MW["SO4"] if dW > 0 else np.nan
        eta = 1.0 - Cout / c.NO3_in_mgN if (dW > 0 and c.NO3_in_mgN > 0) else np.nan
        reac, around, below = self._masks()
        D = self.daily
        op_day = t1
        D["day"].append(op_day); D["t0"].append(t0); D["t1"].append(t1)
        D["eta"].append(eta); D["Cout_mgN"].append(Cout)
        D["SO4_out_mg"].append(SO4o); D["pH_out"].append(self._pH_outflow())
        D["pH_around"].append(self._pH_mean(around)); D["pH_below"].append(self._pH_mean(below))
        pHs = self._pH_cells(self.active & ~self.soil["zone"])
        D["pH_min"].append(float(np.nanmin(pHs)) if np.any(np.isfinite(pHs)) else np.nan)
        D["Keff"].append(Keff); D["Kreac"].append(self._Kreac(self.reac0))
        D["N_to_gw_g"].append(L["IF_N"] * MW["N"]); D["denit_g"].append(L["N_denit"] * MW["N"])
        aq = self.soil["zone"] & self.active
        D["gw_NO3_mgN"].append(float(self.cs.c["NO3"][aq].mean() * MW["N"]) if aq.any() else np.nan)
        D["water_out_L"].append(L["IF_W"] * 1e3)
        D["water_in_L"].append(L.get("W_top_in", 0.0) * 1e3)
        D["N_top_in_g"].append(L.get("N_top_in", 0.0) * MW["N"])
        D["reactive_g"].append(reactive_mass_g(self.cs.S, self.cs.Cal, self.V))
        D["q_mm_day"].append(self.q_at(t0) * 86400 * 1000)
        for k in ("IF_N", "IF_W", "IF_SO4"):
            last[k] = L[k]
        last["N_top_in"] = L.get("N_top_in", 0.0); last["W_top_in"] = L.get("W_top_in", 0.0)
        return eta

    def run_operation(self):
        c = self.c
        g = self.g
        self.t_op0 = self.t
        conc = self._inflow_conc(True)
        q0 = self.q_at(0.0)
        self._q_cur = q0
        if c.flow_mode == "준정상":
            st = self._steady(q0)
        else:
            st, _dt = self._flow_step(q0, 60.0)      # 운영 첫 단계(짧게) — 이후 적응
        self._set_flow(st)
        _, Ksv0, _ = self._material()
        self.Ks_v0 = Ksv0.copy()
        L = self.cs.ledger
        Vt = self.tr.theta * self.V
        L["N_store0"] = float((self.cs.c["NO3"] * Vt).sum())
        L["SO4_store0"] = float((self.cs.c["SO4"] * Vt).sum())
        L["Ca_store0"] = float((self.cs.c["Ca"] * Vt).sum())
        L["ALK_store0"] = float((self.cs.c["ALK"] * Vt).sum())
        L["S_solid0"] = float(self.cs.S.sum() * self.V)
        L["Gyp0"] = float(self.cs.Gyp.sum() * self.V)
        L["W_store0"] = float(Vt.sum())
        if "D" in self.cs.c:
            L["D_store0"] = float((self.cs.c["D"] * Vt * (self.retard or {}).get("D", 1.0)).sum())
        soilz = ~self.soil["zone"]
        L["N_soil0"] = float((self.cs.c["NO3"] * Vt)[soilz].sum())
        self.op_ledger0 = dict(L)
        reac0, _, _ = self._masks()
        self.reac0 = reac0
        self.mass0_g = reactive_mass_g(self.cs.S, self.cs.Cal, self.V)
        self.Keff0 = self.fv.constant_head_Keff(min(g.k_interface, g.nz))
        self.Kreac0 = self._Kreac(reac0)
        self.diag = self._through_flow(reac0)
        if c.flow_mode == "과도":
            self._set_flow(st)
        Keff = self.Keff0
        snap_times = np.linspace(0, c.days, max(2, int(c.n_snapshots)))
        self._snapshot("운영 시작")
        si = 1
        last_flow = 0.0
        last = {"IF_N": L["IF_N"], "IF_W": L["IF_W"], "IF_SO4": L["IF_SO4"],
                "N_top_in": L.get("N_top_in", 0.0), "W_top_in": L.get("W_top_in", 0.0)}
        last_refill = -1e9
        n_refill = {id(r): 0 for r in getattr(self, "refill_rules", [])}
        Ks_last = self._material()[1]
        days = int(np.ceil(c.days - 1e-9))
        rec_t0 = 0.0
        for d in range(1, days + 1):
            day_end = self.t_op0 + min(d, c.days)
            if self.t >= day_end - 1e-9:          # 재보충 등으로 이미 지나간 날
                continue
            while self.t < day_end - 1e-9:
                self._check_stop()
                op = self.t - self.t_op0
                if c.flow_mode == "과도":
                    t_lim = min(day_end, self.t_op0 + self._next_q_change(op))
                    dt_t = min((t_lim - self.t) * 86400.0, c.flow_dt_max_h * 3600.0)
                    self.fv.update_Ks(*self._material())
                    st, dt = self._flow_step(self.q_at(op), dt_t)
                    self._set_flow(st)
                    self._advance(dt, conc, particles=True)
                else:
                    qn = self.q_at(op)
                    if abs(qn - self._q_cur) > 1e-15:          # 강우 시계열이 바뀌면 정상해 다시
                        self.fv.update_Ks(*self._material())
                        self._set_flow(self._steady(qn))
                        self._q_cur = qn
                    t_lim = min(day_end, self.t_op0 + self._next_q_change(op))
                    self._advance((t_lim - self.t) * 86400.0, conc, particles=True)
            self._react(0.0, force=True)
            op_day = self.t - self.t_op0
            _, Ksv, _ = self._material()
            rel = np.max(np.abs(Ksv - Ks_last) / np.maximum(Ks_last, 1e-30)) if self.active.any() else 0
            if c.flow_mode == "준정상" and (rel > 0.05 or op_day - last_flow >= c.flow_update_days - 1e-9):
                self.fv.update_Ks(*self._material())
                self._q_cur = self.q_at(op_day)
                self._set_flow(self._steady(self._q_cur))
                last_flow = op_day
                Ks_last = Ksv.copy()
                Keff = self.fv.constant_head_Keff(min(g.k_interface, g.nz))
            elif c.flow_mode == "과도" and rel > 0.05:
                self.fv.update_Ks(*self._material())
                Ks_last = Ksv.copy()
                Keff = self.fv.constant_head_Keff(min(g.k_interface, g.nz))
            eta = self._record_day(rec_t0, op_day, last, Keff)
            rec_t0 = op_day
            # 재보충 판단: 그날의 η(유입 대비 유출)만 사용 — 평가 시작일로 늦추지 않음(A10)
            if self.sc is not None:
                for rule in getattr(self, "refill_rules", []):
                    if rule.get("_disabled"):
                        continue
                    trig = rule.get("트리거 η(%)")
                    intv = rule.get("간격(일)")
                    nmax = int(rule.get("최대 회차") or 99)
                    if n_refill[id(rule)] >= nmax:
                        continue
                    do = False
                    if trig is None and intv is None:
                        trig = 50.0
                    if trig is not None and np.isfinite(eta) and eta * 100 < trig \
                            and op_day - last_refill >= (intv or 1.0):
                        do = True
                    if trig is None and intv is not None and op_day - max(last_refill, 0) >= intv:
                        do = True
                    if do:
                        rec = self._do_refill(rule, eta, n_refill[id(rule)])
                        n_refill[id(rule)] += 1
                        last_refill = op_day
                        self.refills.append(rec)
                        if rec.get("ok"):
                            self.fv.update_Ks(*self._material())
                            if c.flow_mode == "준정상":
                                self._set_flow(self._steady(self.q_at(self.t - self.t_op0)))
                                last_flow = self.t - self.t_op0
                            Ks_last = self._material()[1]
                            Keff = self.fv.constant_head_Keff(min(g.k_interface, g.nz))
                        else:
                            self.warn.append(f"재보충 시도 실패({op_day:.0f}일): {rec.get('why', '')}")
                            if "불가" in rec.get("why", "") or "없음" in rec.get("why", ""):
                                rule["_disabled"] = True
                        # 재보충이 걸린 시간은 다음 기록 구간에 포함(일 루프 왜곡 방지, A9)
            while si < len(snap_times) and op_day >= snap_times[si] - 1e-6:
                self._snapshot()
                si += 1
            self.progress(min(1.0, op_day / c.days), f"{op_day:.1f}/{c.days:g}일")

    # ── 수지·보고 ────────────────────────────────────────────────────
    def mass_balance(self):
        L = self.cs.ledger
        L0 = self.op_ledger0
        Vt = self.tr.theta * self.V
        out = {}
        cN = float((self.cs.c["NO3"] * Vt).sum())
        dN_in = L["N_in"] - L0["N_in"]; dN_out = L["N_out"] - L0["N_out"]; den = L["N_denit"] - L0["N_denit"]
        errN = (cN - L["N_store0"]) - (dN_in - dN_out - den)
        out["N_err_rel"] = errN / max(dN_in, L["N_store0"], 1e-30)
        SO4 = float((self.cs.c["SO4"] * Vt).sum()); Ssol = float(self.cs.S.sum() * self.V)
        Gyp = float(self.cs.Gyp.sum() * self.V)
        dS_in = L["S_in"] - L0["S_in"]; dS_out = L["S_out"] - L0["S_out"]; dS_inj = L["S_inj"] - L0["S_inj"]
        dS_D = L.get("S_from_D", 0.0) - L0.get("S_from_D", 0.0)
        tot_now = SO4 + Ssol + Gyp
        tot0 = L["SO4_store0"] + L["S_solid0"] + L["Gyp0"]
        errS = (tot_now - tot0) - (dS_in - dS_out + dS_inj + dS_D)
        out["S_err_rel"] = errS / max(tot0 - L["S_solid0"] + dS_in, 1e-30)
        ALK = float((self.cs.c["ALK"] * Vt).sum())
        dA = (L["ALK_in"] - L0["ALK_in"]) - (L["ALK_out"] - L0["ALK_out"]) + (L["ALK_cal"] - L0["ALK_cal"]) \
            - (L["ALK_denit"] - L0["ALK_denit"]) - (L.get("ALK_O2", 0.0) - L0.get("ALK_O2", 0.0))
        errA = (ALK - L["ALK_store0"]) - dA
        out["ALK_err_rel"] = errA / max(abs(L["ALK_in"] - L0["ALK_in"]), abs(L["ALK_store0"]), 1e-30)
        Ca = float((self.cs.c["Ca"] * Vt).sum())
        dC = (L["Ca_in"] - L0["Ca_in"]) - (L["Ca_out"] - L0["Ca_out"]) + (L["Cal_diss"] - L0["Cal_diss"]) \
            - (L["Gyp_prec"] - L0["Gyp_prec"])
        out["Ca_err_rel"] = ((Ca - L["Ca_store0"]) - dC) / max(L["Ca_in"] - L0["Ca_in"], L["Ca_store0"], 1e-30)
        # 물: 경계 유입−유출 vs 저장량 변화. 준정상은 흐름장 교체 때 저장량이 계단식으로 바뀌므로 따로 표시.
        Wn = float(Vt.sum())
        dW_b = (L["W_in"] - L0["W_in"]) - (L["W_out"] - L0["W_out"])
        dW_s = Wn - L["W_store0"]
        out["W_err_rel"] = (dW_b - dW_s) / max(L["W_in"] - L0["W_in"], 1e-30)
        out["W_note"] = ("준정상 근사: 흐름장 교체 때 저장량 계단 변화(물리 플럭스 아님)가 차이에 포함됨"
                         if self.c.flow_mode == "준정상" else "과도 Richards: 경계 플럭스와 저장량 변화의 차")
        # 입자(고체 m3)
        for sp_ in ("P", "PC"):
            srcin = L.get(f"{sp_}_srcin", 0.0)
            if srcin > 0:
                susp = float((self.cs.c[sp_] * Vt).sum()) if sp_ in self.cs.c else 0.0
                clos = srcin - L.get(f"{sp_}_dep", 0.0) - L.get(f"{sp_}_out", 0.0) - susp - L.get("cake_overflow", 0.0)
                out[f"{sp_}_err_rel"] = clos / srcin
        return out

    def check_balance(self):
        """방침 2(b): 허용 오차를 넘으면 경고(질량수지가 맞는 것이 물리적 정확성의 증거는 아님)."""
        mb = self.mass_balance()
        tol = {"N_err_rel": 1e-6, "S_err_rel": 1e-6, "Ca_err_rel": 1e-6, "ALK_err_rel": 1e-6,
               "W_err_rel": 1e-3 if self.c.flow_mode == "과도" else np.inf, "P_err_rel": 1e-6, "PC_err_rel": 1e-6}
        bad = {k: v for k, v in mb.items() if k in tol and np.isfinite(v) and abs(v) > tol[k]}
        for k, v in bad.items():
            self.warn.append(f"질량수지 경고: {k} = {v:.2e} (허용 {tol[k]:.0e})")
        return mb, bad, tol

    def result(self, wall):
        g = self.g
        self._collect_stats()
        D = {k: np.array(v, dtype=float) for k, v in self.daily.items()}
        sc = self.sc
        try:
            mb, bad, tol = self.check_balance() if self.t_op0 is not None and hasattr(self, "op_ledger0") else ({}, {}, {})
        except Exception as e:      # noqa: BLE001
            mb, bad, tol = {"error": str(e)}, {}, {}
        solver = {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                  for k, v in self.solver.items()}
        return {
            "status": self.status, "fail_reason": self.fail_reason,
            "name": sc.name if sc else "대조군(구조물 없음)",
            "is_control": sc is None,
            "mechanism": sc.mechanism if sc else "반응재를 넣지 않은 같은 흙수조.",
            "install_text": sc.install_text if sc else "",
            "events_table": ie.clean_rows(sc.events) if sc else [],
            "reagent": dict(sc.reagent) if sc else {},
            "grid": dict(nx=g.nx, ny=g.ny, nz=g.nz, Lx=g.Lx, Ly=g.Ly, H=g.H, H_soil=g.H_soil,
                         k_interface=g.k_interface, dx=g.dx, dy=g.dy, dz=g.dz),
            "daily": D,
            "snaps": self.snaps,
            "frames": self.frames,
            "events_log": self.events_log,
            "refills": self.refills,
            "role_cell": self.role_cell.reshape(g.shape),
            "obj_cell": self.obj_cell.reshape(g.shape),
            "active": self.active.reshape(g.shape),
            "reac0": getattr(self, "reac0", np.zeros(g.N, bool)).reshape(g.shape),
            "mass0_g": getattr(self, "mass0_g", 0.0),
            "Keff0": getattr(self, "Keff0", np.nan), "Kreac0": getattr(self, "Kreac0", np.nan),
            "ledger": dict(self.cs.ledger),
            "ledger0": dict(getattr(self, "op_ledger0", {})),
            "mass_balance": mb, "mass_balance_bad": bad, "mass_balance_tol": tol,
            "diag": getattr(self, "diag", {}),
            "warnings": list(dict.fromkeys(self.warn)),
            "t_op0": self.t_op0,
            "flow_solves": self.flow_solves,
            "solver": solver,
            "chem_engine": self.chem_engine,
            "wall_s": wall,
            "placement": vars(sc.placement).copy() if sc else None,
            "object_names": [o.name for o in sc.objects] if sc else [],
            "roles_of": self.geom["roles_of"] if self.geom else {},
            "open_state": dict(self.open_state),
            "cake_m3": self.cake_m3,
            "vmax": float(np.max(self.tr.vmag[self.active])) if (self.tr is not None and self.tr.st is not None
                                                                and self.active.any()) else None,
        }


def code_hashes():
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    out = {}
    for fn in sorted(os.listdir(here)):
        if fn.endswith(".py"):
            with open(os.path.join(here, fn), "rb") as fh:
                out[fn] = hashlib.sha256(fh.read()).hexdigest()[:16]
    return out


def provenance(c: Common, sc: Optional[Scenario], sim: Optional["Sim"] = None):
    """재현성 기록(방침 2h): 입력·파라미터·격자·solver 설정·패키지 버전·실행 시각·파일 해시."""
    import scipy
    vers = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "platform": platform.platform()}
    for mod in ("streamlit", "plotly", "pandas", "openpyxl"):
        try:
            vers[mod] = __import__(mod).__version__
        except Exception:
            pass
    try:
        import phreeqc_engine as PE
        vers["phreeqc"] = PE.status()
    except Exception:
        pass
    params = {k: (c.p(k), REGISTRY[k][1], ("보정" if k in getattr(c, "calibrated", {}) else REGISTRY[k][2]))
              for k in REGISTRY}
    return {
        "engine_version": ENGINE_VERSION,
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
        "versions": vers,
        "common": c.to_dict(),
        "params": params,
        "solver_settings": {"flow_mode": c.flow_mode, "flow_dt_max_h": c.flow_dt_max_h, "flow_dt_min_s": c.flow_dt_min_s,
                            "newton_tol_m": 1e-7, "max_total_newton": 60, "lin_relres_ok": 1e-6,
                            "chem_engine": c.chem_engine, "chem_dt_h": c.chem_dt_h, "splitting": c.splitting,
                            "transport": "TVD van Leer, CFL 0.45 × dt_factor", "dt_factor": c.dt_factor},
        "scenario": None if sc is None else {"name": sc.name, "file_name": sc.file_name, "file_sha256": sc.file_hash,
                                             "roles": sc.roles, "placement": vars(sc.placement),
                                             "events": ie.clean_rows(sc.events), "reagent": sc.reagent},
        "code_sha256": code_hashes(),
        "chem_engine_used": sim.chem_engine if sim else None,
    }


def run_scenario(c: Common, sc: Optional[Scenario], progress=None, stop=None, eval_start_day=None):
    """한 비교군(또는 대조군) 실행. 수렴 실패·엔진 오류는 예외를 숨기지 않고 status='FAILED' 결과로 돌려준다."""
    t0 = time.time()
    sim = Sim(c, sc, progress, stop, eval_start_day)
    if sc is not None:
        errs = list(sim.geom["errors"]) + validate_reagent(sc)
        if errs:
            raise ValueError("\n".join(errs))
    try:
        sim.run_installation()
        sim.run_operation()
    except StopRequested:
        raise
    except (SimulationFailed, LinearSolveError, FloatingPointError) as e:
        sim.status = "FAILED"
        sim.fail_reason = str(e)
        sim.warn.append("FAILED: " + str(e))
        if sim.t_op0 is None:
            sim.t_op0 = sim.t
        if not hasattr(sim, "reac0"):
            sim.reac0 = np.zeros(sim.g.N, bool)
    res = sim.result(time.time() - t0)
    res["provenance"] = provenance(c, sc, sim)
    return res


def control_breakthrough_day(ctrl_res, frac=0.5, c_in=None):
    D = ctrl_res["daily"]
    cin = c_in or 1.0
    if not len(D.get("Cout_mgN", [])):
        return None
    ok = np.where(np.isfinite(D["Cout_mgN"]) & (D["Cout_mgN"] >= frac * cin))[0]
    return round(float(D["day"][ok[0]]), 6) if len(ok) else None


def eval_start_from(ctrl_res, c_in):
    """η 평가 시작일: 대조군 유출 농도가 유입의 90%에 도달한 날(없으면 50%, 그것도 없으면 None)."""
    d = control_breakthrough_day(ctrl_res, 0.9, c_in)
    if d is None:
        d = control_breakthrough_day(ctrl_res, 0.5, c_in)
    return d
