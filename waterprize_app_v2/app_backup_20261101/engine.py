"""시뮬레이션 엔진: 설치 타임라인 재현 → 운영(질산염 유입) → 결과 기록.

비교군은 하드코딩하지 않는다. 시나리오(3MF 객체 + 역할표 + 배치 + 설치 단계 표)만 다르고
모든 계산은 같은 물리(flow_richards, transport_reaction)를 쓴다.
대조군 = 구조물·반응재 없는 같은 흙수조(scenario=None).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

import install_events as ie
from flow_richards import FlowBC, FlowState, RichardsFV, vg_h_from_Se, vg_theta
from params import (MW, REGISTRY, RHO_SOLID, ROLE_DEFAULTS, ROLES, SOIL_TEXTURES, VM, Common)
from transport_reaction import (LEDGER_KEYS, ChemState, ReactParams, Transport, alk_from_pH,
                                deposit_particles, kozeny_carman_ratio, pH_of, react)
from voxel import Grid, Placement, compose, make_grid, voxelize_objects


class StopRequested(Exception):
    pass


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
    """사전 배치 반응재 질량(메시 체적 기준·복셀 기준)과 설치 단계 주입 계획 질량(g)."""
    r = c.ratio_S_CaCO3
    rho_mix = (r * RHO_SOLID["S"] + RHO_SOLID["CaCO3"]) / (r + 1) * 1000.0   # g/m3 고체
    pre_mesh = pre_vox = 0.0
    for o in sc.objects:
        rr = sc.roles.get(o.name, {})
        if rr.get("role") != "반응재(고체)":
            continue
        phi = float(rr.get("phi") or ROLE_DEFAULTS["반응재(고체)"]["phi"])
        pre_mesh += o.volume_m3 * sc.placement.scale ** 3 * (1 - phi) * rho_mix
        pre_vox += geom["comp"]["counts"].get(o.name, 0) * g.vcell * (1 - phi) * rho_mix
    inj = 0.0
    for row in ie.clean_rows(sc.events):
        if row["이벤트"] == "시약 주입" and row["부피(L)"]:
            fs = row["고형분비(v/v)"] if row["고형분비(v/v)"] is not None else 0.2
            inj += row["부피(L)"] * 1e-3 * fs * rho_mix
    return {"pre_mesh_g": pre_mesh, "pre_voxel_g": pre_vox, "injection_g": inj,
            "total_g": (pre_mesh if sc.mass_correction else pre_vox) + inj}


# ───────────────────────────── 시뮬레이터 ───────────────────────────────
class Sim:
    def __init__(self, c: Common, sc: Optional[Scenario], progress=None, stop=None, eval_start_day=None):
        self.c = c
        self.sc = sc
        self.g = make_grid(c)
        self.progress = progress or (lambda f, m: None)
        self.stop = stop or (lambda: False)
        self.eval_start_day = eval_start_day
        self.warn: List[str] = []
        self.t = 0.0                       # 일
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
        self.disturb = np.ones(N)
        self.active = np.ones(N, bool)
        self.role_cell = np.full(N, -1, np.int8)
        self.obj_cell = np.full(N, -1, np.int16)
        nx, ny, nz = g.shape
        self.barrier = (np.zeros((nx - 1, ny, nz)), np.zeros((nx, ny - 1, nz)),
                        np.zeros((nx, ny, nz - 1)), np.zeros((nx, ny)))
        self.inserted = set()
        self.open_state: Dict[str, bool] = {}
        # 화학 초기
        p = c.p
        self.pCO2 = p("pCO2")
        self.I_bg = p("I_bg")
        alk_soil = alk_from_pH(c.soil_pH0, self.pCO2, self.I_bg)
        self.alk_in = c.alk_in_mgCaCO3 / 50.04          # meq/L = eq/m3
        zone = self.soil["zone"]
        self.cs = ChemState(
            c={"NO3": np.where(zone, c.gw_NO3_mgN, c.soil_NO3_mgN) / MW["N"],
               "SO4": np.full(N, c.SO4_in_mg / MW["SO4"]),
               "Ca": np.full(N, c.Ca_in_mg / MW["Ca"]),
               "ALK": np.where(zone, self.alk_in, alk_soil),
               "P": np.zeros(N)},
            S=np.zeros(N), Cal=np.zeros(N), Gyp=np.zeros(N), Bio=np.zeros(N), Inert=np.zeros(N),
            aCal=np.full(N, 6.0 / p("cal_d_p")), phi_matrix=self.soil["ths"].copy(),
            ledger={k: 0.0 for k in LEDGER_KEYS})
        self.rp = ReactParams(k_half=p("k_half"), S_ref=p("S_ref_frac"), pH_low=p("pH_low"),
                              pH_high=p("pH_high"), theta_T=p("theta_T"), T=c.temperature, T_ref=p("T_ref"),
                              stoich_SO4=p("stoich_SO4"), stoich_H=p("stoich_H"), stoich_S=p("stoich_S"),
                              pCO2=self.pCO2, I_bg=self.I_bg, bio_vol=p("bio_vol_per_mol"),
                              bio_decay=p("bio_decay"), calcite_mode=c.calcite_mode,
                              lambda_filt=p("lambda_filt"), ratio=c.ratio_S_CaCO3)
        # 수리 경계
        self.wt_depth = c.wt_depth if c.wt_depth is not None else g.interface_depth
        self.z_wt = g.H - self.wt_depth
        self.k_wt = int(np.clip(np.floor(self.wt_depth / g.dz + 0.5), 0, nz - 1))
        Ks_aq = SOIL_TEXTURES[c.aq_texture]["Ks"]
        self.gw_grad = (c.gw_velocity / 86400.0) / Ks_aq if (c.H_aq > 0 and c.gw_velocity) else 0.0
        self.bottom = c.bottom_bc if c.H_aq > 0 or c.bottom_bc == "자유배수" else c.bottom_bc
        # 초기 수두
        Se0 = np.clip((c.init_saturation * self.soil["ths"] - self.thr) / (self.soil["ths"] - self.thr), 1e-3, 1.0)
        h = vg_h_from_Se(Se0, self.alpha, self.n)
        zc = (g.H - np.broadcast_to(g.dc[None, None, :], g.shape)).ravel()
        if self.bottom != "자유배수":
            h = np.where(zc < self.z_wt, self.z_wt - zc, h)
        self.h = h.reshape(g.shape)
        self.zc = zc
        # 기록
        self.frames: List[dict] = []
        self.snaps: List[dict] = []
        self.daily: Dict[str, list] = {k: [] for k in (
            "day", "eta", "Cout_mgN", "SO4_out_mg", "pH_out", "pH_around", "pH_below", "Keff", "Kreac",
            "N_to_gw_g", "gw_NO3_mgN", "denit_g", "water_out_L", "reactive_g")}
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

    # ── 구조물 적용 ───────────────────────────────────────────────────
    def _apply_structure(self):
        """현재 꽂혀 있는 객체로 셀 역할/장벽/재료 갱신."""
        g = self.g
        sc = self.sc
        ov = [v for v in self.geom["ov"] if v.name in self.inserted]
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
            # comp의 객체 번호(부분 목록 기준) → 전체 목록 번호
            idx_map = np.array([self.objects_by_name[v.name] for v in ov] + [-1], dtype=np.int16)
            obj = np.where(o2 >= 0, idx_map[np.maximum(o2, 0)], -1).astype(np.int16)
            barrier = comp["barrier"]
        newly = (role >= 0) & (self.role_cell < 0)
        removed = (role < 0) & (self.role_cell >= 0)
        # 새로 생긴 구조 셀
        p = self.c
        rmix = p.ratio_S_CaCO3
        for ri, rname in enumerate(ROLES):
            m = newly & (role == ri)
            if not m.any():
                continue
            if rname == "불투수":
                continue
            o_ids = obj[m]
            for oid in np.unique(o_ids):
                name = sc.objects[oid].name
                rr = sc.roles.get(name, {})
                cells = np.where(m & (obj == oid))[0]
                d = ROLE_DEFAULTS[rname]
                K = rr.get("K", d["K"])
                phi = rr.get("phi", d["phi"])
                al = rr.get("alpha", d["alpha"])
                nn = rr.get("n", d["n"])
                if rname in ("토양", "주입 개구부") and (K in (None, "") or K != K):
                    continue
                K = float(K if K not in (None, "") else (d["K"] or 1e-5))
                phi = float(phi if phi not in (None, "") else (d["phi"] or 0.4))
                self.Ks_ref_h[cells] = K
                self.Ks_ref_v[cells] = K
                self.alpha[cells] = float(al if al not in (None, "") else 14.5)
                self.n[cells] = float(nn if nn not in (None, "") else 2.68)
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
                else:
                    self.cs.phi_matrix[cells] = phi
                    self.phi_ref[cells] = phi
        # 제거된 구조 셀(인발) → 토양으로
        if removed.any():
            m = removed
            self.Ks_ref_h[m] = self.soil["Ks_h"][m]; self.Ks_ref_v[m] = self.soil["Ks_v"][m]
            self.alpha[m] = self.soil["alpha"][m]; self.n[m] = self.soil["n"][m]; self.thr[m] = self.soil["thr"][m]
            self.cs.phi_matrix[m] = self.soil["ths"][m]; self.phi_ref[m] = self.soil["ths"][m]
            self.cs.S[m] = 0; self.cs.Cal[m] = 0; self.cs.Gyp[m] = 0; self.cs.Bio[m] = 0
        new_active = role != ROLES.index("불투수")
        lost = self.active & ~new_active
        if lost.any():
            for k in self.cs.c:
                self.cs.c[k][lost] = 0.0
        self.active = new_active
        self.role_cell = role
        self.obj_cell = obj
        self.barrier = barrier
        # 교란대
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
        ths = np.maximum(phi, self.thr + 0.01)
        return Ksh, Ksv, ths

    def _rebuild_models(self):
        Ksh, Ksv, ths = self._material()
        self.fv = RichardsFV(self.g, self.active.reshape(self.g.shape), Ksh, Ksv, self.thr, ths,
                             self.alpha, self.n, self.barrier)
        p = self.c
        self.tr = Transport(self.g, self.fv, dict(alpha_L=p.p("alpha_L"), alpha_T_ratio=p.p("alpha_T_ratio"),
                                                  Dm=p.p("Dm"), phi_cells=ths))
        self.st = None

    def _bc(self, q, src=None, fixed=None):
        return FlowBC(q_top=q, bottom=self.bottom, z_wt=self.z_wt, gw_grad=self.gw_grad,
                      wt_k=self.k_wt, src=src, fixed_h=fixed)

    def _set_flow(self, st):
        """흐름장 교체(수분량 변화 시 질량 보존: c ← c·θ_old/θ_new)."""
        th_new = st.theta.ravel()
        if self.tr.st is not None:
            th_old = self.tr.theta
            ok = th_new > 0
            for k in self.cs.c:
                self.cs.c[k] = np.where(ok, self.cs.c[k] * th_old / np.where(ok, th_new, 1.0), 0.0)
        else:
            th_old = getattr(self, "_theta_prev", None)
            if th_old is not None and th_old.shape == th_new.shape:
                ok = th_new > 0
                for k in self.cs.c:
                    self.cs.c[k] = np.where(ok, self.cs.c[k] * th_old / np.where(ok, th_new, 1.0), 0.0)
        self.tr.p["phi_cells"] = self._material()[2]
        self.tr.set_flow(st)
        self.st = st
        self._theta_prev = th_new.copy()
        self.h = st.h

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
        st = self.fv.solve_steady(None, bc)
        self.flow_solves += 1
        # 지표 유출 재분배(가정): 구조물 덮개·포화로 못 들어간 물은 열린 지표로 흘러가 들어간다(흙수조 물수지 보존)
        if self.c.redistribute_runoff and q > 0:
            for _ in range(3):
                tot = q * self.g.Lx * self.g.Ly
                if st.runoff <= 1e-4 * tot:
                    break
                A = self.g.dx * self.g.dy
                tc = self.fv.cmap[self.fv.top_cells]
                open_ = self.fv.top_ok & ~st.ponded.ravel()
                if not open_.any():
                    break
                qarr = np.zeros(len(tc))
                qarr[open_] = tot / (open_.sum() * A)
                bc = self._bc(qarr, src, fixed)
                st = self.fv.solve_steady(None, bc)
                self.flow_solves += 1
            self.last_bc_q = bc.q_top
        if not st.info.get("final_conv", False):
            self.warn.append(f"{self.t:.2f}일: 흐름 정상해가 완전히 수렴하지 않았습니다(결과 해석 주의).")
        return st

    def _transient(self, q, dt_s, src=None, fixed=None):
        """적응 시간 간격 과도 Richards: 쉽게 수렴하면 Δt를 키우고, 어려우면 줄인다."""
        bc = self._bc(q, src, fixed)
        h = self.h
        pond = None
        t = 0.0
        dt = min(dt_s, getattr(self, "_dt_flow", 30.0))
        fails = 0
        while t < dt_s * (1 - 1e-12):
            dt = min(dt, dt_s - t)
            hn, conv, it, pn = self.fv.step(h, dt, bc, pond, max_iter=18)
            self.flow_solves += 1
            if conv:
                h, pond = hn, pn
                t += dt
                fails = 0
                dt = dt * (2.0 if it <= 5 else (1.0 if it <= 10 else 0.6))
            else:
                fails += 1
                if dt < 0.05 or fails > 12:
                    self.warn.append(f"{self.t:.3f}일: 과도 흐름 단계가 수렴하지 않아 근사값으로 진행했습니다.")
                    h, pond = hn, pn
                    t += dt
                else:
                    dt *= 0.3
        self._dt_flow = max(dt, 1.0)
        return self.fv.fluxes(h, bc, pond)

    # ── 이송·반응 한 구간 ────────────────────────────────────────────
    def _advance(self, dt_total_s, bc_conc, src_conc=None, particles=False, react_on=True):
        t_left = dt_total_s
        while t_left > 1e-9:
            if self.stop():
                raise StopRequested()
            dt = min(self.tr.dt_max, t_left)
            self.tr.step(self.cs.c, dt, bc_conc, self.cs.ledger, src_conc=src_conc)
            if particles:
                deposit_particles(self.cs, self.tr.theta, self.tr.vmag, dt, self.rp, self.V,
                                  d_p=self.c.p("inj_d_p"))
            if react_on:
                react(self.cs, self.tr.theta, self.V, dt, self.rp)
            t_left -= dt
            self.t += dt / 86400.0

    # ── 기록 ─────────────────────────────────────────────────────────
    def _frame(self, label, kind, extra=None):
        g = self.g
        solid = (self.cs.S * VM["S"] + self.cs.Cal * VM["CaCO3"]).reshape(g.shape).astype(np.float32)
        fr = {"t": self.t, "label": label, "kind": kind, "inserted": sorted(self.inserted),
              "open": dict(self.open_state), "solid": solid,
              "P": self.cs.c["P"].reshape(g.shape).astype(np.float32),
              "theta": (self.tr.theta.reshape(g.shape).astype(np.float32) if self.tr and self.tr.st is not None
                        else None)}
        if extra:
            fr.update(extra)
        self.frames.append(fr)

    def _snapshot(self, label="운영"):
        g = self.g
        c = self.cs.c
        pH = pH_of(c["ALK"], c["Ca"], c["SO4"], c["NO3"], self.pCO2, self.I_bg)
        pH = np.where(self.active, pH, np.nan)
        Ksh, Ksv, ths = self._material()
        kr = np.where(self.active, Ksv / np.maximum(self.Ks_v0, 1e-30), np.nan)
        self.snaps.append({
            "t": self.t, "op_day": self.t - self.t_op0, "label": label,
            "NO3": (c["NO3"] * MW["N"]).reshape(g.shape).astype(np.float32),
            "SO4": (c["SO4"] * MW["SO4"]).reshape(g.shape).astype(np.float32),
            "pH": pH.reshape(g.shape).astype(np.float32),
            "solid": (self.cs.S * VM["S"] + self.cs.Cal * VM["CaCO3"]).reshape(g.shape).astype(np.float32),
            "Kratio": kr.reshape(g.shape).astype(np.float32),
            "theta": self.tr.theta.reshape(g.shape).astype(np.float32),
        })

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

    def _insert(self, row):
        targets = self._targets(row, None)
        pl = self.sc.placement
        moved = False
        for key, attr in (("x(m)", "x"), ("y(m)", "y"), ("깊이(m)", "top_depth"),
                          ("기울기(°)", "tilt_deg"), ("회전(°)", "rot_deg")):
            v = row.get(key)
            if v is not None and getattr(pl, attr) != v:
                setattr(pl, attr, float(v))
                moved = True
        if moved:   # 배치가 바뀌면 다시 복셀화
            self.geom = build_geometry(self.sc, self.c, self.g)
        nfr = 4 if self.c.light else 8
        dur = (row.get("소요(분)") or 0) / 1440.0
        t0 = self.t
        for k in range(nfr + 1):
            self.t = t0 + dur * k / nfr
            self._frame(f"구조물 삽입 ({k}/{nfr})", "insert",
                        {"insert_progress": k / nfr, "inserting": targets})
        self.inserted |= set(targets)
        self._apply_structure()
        self._set_flow(self._static_flow())
        self.events_log.append({"t": self.t, "event": "구조물 삽입", "targets": targets})

    def _pull(self, row):
        targets = self._targets(row, None)
        self.inserted -= set(targets)
        self._apply_structure()
        self._set_flow(self._static_flow())
        self.events_log.append({"t": self.t, "event": "케이싱 인발", "targets": targets})
        self._frame("케이싱 인발", "pull")

    def _inject(self, row, operating=False, label="시약 주입"):
        """개구부를 통한 슬러리 주입.
        흐름 = 현재 불포화 흐름장 + 주입 압력 섭동장(포화 Darcy, 중첩 근사 — 가정).
        입자 = 이송 + 여과 포획(λ|v|). 포획으로 공극이 줄면 K가 줄고, 다음 단계 섭동장에 반영."""
        import copy
        g = self.g
        allt = self._targets(row, "주입 개구부")
        targets = [n for n in allt if self.open_state.get(n, True)]
        closed = [n for n in allt if not self.open_state.get(n, True)]
        if closed:
            self.warn.append(f"{label}: 닫힌 개구부 {closed}는 건너뜁니다.")
        cells = np.where(np.isin(self.obj_cell, [self.objects_by_name[n] for n in targets]) &
                         (self.role_cell == ROLES.index("주입 개구부")) & self.active)[0]
        rec = {"t": self.t, "event": label, "targets": targets, "planned_L": row.get("부피(L)") or 0.0}
        if len(cells) == 0:
            self.warn.append(f"{label}: 주입할 개구부 셀이 없습니다(꽂히지 않았거나 닫힘).")
            rec["delivered_L"] = 0.0
            self.events_log.append(rec)
            return rec
        V_L = float(row.get("부피(L)") or 0.0)
        fs = row.get("고형분비(v/v)")
        fs = 0.2 if fs is None else float(fs)
        mode = row.get("방식") or "펌프"
        dur_min = row.get("소요(분)")
        Vw = V_L * 1e-3 * (1 - fs)
        P_in = fs / max(1 - fs, 1e-6)
        q = self.c.q_mm_day / 1000.0 / 86400.0 if operating else 0.0
        base = self.st if self.st is not None else (self._static_flow() if not operating else self._steady(q))
        bottom_dir = self.bottom != "자유배수"
        delivered = 0.0
        t_cap = (float(dur_min) * 60.0) if dur_min else None
        elapsed = 0.0
        top_conc = self._inflow_conc(operating)
        src_conc = {"NO3": 0.0, "SO4": self.c.SO4_in_mg / MW["SO4"], "Ca": self.c.Ca_in_mg / MW["Ca"],
                    "ALK": self.alk_in, "P": P_in}
        nsteps = 4 if self.c.light else 8
        it = 0
        h_now = self.h.ravel()
        while delivered < Vw * (1 - 1e-6) and it < 60:
            self.fv.update_Ks(*self._material())
            if mode == "펌프":
                Qmin = row.get("유량(L/min)")
                if Qmin:
                    Qw = float(Qmin) / 60000.0 * (1 - fs)
                elif dur_min:
                    Qw = Vw / (float(dur_min) * 60.0)
                else:
                    Qw = 1.0 / 60000.0 * (1 - fs)
                src = np.zeros(g.N); src[cells] = Qw / len(cells)
                pert = self.fv.injection_perturbation(src=src, bottom_dirichlet=bottom_dir)
            else:
                fixed = np.full(g.N, np.nan)
                depth = g.H - self.zc[cells]
                hp = float(row.get("압력(kPa)") or 0.0) * 1000.0 / 9810.0 if mode == "가압" else 0.0
                fixed[cells] = np.maximum(hp + depth - h_now[cells], 0.0)   # 관을 지표까지 채운 수두 + 가압
                pert = self.fv.injection_perturbation(fixed_dH=fixed, bottom_dirichlet=bottom_dir)
            Qin = float(pert["Fsrc"].ravel()[cells].clip(min=0).sum())
            rec["max_pressure_kPa"] = max(rec.get("max_pressure_kPa", 0.0),
                                          float(np.max(pert["dH"].ravel()[cells])) * 9.81)
            if Qin <= 1e-15:
                self.warn.append(f"{label}: 주입 유량이 0입니다(압력 부족/막힘). 주입 중단.")
                break
            remain = Vw - delivered
            dt_s = remain / Qin / max(1, nsteps - it) if it < nsteps else remain / Qin
            if t_cap is not None and elapsed + dt_s > t_cap:
                dt_s = max(t_cap - elapsed, 0.0)
            if dt_s <= 0:
                break
            st = copy.copy(base)
            st.Fx = base.Fx + pert["Fx"]; st.Fy = base.Fy + pert["Fy"]; st.Fz = base.Fz + pert["Fz"]
            st.Ftop = base.Ftop + pert["Ftop"]; st.Fbot = base.Fbot + pert["Fbot"]
            st.Fsrc = base.Fsrc + pert["Fsrc"]
            self._set_flow(st)
            self._advance(dt_s, top_conc, src_conc=src_conc, particles=True, react_on=operating)
            delivered += Qin * dt_s
            self.cs.ledger["P_inj"] += Qin * dt_s * P_in
            elapsed += dt_s
            it += 1
            self._frame(f"{label} ({min(100, 100 * delivered / max(Vw, 1e-12)):.0f}%)", "inject",
                        {"inject_cells": cells, "Q_Lmin": Qin / (1 - fs) * 60000})
            if t_cap is not None and elapsed >= t_cap - 1e-6:
                break
        rec["delivered_L"] = delivered / (1 - fs) * 1e3 if fs < 1 else 0.0
        rec["duration_min"] = elapsed / 60.0
        rec["mode"] = mode
        if rec["delivered_L"] < 0.95 * V_L:
            self.warn.append(f"{label}: 계획 {V_L:.2f} L 중 {rec['delivered_L']:.2f} L만 주입됨(시간 제한/압력 부족/막힘).")
        if rec.get("max_pressure_kPa", 0) > 300:
            self.warn.append(f"{label}: 정해진 유량을 넣는 데 필요한 압력이 {rec['max_pressure_kPa']:.0f} kPa로 매우 큽니다"
                             "(개구부 주변 막힘). 실제로는 토양 할렬·역류가 생길 수 있습니다.")
        self.events_log.append(rec)
        self.fv.update_Ks(*self._material())
        self._set_flow(base)
        return rec

    def _standby(self, row, operating=False):
        """정치: 흐름장은 직전 상태 유지(설치 단계는 정지 상태 — 가정), 부유 입자는 계속 이동·포획 후 모두 포획."""
        dur = float(row.get("소요(분)") or 0.0) * 60.0
        if dur > 0:
            nst = 3 if self.c.light else 6
            for k in range(nst):
                self._advance(dur / nst, self._inflow_conc(operating), particles=True, react_on=operating)
                self._frame(f"정치(대기) {100 * (k + 1) / nst:.0f}%", "standby")
        dep = deposit_particles(self.cs, self.tr.theta, self.tr.vmag, 0.0, self.rp, self.V, all_now=True,
                                d_p=self.c.p("inj_d_p"))
        self.events_log.append({"t": self.t, "event": "정치(대기)", "deposited_m3": dep})
        self._frame("정치 완료 — 입자 포획", "standby")

    def _inflow_conc(self, operating):
        c = self.c
        if not operating:
            return {"top": {}, "gw": {"NO3": c.gw_NO3_mgN / MW["N"], "SO4": c.SO4_in_mg / MW["SO4"],
                                      "Ca": c.Ca_in_mg / MW["Ca"], "ALK": self.alk_in}}
        top = {"NO3": c.NO3_in_mgN / MW["N"], "SO4": c.SO4_in_mg / MW["SO4"], "Ca": c.Ca_in_mg / MW["Ca"],
               "ALK": self.alk_in, "P": 0.0}
        gw = {"NO3": c.gw_NO3_mgN / MW["N"], "SO4": c.SO4_in_mg / MW["SO4"], "Ca": c.Ca_in_mg / MW["Ca"],
              "ALK": self.alk_in}
        return {"top": top, "gw": gw}

    # ── 설치 단계 ────────────────────────────────────────────────────
    def run_installation(self):
        self._rebuild_models()
        self._dt_flow = 120.0
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
        for r in rows:
            if r["이벤트"] == "재보충":
                continue
            if r["t_start"] is not None and r["t_start"] > self.t:
                gap = (r["t_start"] - self.t) * 86400.0
                self._advance(gap, self._inflow_conc(False), particles=True, react_on=False)
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
                self.t += (r["소요(분)"] or 0.0) / 1440.0
            elif ev == "시약 주입":
                self._inject(r)
                # 정치 단계가 바로 뒤에 없으면 즉시 포획
                nxt = rows[rows.index(r) + 1]["이벤트"] if rows.index(r) + 1 < len(rows) else None
                if nxt != "정치(대기)":
                    deposit_particles(self.cs, self.tr.theta, self.tr.vmag, 0.0, self.rp, self.V,
                                      all_now=True, d_p=self.c.p("inj_d_p"))
                    self._frame("주입 종료 — 입자 포획", "inject")
            elif ev == "정치(대기)":
                self._standby(r)
            elif ev == "센서 측정":
                self.events_log.append({"t": self.t, "event": ev})
                self._frame("센서 측정", "sensor")
                self.t += (r["소요(분)"] or 0.0) / 1440.0
        # 남은 부유 입자 포획
        deposit_particles(self.cs, self.tr.theta, self.tr.vmag, 0.0, self.rp, self.V, all_now=True,
                          d_p=self.c.p("inj_d_p"))

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
        c = self.cs.c
        pH = pH_of(c["ALK"][mask], c["Ca"][mask], c["SO4"][mask], c["NO3"][mask], self.pCO2, self.I_bg)
        return float(np.mean(pH))

    def _pH_outflow(self):
        g = self.g
        k = min(g.k_interface, g.nz) - 1
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
        return float(pH_of(np.array([ALK]), np.array([Ca]), np.array([SO4]), np.array([NO3]),
                           self.pCO2, self.I_bg)[0])

    def _Kreac(self, reac_mask0):
        if reac_mask0 is None or not reac_mask0.any():
            return np.nan
        _, Ksv, _ = self._material()
        k = Ksv[reac_mask0]
        return float(len(k) / np.sum(1.0 / np.maximum(k, 1e-30)))

    def _do_refill(self, rule, eta_now, n_done):
        method = rule.get("방식") or "펌프"
        rec = {"day": self.t - self.t_op0, "eta_before": eta_now, "method": method, "n": n_done + 1,
               "duration_min": float(rule.get("소요(분)") or 0.0)}
        sc = self.sc
        if method == "카트리지 교체":
            names = self._targets(rule, "반응재(고체)")
            acc = [n for n in names if sc.roles.get(n, {}).get("accessible", False) or sc.allow_refill_wo_opening]
            if not acc:
                rec["ok"] = False
                rec["why"] = "교체 가능한 반응재 객체가 없음(역할표 '교체 가능' 체크 필요)"
                return rec
            ids = [self.objects_by_name[n] for n in acc]
            cells = np.isin(self.obj_cell, ids) & (self.role_cell == ROLES.index("반응재(고체)"))
            rmix = self.c.ratio_S_CaCO3
            vs = 1.0 - self.phi_ref[cells]
            self.cs.S[cells] = vs * rmix / (rmix + 1) / VM["S"]
            self.cs.Cal[cells] = vs / (rmix + 1) / VM["CaCO3"]
            self.cs.Gyp[cells] = 0; self.cs.Bio[cells] = 0; self.cs.Inert[cells] = 0
            self.cs.ledger["S_inj"] += float(self.cs.S[cells].sum() * self.V)
            self.t += rec["duration_min"] / 1440.0
            rec["ok"] = True
            rec["targets"] = acc
        else:
            if not any(self.geom["roles_of"][o.name] == "주입 개구부" for o in sc.objects):
                rec["ok"] = False
                rec["why"] = "개구부가 없어 재보충 불가(R=0)"
                return rec
            r = self._inject(rule, operating=True, label=f"재보충 {n_done + 1}회")
            deposit_particles(self.cs, self.tr.theta, self.tr.vmag, 0.0, self.rp, self.V, all_now=True,
                              d_p=self.c.p("inj_d_p"))
            rec["ok"] = r.get("delivered_L", 0) > 0
            rec["delivered_L"] = r.get("delivered_L", 0)
            rec["targets"] = r.get("targets", [])
            if not rec.get("duration_min"):
                rec["duration_min"] = r.get("duration_min", 0.0)
        return rec

    def run_operation(self):
        c = self.c
        g = self.g
        self.t_op0 = self.t
        q = c.q_mm_day / 1000.0 / 86400.0
        conc = self._inflow_conc(True)
        st = self._steady(q) if c.flow_mode == "준정상" else self._transient(q, 86400.0)
        self._set_flow(st)
        _, Ksv0, _ = self._material()
        self.Ks_v0 = Ksv0.copy()
        L = self.cs.ledger
        # 수지 기준점(운영 시작)
        Vt = self.tr.theta * self.V
        L["N_store0"] = float((self.cs.c["NO3"] * Vt).sum())
        L["SO4_store0"] = float((self.cs.c["SO4"] * Vt).sum())
        L["Ca_store0"] = float((self.cs.c["Ca"] * Vt).sum())
        L["ALK_store0"] = float((self.cs.c["ALK"] * Vt).sum())
        L["S_solid0"] = float(self.cs.S.sum() * self.V)
        L["Gyp0"] = float(self.cs.Gyp.sum() * self.V)
        self.op_ledger0 = dict(L)
        reac0, _, _ = self._masks()
        self.reac0 = reac0
        self.mass0_g = reactive_mass_g(self.cs.S, self.cs.Cal, self.V)
        self.Keff0 = self.fv.constant_head_Keff(min(g.k_interface, g.nz))
        self.Keff_soil0 = self.Keff0
        self.Kreac0 = self._Kreac(reac0)
        self.diag = self._through_flow(reac0)
        Keff = self.Keff0
        snap_times = np.linspace(0, c.days, max(2, int(c.n_snapshots)))
        si = 0
        self._snapshot("운영 시작")
        si = 1
        last_flow = 0.0
        last_IF = (L["IF_N"], L["IF_W"], L["IF_SO4"])
        last_refill = -1e9
        n_refill = {id(r): 0 for r in getattr(self, "refill_rules", [])}
        Ks_last = self._material()[1]
        days = int(np.ceil(c.days))
        for d in range(1, days + 1):
            day_end = self.t_op0 + min(d, c.days)
            # 하루 진행(흐름 갱신 포함)
            while self.t < day_end - 1e-9:
                if self.stop():
                    raise StopRequested()
                dt_s = (day_end - self.t) * 86400.0
                if c.flow_mode == "과도":
                    dt_s = min(dt_s, max(c.flow_update_days, 0.05) * 86400.0)
                    st = self._transient(q, dt_s)
                    self._set_flow(st)
                self._advance(dt_s, conc)
            op_day = self.t - self.t_op0
            # 투수성 변화 → 흐름 재계산
            _, Ksv, _ = self._material()
            rel = np.max(np.abs(Ksv - Ks_last) / np.maximum(Ks_last, 1e-30)) if self.active.any() else 0
            if c.flow_mode == "준정상" and (rel > 0.05 or op_day - last_flow >= c.flow_update_days - 1e-9):
                self.fv.update_Ks(*self._material())
                st = self._steady(q)
                self._set_flow(st)
                last_flow = op_day
                Ks_last = Ksv.copy()
                Keff = self.fv.constant_head_Keff(min(g.k_interface, g.nz))
            elif c.flow_mode == "과도" and rel > 0.05:
                self.fv.update_Ks(*self._material())
                Ks_last = Ksv.copy()
                Keff = self.fv.constant_head_Keff(min(g.k_interface, g.nz))
            # 하루 지표
            dN = L["IF_N"] - last_IF[0]; dW = L["IF_W"] - last_IF[1]; dS = L["IF_SO4"] - last_IF[2]
            last_IF = (L["IF_N"], L["IF_W"], L["IF_SO4"])
            Cout = dN / dW * MW["N"] if dW > 0 else np.nan
            SO4o = dS / dW * MW["SO4"] if dW > 0 else np.nan
            eta = 1.0 - Cout / c.NO3_in_mgN if (dW > 0 and c.NO3_in_mgN > 0) else np.nan
            reac, around, below = self._masks()
            D = self.daily
            D["day"].append(op_day); D["eta"].append(eta); D["Cout_mgN"].append(Cout)
            D["SO4_out_mg"].append(SO4o); D["pH_out"].append(self._pH_outflow())
            D["pH_around"].append(self._pH_mean(around)); D["pH_below"].append(self._pH_mean(below))
            D["Keff"].append(Keff); D["Kreac"].append(self._Kreac(self.reac0))
            D["N_to_gw_g"].append(L["IF_N"] * MW["N"]); D["denit_g"].append(L["N_denit"] * MW["N"])
            aq = self.soil["zone"] & self.active
            D["gw_NO3_mgN"].append(float(self.cs.c["NO3"][aq].mean() * MW["N"]) if aq.any() else np.nan)
            D["water_out_L"].append(L["IF_W"] * 1e3)
            D["reactive_g"].append(reactive_mass_g(self.cs.S, self.cs.Cal, self.V))
            # 재보충 판단
            if self.sc is not None:
                for rule in getattr(self, "refill_rules", []):
                    if rule.get("_disabled"):
                        continue
                    trig = rule.get("트리거 η(%)")
                    intv = rule.get("간격(일)")
                    nmax = int(rule.get("최대 회차") or 99)
                    if n_refill[id(rule)] >= nmax:
                        continue
                    start_ok = self.eval_start_day is None or op_day >= self.eval_start_day - 1e-6
                    do = False
                    if trig is None and intv is None:
                        trig = 50.0
                    if trig is not None and start_ok and np.isfinite(eta) and eta * 100 < trig \
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
                            st = self._steady(q) if c.flow_mode == "준정상" else self._transient(q, 3600.0)
                            self._set_flow(st)
                            last_flow = op_day
                            Ks_last = self._material()[1]
                            Keff = self.fv.constant_head_Keff(min(g.k_interface, g.nz))
                        else:
                            self.warn.append(f"재보충 시도 실패({op_day:.0f}일): {rec.get('why', '')}")
                            if "불가" in rec.get("why", "") or "없음" in rec.get("why", ""):
                                rule["_disabled"] = True     # 구조상 불가능 → 이후 시도 안 함
            while si < len(snap_times) and op_day >= snap_times[si] - 1e-6:
                self._snapshot()
                si += 1
            self.progress(min(1.0, op_day / c.days), f"{op_day:.0f}/{c.days:.0f}일")

    def _through_flow(self, reac):
        """반응재 영역으로 들어가는 물의 양 / 반응재 평면 투영 면적에 내리는 비."""
        if reac is None or not reac.any():
            return {"through_ratio": np.nan}
        g = self.g
        R = reac.reshape(g.shape)
        inflow = 0.0
        for ax, F in enumerate((self.st.Fx, self.st.Fy, self.st.Fz)):
            sa = [slice(None)] * 3; sb = [slice(None)] * 3
            sa[ax] = slice(0, -1); sb[ax] = slice(1, None)
            a = R[tuple(sa)]; b = R[tuple(sb)]
            inflow += float(np.clip(F, 0, None)[~a & b].sum() + np.clip(-F, 0, None)[a & ~b].sum())
        inflow += float(np.clip(self.st.Ftop, 0, None)[R[:, :, 0]].sum())
        foot = R.any(axis=2).sum() * g.dx * g.dy
        rain = self.c.q_mm_day / 1000.0 / 86400.0 * foot
        return {"through_ratio": inflow / rain if rain > 0 else np.nan, "inflow_m3s": inflow,
                "footprint_m2": float(foot)}

    def mass_balance(self):
        L = self.cs.ledger
        L0 = self.op_ledger0
        Vt = self.tr.theta * self.V
        cN = float((self.cs.c["NO3"] * Vt).sum())
        out = {}
        dN_in = L["N_in"] - L0["N_in"]; dN_out = L["N_out"] - L0["N_out"]; den = L["N_denit"] - L0["N_denit"]
        errN = (cN - L["N_store0"]) - (dN_in - dN_out - den)
        out["N_err_rel"] = errN / max(dN_in, L["N_store0"], 1e-30)
        SO4 = float((self.cs.c["SO4"] * Vt).sum()); Ssol = float(self.cs.S.sum() * self.V)
        Gyp = float(self.cs.Gyp.sum() * self.V)
        dS_in = L["S_in"] - L0["S_in"]; dS_out = L["S_out"] - L0["S_out"]; dS_inj = L["S_inj"] - L0["S_inj"]
        tot_now = SO4 + Ssol + Gyp
        tot0 = L["SO4_store0"] + L["S_solid0"] + L["Gyp0"]
        errS = (tot_now - tot0) - (dS_in - dS_out + dS_inj)
        out["S_err_rel"] = errS / max(tot0 - L["S_solid0"] + dS_in, 1e-30)
        ALK = float((self.cs.c["ALK"] * Vt).sum())
        dA = (L["ALK_in"] - L0["ALK_in"]) - (L["ALK_out"] - L0["ALK_out"]) + (L["ALK_cal"] - L0["ALK_cal"]) \
            - (L["ALK_denit"] - L0["ALK_denit"])
        errA = (ALK - L["ALK_store0"]) - dA
        out["ALK_err_rel"] = errA / max(abs(L["ALK_in"] - L0["ALK_in"]), abs(L["ALK_store0"]), 1e-30)
        Ca = float((self.cs.c["Ca"] * Vt).sum())
        dC = (L["Ca_in"] - L0["Ca_in"]) - (L["Ca_out"] - L0["Ca_out"]) + (L["Cal_diss"] - L0["Cal_diss"]) \
            - (L["Gyp_prec"] - L0["Gyp_prec"])
        out["Ca_err_rel"] = ((Ca - L["Ca_store0"]) - dC) / max(L["Ca_in"] - L0["Ca_in"], L["Ca_store0"], 1e-30)
        return out

    def result(self, wall):
        g = self.g
        D = {k: np.array(v, dtype=float) for k, v in self.daily.items()}
        sc = self.sc
        return {
            "name": sc.name if sc else "대조군(구조물 없음)",
            "is_control": sc is None,
            "mechanism": sc.mechanism if sc else "반응재를 넣지 않은 같은 흙수조.",
            "install_text": sc.install_text if sc else "",
            "events_table": ie.clean_rows(sc.events) if sc else [],
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
            "reac0": self.reac0.reshape(g.shape),
            "mass0_g": self.mass0_g,
            "Keff0": self.Keff0, "Kreac0": self.Kreac0,
            "ledger": dict(self.cs.ledger),
            "mass_balance": self.mass_balance(),
            "diag": self.diag,
            "warnings": list(dict.fromkeys(self.warn)),
            "t_op0": self.t_op0,
            "flow_solves": self.flow_solves,
            "wall_s": wall,
            "placement": vars(sc.placement).copy() if sc else None,
            "object_names": [o.name for o in sc.objects] if sc else [],
            "roles_of": self.geom["roles_of"] if self.geom else {},
            "open_state": dict(self.open_state),
        }


def run_scenario(c: Common, sc: Optional[Scenario], progress=None, stop=None, eval_start_day=None):
    t0 = time.time()
    sim = Sim(c, sc, progress, stop, eval_start_day)
    if sc is not None and sim.geom["errors"]:
        raise ValueError("\n".join(sim.geom["errors"]))
    sim.run_installation()
    sim.run_operation()
    return sim.result(time.time() - t0)


def control_breakthrough_day(ctrl_res, frac=0.5, c_in=None):
    D = ctrl_res["daily"]
    cin = c_in or 1.0
    ok = np.where(np.isfinite(D["Cout_mgN"]) & (D["Cout_mgN"] >= frac * cin))[0]
    return round(float(D["day"][ok[0]]), 6) if len(ok) else None


def eval_start_from(ctrl_res, c_in):
    """η 평가 시작일: 대조군 유출 농도가 유입의 90%에 도달한 날(없으면 50%, 그것도 없으면 None)."""
    d = control_breakthrough_day(ctrl_res, 0.9, c_in)
    if d is None:
        d = control_breakthrough_day(ctrl_res, 0.5, c_in)
    return d
