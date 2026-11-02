"""공통 상수·기본값 레지스트리.

모든 물리 파라미터에 출처 또는 '가정' 라벨을 붙인다(정직성 규칙).
라벨: "문헌" = 출처 있음 / "가정" = 임의 설정 / "보정" = 데이터에 맞춰 조정한 값.
"""
from __future__ import annotations

import re

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

# ───────────────────────────── 물리 상수(25°C) ─────────────────────────────
G = 9.81
RHO_W = 1000.0
MW = {"N": 14.007, "S": 32.06, "SO4": 96.06, "Ca": 40.078, "CaCO3": 100.09, "gyp": 172.17}
RHO_SOLID = {"S": 2070.0, "CaCO3": 2710.0, "gyp": 2320.0}          # kg/m3 (화학 편람값)
VM = {k: MW[k if k != "gyp" else "gyp"] * 1e-3 / RHO_SOLID[k] for k in ("S", "CaCO3", "gyp")}  # m3/mol

LOGK = {   # PHREEQC phreeqc.dat (Parkhurst & Appelo 2013), 25°C
    "K1": -6.351, "K2": -10.329, "KH": -1.468, "Kw": -14.0,
    "Ksp_cal": -8.48, "Ksp_gyp": -4.58,
    # 이온쌍(phreeqc.dat 3.7.x): Ca+2 + SO4-2 = CaSO4 ; Ca+2 + CO3-2 = CaCO3 ; Ca+2 + HCO3- = CaHCO3+ (11.435−10.329)
    "CaSO4": 2.25, "CaCO3": 3.224, "CaHCO3": 1.106,
}

# ───────────────────────────── 파라미터 레지스트리 ─────────────────────────
# key: (값, 단위, 라벨, 설명/출처)
REGISTRY: Dict[str, tuple] = {
    "stoich_SO4": (1.10, "mol SO4/mol NO3", "문헌",
                   "Batchelor & Lawrence (1978) 황 독립영양 탈질 양론. NO3-N 1 mg당 SO4 7.54 mg(지표 문서 이론값)과 일치"),
    "stoich_H": (1.28, "eq 산/mol NO3", "문헌",
                 "Batchelor & Lawrence (1978). NO3-N 1 mg당 알칼리도 4.57 mg as CaCO3 소비와 일치"),
    "stoich_S": (1.10, "mol S/mol NO3", "문헌", "Batchelor & Lawrence (1978)"),
    "k_half": (5.0, "mg-N^½/L^½/h", "문헌",
               "반-차수 속도상수. 선행 SLAD 칼럼(알칼리도 850 mg/L) 보고 범위 4.36~6.91의 중간값 — 참고 시뮬 A 문서 인용, 원 논문 대조 필요"),
    "S_ref_frac": (0.45, "m3 S/m3 충전층", "가정",
                   "속도상수를 측정한 충전층의 황 부피분율(공극 0.4, 황:석회석 3:1 가정). 반응 속도를 황 함량 비로 환산할 때 기준"),
    "pH_low": (6.2, "-", "가정", "이 pH 아래에서 탈질 속도가 절반 이하로 떨어짐(Sengupta & Ergas 계열 경향 참고)"),
    "pH_high": (9.0, "-", "가정", "이 pH 위에서 탈질 속도 감소"),
    "theta_T": (1.07, "-", "가정", "온도 보정 계수(아레니우스형), 기준 22°C"),
    "T_ref": (22.0, "°C", "가정", "속도상수 기준 온도"),
    "cal_d_p": (2.0e-3, "m", "가정", "충전 석회석 입경(비표면적 6/d 계산)"),
    "inj_d_p": (2.0e-4, "m", "가정", "주입 슬러리 입자 입경"),
    "pCO2": (-2.0, "log10 atm", "가정", "토양 공기 CO2 분압(일반 범위 10^-3 ~ 10^-1.5). 깊이별 모드에서는 지표 값"),
    "pCO2_bottom": (-1.7, "log10 atm", "가정", "깊이별 pCO2 모드: 흙수조 바닥 값(지표와 선형 보간, B4)"),
    "pCO2_reac": (-1.5, "log10 atm", "가정", "깊이별 pCO2 모드: 반응재 셀 값(미생물 호흡·탈질로 높음 가정)"),
    "I_bg": (0.005, "mol/L", "가정", "배경 이온강도(Davies 활동도 계산용)"),
    "bio_density": (25.0, "kg/m3", "가정",
                    "생물막 건조밀도. 탈질 1 mol당 생물막 부피 = 0.08 mol 세포 × 113 g/mol ÷ 밀도. 불확실 범위 10~100 kg/m3"),
    "bio_decay": (0.05, "1/day", "가정", "생물막 감쇠율"),
    "bio_det": (0.02, "1/m", "가정",
                "생물막 유속 탈착 계수: 공극수가 1 m 흐를 때 떨어져 나가는 비율(B8). 기질 확산 제한은 미구현"),
    "lambda_filt": (1000.0, "1/m", "가정",
                    "주입 입자 포획(여과) 계수. 콜로이드 여과 이론(Yao et al. 1971) λ=3(1−φ)ηα/(2d_c) 규모 추정. "
                    "v2 기본 30 1/m은 낙관적(거의 안 걸러짐) — B7. 불확실 범위 300~3000"),
    "strain_ratio": (0.154, "-", "문헌",
                     "입자/토양입자 지름비가 이 값을 넘으면 입자가 토양 공극에 못 들어가 개구부에 필터 케이크로 쌓임. "
                     "Herzig et al. (1970), McDowell-Boyer et al. (1986) 정리값 — 원 논문 대조 필요"),
    "K_open": (1.0e-2, "m/s", "가정",
               "주입 개구부(스크린·관) 셀의 투수계수. v2는 토양 K를 그대로 써서 주입이 실제보다 어렵거나 쉽게 나올 수 있었음(A15)"),
    "colloid_d_p": (1.0e-6, "m", "가정", "콜로이드·나노 황 입경(시약 형태 e)"),
    "lambda_colloid": (50.0, "1/m", "가정", "콜로이드 포획 계수(시약 형태 e) — 입자 슬러리보다 작게 가정"),
    "k_O2": (1.0e-6, "mol/m3/s", "가정", "O2에 의한 황 표면 산화 속도(황 부피분율 기준값일 때). 초기 O2 단계(B5)"),
    "K_O2": (0.01, "mol/m3", "가정", "황 산화 O2 반포화 농도(≈0.3 mg/L)"),
    "K_I_O2": (0.006, "mol/m3", "가정", "O2에 의한 탈질 억제 상수(≈0.2 mg/L)"),
    "DO_in": (8.0, "mg/L", "가정", "유입수 용존산소(공기 포화 근사)"),
    "alpha_L": (0.01, "m", "가정", "종분산 길이(흙수조 규모)"),
    "alpha_T_ratio": (0.1, "-", "가정", "횡분산 길이 / 종분산 길이"),
    "Dm": (1.7e-9, "m2/s", "문헌", "질산염 자유수 확산계수(25°C, Li & Gregory 1974)"),
    "pH_ok_window": (0.5, "-", "문헌", "지표 문서: |ΔpH|<0.5 = 적정 완충"),
    "SO4_increase_thr": (0.1, "mol/m3", "가정", "V_eff 계산 시 'SO4 증가 확인' 판정 문턱(0.1 mM ≈ 9.6 mg/L)"),
    "reactive_thr_frac": (0.01, "-", "가정", "확산반경 r 판정: 황 부피분율이 충전층 대비 1% 이상인 셀"),
}


def reg(key):
    return REGISTRY[key][0]


# 토성별 대표 입경 d50 [m] — 입자 막힘(straining) 판정용. 모두 '가정'(USDA 토성 삼각도 중앙 조성에서 어림).
D50_TEXTURE = {
    "모래(Sand)": 3.0e-4, "양질사토(Loamy sand)": 2.0e-4, "사양토(Sandy loam)": 1.0e-4, "양토(Loam)": 4.0e-5,
    "미사질양토(Silt loam)": 2.5e-5, "미사(Silt)": 1.5e-5, "사질식양토(Sandy clay loam)": 6.0e-5,
    "식양토(Clay loam)": 2.0e-5, "미사질식양토(Silty clay loam)": 1.0e-5, "사질식토(Sandy clay)": 3.0e-5,
    "미사질식토(Silty clay)": 5.0e-6, "식토(Clay)": 3.0e-6,
}


# ───────────────────────────── 토성(Carsel & Parrish 1988) ─────────────────
# θr, θs, α[1/cm], n, Ks[cm/day]
_CP = {
    "모래(Sand)": (0.045, 0.43, 0.145, 2.68, 712.8),
    "양질사토(Loamy sand)": (0.057, 0.41, 0.124, 2.28, 350.2),
    "사양토(Sandy loam)": (0.065, 0.41, 0.075, 1.89, 106.1),
    "양토(Loam)": (0.078, 0.43, 0.036, 1.56, 24.96),
    "미사질양토(Silt loam)": (0.067, 0.45, 0.020, 1.41, 10.80),
    "미사(Silt)": (0.034, 0.46, 0.016, 1.37, 6.00),
    "사질식양토(Sandy clay loam)": (0.100, 0.39, 0.059, 1.48, 31.44),
    "식양토(Clay loam)": (0.095, 0.41, 0.019, 1.31, 6.24),
    "미사질식양토(Silty clay loam)": (0.089, 0.43, 0.010, 1.23, 1.68),
    "사질식토(Sandy clay)": (0.100, 0.38, 0.027, 1.23, 2.88),
    "미사질식토(Silty clay)": (0.070, 0.36, 0.005, 1.09, 0.48),
    "식토(Clay)": (0.068, 0.38, 0.008, 1.09, 4.80),
}
SOIL_TEXTURES = {k: {"theta_r": v[0], "theta_s": v[1], "alpha": v[2] * 100.0, "n": v[3],
                     "Ks": v[4] / 100.0 / 86400.0} for k, v in _CP.items()}
SOIL_SOURCE = "Carsel & Parrish (1988) Water Resour. Res. 24(5) 토성별 평균값"


# ───────────────────────────── 객체 역할 ─────────────────────────────────
ROLES = ["반응재(고체)", "주입 개구부", "관 내부(빈 공간)", "불투수", "저투수 실링", "투수 충전재", "토양"]

ROLE_DEFAULTS = {
    #               K[m/s], 공극률, 반응, α[1/m], n,   우선순위, 색
    "반응재(고체)":     dict(K=5e-4, phi=0.40, reactive=True, alpha=14.5, n=2.68, prio=4, color="#E3B505"),
    "주입 개구부":      dict(K=1e-2, phi=0.90, reactive=False, alpha=50.0, n=3.0, prio=7, color="#D7263D"),
    "관 내부(빈 공간)": dict(K=1e-2, phi=0.95, reactive=False, alpha=50.0, n=3.0, prio=2, color="#A9D6E5"),
    "불투수":          dict(K=0.0, phi=0.0, reactive=False, alpha=1.0, n=1.1, prio=6, color="#4A4E57"),
    "저투수 실링":      dict(K=1e-10, phi=0.45, reactive=False, alpha=0.5, n=1.09, prio=5, color="#7B5E57"),
    "투수 충전재":      dict(K=1e-4, phi=0.38, reactive=False, alpha=14.5, n=2.68, prio=3, color="#C9B79C"),
    "토양":            dict(K=None, phi=None, reactive=False, alpha=None, n=None, prio=1, color="#8B6F47"),
}
ROLE_SOURCE = ("역할별 K·공극률·vG 값은 가정(충전층·벤토나이트·모래 일반 범위). "
               "불투수는 계산에서 제외(흐름·물질 이동 없음). 주입 개구부는 기본 K=1e-2 m/s, 공극률 0.9(스크린·관 내부를 큰 K의 "
               "다공질로 근사 — 가정, A15). 토양 역할은 빈칸이면 토양 값을 그대로 씀. 관 내부 자유수(압력·유량 연속 조건)는 미구현.")

_KEYWORDS = [
    ("주입 개구부", ["screen", "slot", "port", "opening", "inlet", "nozzle", "개구", "포트", "스크린", "주입구"]),
    ("반응재(고체)", ["reactive", "media", "slad", "sulfur", "sulphur", "limestone", "cartridge",
                    "반응", "황", "석회", "카트리지"]),
    ("저투수 실링", ["bentonite", "seal", "grout", "clay", "벤토", "실링", "그라우트", "점토"]),
    ("관 내부(빈 공간)", ["void", "interior", "bore_space", "lumen", "빈공간", "관내부", "내부공간"]),
    ("토양", ["soil", "backfill", "surface_fill", "native", "토양", "되메움"]),
    ("투수 충전재", ["sand", "gravel", "pack", "filter", "모래", "자갈", "충전"]),
    ("불투수", ["shell", "casing", "pipe", "wall", "floor", "pad", "concrete", "leg", "point", "cap",
               "plate", "steel", "pvc", "케이싱", "벽", "바닥", "콘크리트", "관", "다리"]),
]


def suggest_role(name: str) -> str:
    """이름 키워드로 역할 제안. 영문은 단어(토큰) 단위로 비교해 'support'가 'port'로 잡히는 오류를 막는다."""
    low = name.lower()
    tokens = [t for t in re.split(r"[^a-z0-9가-힣]+", low) if t]
    for role, keys in _KEYWORDS:
        for k in keys:
            if re.search(r"[가-힣]", k):
                if k in low:
                    return role
            elif "_" in k:
                if k in low:
                    return role
            elif any(t == k or (t.startswith(k) and len(k) >= 4) for t in tokens):
                return role
    return "불투수"


# ───────────────────────────── 공통 조건 ─────────────────────────────────
@dataclass
class SoilLayer:
    top: float          # 깊이 m
    bottom: float
    theta_r: float
    theta_s: float
    alpha: float        # 1/m
    n: float
    Ks: float           # m/s (수평)
    aniso: float = 1.0  # Kz/Kx


@dataclass
class Common:
    # 흙수조
    Lx: float = 0.50
    Ly: float = 0.50
    H_soil: float = 0.70         # 토양층(불포화대) 깊이 m
    H_aq: float = 0.15           # 아래 포화대(지하수층) 두께 m (0이면 없음)
    wt_depth: Optional[float] = None   # 지하수면 깊이 m (None이면 토양층 바닥)
    gw_velocity: float = 0.0     # 지하수 다르시 유속 m/day (+x 방향)
    bottom_bc: str = "지하수 수위"   # "지하수 수위" | "누수(배수 저항)" | "자유배수"
    leak_thick: float = 0.05         # 누수 하단: 배수층 두께 m (가정)
    leak_K: float = 1.0e-4           # 누수 하단: 배수층 K m/s (가정)
    grid_preset: str = "보통"
    nx: int = 34
    ny: int = 34
    nz: int = 56
    # 토양
    soil_mode: str = "간단"
    texture: str = "사양토(Sandy loam)"
    Ks_override: Optional[float] = None    # m/s
    init_saturation: float = 0.5           # 초기 포화도(0~1)
    layers: List[SoilLayer] = field(default_factory=list)
    aniso: float = 1.0
    aq_texture: str = "모래(Sand)"
    # 유입수
    q_mm_day: float = 5.0
    rain_series: List[List[float]] = field(default_factory=list)   # [[시작일, mm/일], ...] 계단형. 비면 q_mm_day 일정
    et_mm_day: float = 0.0           # 잠재 증발산 mm/일(지표에서 빠짐, 용질은 남음)
    h_crit: float = -100.0           # 증발 한계 압력수두 m
    NO3_in_mgN: float = 50.0
    SO4_in_mg: float = 10.0
    Ca_in_mg: float = 20.0
    alk_in_mgCaCO3: float = 50.0
    gw_NO3_mgN: float = 0.0
    soil_NO3_mgN: float = 0.0
    soil_pH0: float = 7.0
    # 반응재
    ratio_S_CaCO3: float = 3.0
    packed_porosity: float = 0.40
    temperature: float = 22.0
    # 기간·수치
    days: float = 365.0
    flow_mode: str = "과도"         # "과도"(기본, 매 단계 Richards) | "준정상"(빠른 근사)
    flow_update_days: float = 10.0  # 준정상 재계산 주기
    flow_dt_max_h: float = 6.0      # 과도 흐름 최대 시간 간격(시간)
    flow_dt_min_s: float = 0.5      # 이보다 작아져도 수렴 못하면 FAILED
    calcite_mode: str = "속도론"    # or "평형"
    chem_engine: str = "PHREEQC"    # "PHREEQC"(엔진 사용) | "내장"(빠름, 이온쌍 포함)
    chem_dt_h: float = 2.0          # PHREEQC 반응 계산 간격(시간)
    splitting: str = "Lie"          # "Lie" | "Strang"(B3)
    pCO2_mode: str = "균일"          # "균일" | "깊이별"(B4)
    o2_stage: bool = False          # 초기 O2 소비 단계(B5)
    chem_mode: str = "SLAD"         # "SLAD" | "추적자(반응 없음)" | "1차 감쇠"(대체품 모드)
    k1_decay_day: float = 0.1       # 1차 감쇠 모드 속도상수 1/일
    inject_model: str = "과도 Richards"   # 또는 "중첩 근사(빠름)"(B1)
    straining: bool = True          # 입자/토양 입경비로 필터 케이크 판정(B7)
    dt_factor: float = 1.0          # 이송 시간 간격 배율(시간 간격 독립성 점검용, 0.5 = 절반)
    heterogeneity: Dict[str, float] = field(default_factory=dict)   # {"sigma_lnK", "corr_len", "seed"} 토양 불균질성
    phi_reac_override: Optional[float] = None    # 반응재 초기 공극률 일괄 지정(불확실성 분석용)
    calibrated: Dict[str, float] = field(default_factory=dict)      # 실험 보정으로 정한 파라미터('보정' 라벨)
    n_snapshots: int = 30
    light: bool = False
    redistribute_runoff: bool = True   # 구조물 덮개 등에서 생긴 지표 유출을 열린 지표로 재분배(가정)
    disturb_thick: float = 0.0     # 삽입 교란대 두께 m
    disturb_Kmult: float = 1.0     # 교란대 K 배율
    params: Dict[str, float] = field(default_factory=dict)   # REGISTRY 덮어쓰기

    def p(self, key):
        return self.params.get(key, REGISTRY[key][0])

    def to_dict(self):
        d = asdict(self)
        return d

    @staticmethod
    def from_dict(d):
        d = dict(d)
        d["layers"] = [SoilLayer(**l) for l in d.get("layers", [])]
        return Common(**{k: v for k, v in d.items() if k in Common.__dataclass_fields__})


GRID_PRESETS = {"가벼움": (22, 22, 36), "보통": (32, 32, 52), "정밀": (46, 46, 76)}
