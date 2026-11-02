"""PHREEQC(USGS, 퍼블릭 도메인) 지화학 엔진 연결 — IPhreeqc 공유 라이브러리를 ctypes로 직접 호출.

받은 코드(IPhreeqc 라이브러리, phreeqc.dat)는 원본을 그대로 쓰고, 우리 연구에 필요한 정의만
이 파일에서 PHREEQC 입력 문자열로 덧붙인다(C++ 수정 없음):

  1) 질산염을 산화환원과 분리한 보존성 성분 'Nit'(Nit-)로 정의
     — phreeqc.dat의 N(5)/N(0) 산화환원 평형이 탈질을 '평형'으로 처리해 버리는 것을 막기 위함.
  2) RATES 'SLAD'   : 반-차수 독립영양 탈질 속도식(BASIC). 황 잔량·pH·온도 보정 포함.
     KINETICS 반응식 = HNit −1, H2SO4 +s (전하 중성). 산 생성 = 2s−1 eq/mol (s=1.10 → 1.20 eq).
  3) RATES 'Calcite_c': PWP 속도식(Plummer et al. 1978). phreeqc.dat의 Calcite 속도식과 같은 식이지만
     k3가 25 °C에서 끊기는 것을 연속이 되게 고친 판(알려진 문제 A5).
  4) EQUILIBRIUM_PHASES: CO2(g)(셀별 pCO2 고정), Gypsum(평형 침전/용해), 선택 시 Calcite 평형.
  5) 이온쌍(CaSO4°, CaCO3°, CaHCO3⁺ 등)과 활동도(phreeqc.dat의 Davies/WATEQ Debye-Hückel)는 DB 값 사용.

라이브러리 탐색 순서: ① 환경변수 WATERPRIZE_IPHREEQC ② pip 패키지 phreeqpy 안의 라이브러리
③ 이 폴더 engine_bin/ 안의 라이브러리. 찾지 못하면 available()=False 이고 앱은 내장 화학(이온쌍 포함)으로 돌아간다.
"""
from __future__ import annotations

import ctypes
import os
import sys
from typing import Dict, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "engine_data", "phreeqc.dat")

EXTRA_DEFS = r"""
SOLUTION_MASTER_SPECIES
    Nit     Nit-    0   Nit     14.0067
SOLUTION_SPECIES
    Nit- = Nit-
    log_k 0
PHASES
    HNit
    HNit = H+ + Nit-
    log_k 10
RATES
SLAD
-start
10 REM PARM: 1 k_half 2 fS_coef 3 pH_low 4 pH_high 5 f0 6 fT
20 c = TOT("Nit") * 14006.7
30 IF (c <= 1e-12) OR (M <= 0) THEN GOTO 200
40 fs = (M * PARM(2)) ^ (2/3)
50 ph = -LA("H+")
60 fp = 1 / (1 + 10 ^ (PARM(3) - ph)) / (1 + 10 ^ (ph - PARM(4))) / PARM(5)
70 rate = PARM(1) / 3600 * SQRT(c) / 14006.7 * fs * fp * PARM(6)
80 moles = rate * TIME
90 IF moles > TOT("Nit") THEN moles = TOT("Nit")
100 IF moles > M THEN moles = M
200 SAVE moles
-end
Calcite_c
-start
10 REM PARM: 1 specific area m2 per mol calcite
20 si_cc = SI("Calcite")
30 IF (M <= 0) AND (si_cc < 0) THEN GOTO 200
40 tkv = TK
50 k1 = 10 ^ (0.198 - 444.0 / tkv)
60 k2 = 10 ^ (2.84 - 2177.0 / tkv)
70 k3 = 10 ^ (-5.86 - 317.0 / tkv)
80 IF tkv > 298.15 THEN k3 = 10 ^ (-1.1 - 1737.0 / tkv) * 10 ^ ((-5.86 - 317.0 / 298.15) - (-1.1 - 1737.0 / 298.15))
90 rf = (k1 * ACT("H+") + k2 * ACT("CO2") + k3 * ACT("H2O")) * 10
100 area = PARM(1) * M
110 rate = area * rf * (1 - 10 ^ (2/3 * si_cc))
120 moles = rate * TIME
130 IF moles > M THEN moles = M
200 SAVE moles
-end
"""


def _candidate_libs():
    env = os.environ.get("WATERPRIZE_IPHREEQC")
    if env:
        yield env
    try:
        import phreeqpy  # noqa: F401
        base = os.path.join(os.path.dirname(phreeqpy.__file__), "iphreeqc")
        if sys.platform == "win32":
            yield os.path.join(base, "phreeqc3", "IPhreeqc-3.7.3.dll")
            yield os.path.join(base, "IPhreeqc.dll")
        elif sys.platform.startswith("linux"):
            yield os.path.join(base, "phreeqc3", "libiphreeqc-3.7.3.so")
            yield os.path.join(base, "libiphreeqc.so.0.0.0")
        elif sys.platform == "darwin":
            yield os.path.join(base, "phreeqc3", "libiphreeqc-3.7.3.dylib")
    except Exception:
        pass
    b = os.path.join(HERE, "engine_bin")
    if sys.platform == "win32":
        yield os.path.join(b, "IPhreeqc-3.7.3.dll")
    elif sys.platform.startswith("linux"):
        yield os.path.join(b, "libiphreeqc.so")
    elif sys.platform == "darwin":
        yield os.path.join(b, "libiphreeqc.dylib")


class PhreeqcError(RuntimeError):
    pass


class Phreeqc:
    """IPhreeqc 인스턴스 하나(데이터베이스 + 우리 정의를 한 번 적재)."""

    def __init__(self, lib_path: Optional[str] = None, db_path: Optional[str] = None):
        tried = []
        lib = None
        for p in ([lib_path] if lib_path else list(_candidate_libs())):
            if not p or not os.path.exists(p):
                tried.append(f"{p} (없음)")
                continue
            try:
                lib = ctypes.CDLL(p)
                self.lib_path = p
                break
            except OSError as e:
                tried.append(f"{p} ({e})")
        if lib is None:
            raise PhreeqcError("IPhreeqc 라이브러리를 찾지 못함: " + "; ".join(tried))
        ci, cs = ctypes.c_int, ctypes.c_char_p
        for name, args, res in (("CreateIPhreeqc", [], ci), ("DestroyIPhreeqc", [ci], ci),
                                ("LoadDatabaseString", [ci, cs], ci), ("RunString", [ci, cs], ci),
                                ("GetErrorString", [ci], cs), ("SetSelectedOutputStringOn", [ci, ci], ci),
                                ("GetSelectedOutputString", [ci], cs), ("SetOutputFileOn", [ci, ci], ci),
                                ("SetErrorFileOn", [ci, ci], ci), ("SetLogFileOn", [ci, ci], ci),
                                ("SetSelectedOutputFileOn", [ci, ci], ci), ("GetVersionString", [], cs)):
            try:
                f = getattr(lib, name)
            except AttributeError:
                if name == "GetVersionString":
                    continue
                raise PhreeqcError(f"IPhreeqc 함수 {name} 없음")
            f.argtypes = args
            f.restype = res
        self.lib = lib
        self.id = lib.CreateIPhreeqc()
        if self.id < 0:
            raise PhreeqcError("CreateIPhreeqc 실패")
        for fn in ("SetOutputFileOn", "SetErrorFileOn", "SetLogFileOn", "SetSelectedOutputFileOn"):
            getattr(lib, fn)(self.id, 0)
        lib.SetSelectedOutputStringOn(self.id, 1)
        self.db_path = db_path or DB_PATH
        with open(self.db_path, "rb") as fh:
            db = fh.read()
        n = lib.LoadDatabaseString(self.id, db)
        if n != 0:
            raise PhreeqcError("데이터베이스 적재 오류: " + self._err())
        lib.SetSelectedOutputStringOn(self.id, 1)      # DB 적재 뒤에 켜야 문자열 출력이 남는다(3.7.3에서 확인)
        try:
            self.version = lib.GetVersionString().decode("ascii", "replace")
        except Exception:
            self.version = "?"
        self.calls = 0

    def _err(self):
        s = self.lib.GetErrorString(self.id)
        return s.decode("utf-8", "replace") if s else ""

    def run(self, text: str) -> str:
        n = self.lib.RunString(self.id, text.encode("ascii", "replace"))
        self.calls += 1
        if n != 0:
            raise PhreeqcError(self._err()[:2000])
        s = self.lib.GetSelectedOutputString(self.id)
        return s.decode("ascii", "replace") if s else ""

    def close(self):
        try:
            self.lib.DestroyIPhreeqc(self.id)
        except Exception:
            pass


_INSTANCE: Dict[str, object] = {}


def get_engine():
    """한 프로세스에 하나. 실패하면 예외 메시지를 기록해 두고 None."""
    if "eng" in _INSTANCE:
        return _INSTANCE["eng"]
    try:
        eng = Phreeqc()
        _INSTANCE["eng"] = eng
        _INSTANCE["error"] = ""
    except Exception as e:      # noqa: BLE001
        _INSTANCE["eng"] = None
        _INSTANCE["error"] = str(e)
    return _INSTANCE["eng"]


def available() -> bool:
    return get_engine() is not None


def status() -> Dict[str, str]:
    eng = get_engine()
    if eng is None:
        return {"ok": "아니오", "error": _INSTANCE.get("error", ""), "lib": "", "version": "", "db": DB_PATH}
    return {"ok": "예", "error": "", "lib": eng.lib_path, "version": eng.version, "db": eng.db_path}


_COLS = ("cell", "Ca", "S6", "Nit", "Alk", "pH", "SLAD_m", "Cal_m", "Gyp_m", "SI_cal", "SI_gyp", "dSLAD", "mu", "Cal_eq", "mass_w")


def react_cells(cells: Dict[str, np.ndarray], dt: float, rp, pCO2_log: np.ndarray, calcite_mode: str,
                eng: Optional[Phreeqc] = None, chunk: int = 400, fT_cells=None) -> Dict[str, np.ndarray]:
    """셀별 회분(batch) 반응 dt 초. 입력·출력 단위 = mol/L 물(=mol/kgw로 근사), 알칼리도 eq/L.

    cells: NO3, SO4, Ca, ALK, Ssol, Cal, Gyp (각 셀 물 1 L당 mol), theta(셀 체적함수비),
           aCal(석회석 비표면적 m2/m3 석회석).
    반환: 같은 키(반응 후) + pH, dN(탈질 mol/L), SI_cal, SI_gyp, mu.
    """
    from params import VM
    eng = eng or get_engine()
    if eng is None:
        raise PhreeqcError("PHREEQC 엔진 없음")
    n = len(cells["NO3"])
    s = float(rp.stoich_SO4)
    f0 = 1.0 / (1.0 + 10 ** (rp.pH_low - 7.5)) / (1.0 + 10 ** (7.5 - rp.pH_high))
    fT = rp.theta_T ** (rp.T - rp.T_ref)
    out = {k: np.zeros(n) for k in ("NO3", "SO4", "Ca", "ALK", "Ssol", "Cal", "Gyp", "pH", "dN", "SI_cal", "SI_gyp", "mu")}
    for s0 in range(0, n, chunk):
        idx = np.arange(s0, min(n, s0 + chunk))
        lines = []
        for j, i in enumerate(idx, start=1):
            th = float(cells["theta"][i])
            Ssol = max(float(cells["Ssol"][i]), 0.0)
            Mslad = Ssol / max(rp.stoich_S, 1e-12)
            # fS = (S_bulk·VM_S / S_ref)^(2/3),  S_bulk[mol/m3] = M_slad·stoich_S·θ·1000
            fs_coef = rp.stoich_S * th * 1000.0 * VM["S"] / max(rp.S_ref, 1e-12)
            Cal = max(float(cells["Cal"][i]), 0.0)
            a_mol = float(cells["aCal"][i]) * VM["CaCO3"]          # m2 / mol 석회석
            lines.append(
                f"SOLUTION {j}\n -temp {rp.T:.3f}\n -units mmol/kgw\n pH 7\n"
                f" Ca {max(cells['Ca'][i], 0) * 1e3:.10g}\n S(6) {max(cells['SO4'][i], 0) * 1e3:.10g}\n"
                f" Nit {max(cells['NO3'][i], 0) * 1e3:.10g}\n Alkalinity {cells['ALK'][i] * 1e3:.10g}\n"
                f"EQUILIBRIUM_PHASES {j}\n CO2(g) {pCO2_log[i]:.6f} 10\n Gypsum 0 {max(cells['Gyp'][i], 0):.10g}\n"
                + (f" Calcite 0 {Cal:.10g}\n" if calcite_mode == "평형" else "")
                + f"KINETICS {j}\nSLAD\n -formula HNit -1 H2SO4 {s:.6g}\n -m {Mslad:.10g}\n -m0 {Mslad:.10g}\n"
                f" -parms {rp.k_half:.8g} {fs_coef:.8g} {rp.pH_low:.6g} {rp.pH_high:.6g} {f0:.8g} "
                f"{fT * (1.0 if fT_cells is None else float(fT_cells[i])):.8g}\n"
                + (f"Calcite_c\n -formula CaCO3 1\n -m {Cal:.10g}\n -parms {a_mol:.8g}\n" if calcite_mode != "평형" else "")
                + f" -steps {dt:.8g}\n -step_divide 1\n -runge_kutta 3\n")
        m = len(idx)
        lines.append(
            "SELECTED_OUTPUT\n -reset false\n -high_precision true\nUSER_PUNCH\n -headings "
            + " ".join(_COLS) + "\n -start\n"
            "10 PUNCH CELL_NO, TOT(\"Ca\"), TOT(\"S(6)\"), TOT(\"Nit\"), ALK, -LA(\"H+\")\n"
            "20 PUNCH KIN(\"SLAD\"), KIN(\"Calcite_c\"), EQUI(\"Gypsum\"), SI(\"Calcite\"), SI(\"Gypsum\")\n"
            "30 PUNCH KIN_DELTA(\"SLAD\"), MU, EQUI(\"Calcite\"), TOT(\"water\")\n -end\n"
            "USE solution none\nEND\n"          # 정의만 하고 기본 회분 반응은 막는다(이중 반응 방지)
            f"RUN_CELLS\n -cells 1-{m}\n -time_step {dt:.8g}\nEND\n")
        txt = eng.run(EXTRA_DEFS + "".join(lines))
        rows = [r.split() for r in txt.strip().splitlines()[1:] if r.strip()]
        # RUN_CELLS는 셀마다 반응 후 한 줄
        vals = {}
        for r in rows:
            try:
                v = [float(x) for x in r]
            except ValueError:
                continue
            vals[int(v[0])] = v
        for j, i in enumerate(idx, start=1):
            v = vals.get(j)
            if v is None:
                raise PhreeqcError(f"셀 {j} 결과 없음")
            Ca, S6, Nit, Alk, pH, Ms, Mc, Mg, SIc, SIg, dS, mu, Mce, mw = v[1:15]
            # TOT 은 몰랄농도(mol/kgw). 반응으로 물 질량이 바뀌므로(석고 수화수 등) 처음 물 1 kg 기준 양으로 환산
            Ca *= mw; S6 *= mw; Nit *= mw; Alk *= mw
            out["Ca"][i] = Ca; out["SO4"][i] = S6; out["NO3"][i] = Nit; out["ALK"][i] = Alk
            out["pH"][i] = pH
            out["Ssol"][i] = max(Ms, 0.0) * rp.stoich_S
            out["Cal"][i] = max(Mc if calcite_mode != "평형" else Mce, 0.0)
            out["Gyp"][i] = max(Mg, 0.0)
            out["dN"][i] = max(cells["NO3"][i] - Nit, 0.0)
            out["SI_cal"][i] = SIc; out["SI_gyp"][i] = SIg; out["mu"][i] = mu
    return out

