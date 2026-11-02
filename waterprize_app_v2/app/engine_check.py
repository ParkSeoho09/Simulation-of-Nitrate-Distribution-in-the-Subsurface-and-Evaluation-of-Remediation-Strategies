"""엔진 후보 설치·실행 확인(이 컴퓨터에서 실제로 해 본 결과만 기록).

    python engine_check.py            # 결과를 engine_check.json 에 저장

- PHREEQC(IPhreeqc): 라이브러리 적재 + 작은 계산(순수 물 pH) 실행
- OpenGeoSys 6: python 패키지 ogs / ogstools / ogs6py import, `ogs --version` 실행
- PFLOTRAN: `pflotran` 실행 파일 존재 여부
확인하지 못한 것은 '확인 안 됨'으로 남긴다.
"""
from __future__ import annotations

import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _try_import(name):
    try:
        m = importlib.import_module(name)
        return {"import": "성공", "version": getattr(m, "__version__", "?"), "path": os.path.dirname(getattr(m, "__file__", "") or "")}
    except Exception as e:      # noqa: BLE001
        return {"import": "실패", "error": f"{type(e).__name__}: {e}"[:300]}


def _try_exe(name, args=("--version",)):
    p = shutil.which(name)
    if not p:
        return {"실행 파일": "없음(PATH)"}
    try:
        r = subprocess.run([p, *args], capture_output=True, text=True, timeout=60)
        return {"실행 파일": p, "반환 코드": r.returncode, "출력": (r.stdout or r.stderr)[:300]}
    except Exception as e:      # noqa: BLE001
        return {"실행 파일": p, "error": str(e)[:300]}


def check(write=False):
    out = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "python": sys.version.split()[0], "platform": platform.platform()}
    import phreeqc_engine as PE
    st = PE.status()
    out["PHREEQC"] = dict(st)
    if st["ok"] == "예":
        try:
            eng = PE.get_engine()
            txt = eng.run("SOLUTION 1\n pH 7 charge\nSELECTED_OUTPUT\n -reset false\nUSER_PUNCH\n -headings pH\n -start\n"
                          "10 PUNCH -LA(\"H+\")\n -end\nEND\n")
            out["PHREEQC"]["시험 계산(순수 물 pH)"] = txt.strip().splitlines()[-1].strip() if txt.strip() else "출력 없음"
        except Exception as e:      # noqa: BLE001
            out["PHREEQC"]["시험 계산"] = f"실패: {e}"[:300]
    out["phreeqpy"] = _try_import("phreeqpy")
    out["OGS(ogs 패키지)"] = _try_import("ogs")
    out["ogstools"] = _try_import("ogstools")
    out["ogs6py"] = _try_import("ogs6py")
    out["ogs 실행 파일"] = _try_exe("ogs")
    out["PFLOTRAN 실행 파일"] = _try_exe("pflotran", ("-help",))
    out["채택"] = ("PHREEQC(IPhreeqc)를 지화학 엔진으로 사용" if st["ok"] == "예" else "PHREEQC 없음 → 내장 화학") + \
                 " · 흐름·이송은 v2 엔진 유지(사유는 README '엔진 채택 결정')"
    if write:
        with open(os.path.join(HERE, "engine_check.json"), "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=1)
    return out


if __name__ == "__main__":
    print(json.dumps(check(write=True), ensure_ascii=False, indent=1))
