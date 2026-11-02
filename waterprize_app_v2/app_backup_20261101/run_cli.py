"""Streamlit 없이 프로젝트(.json)를 계산하고 결과(Excel, json)를 저장.

    python run_cli.py slad_project.json            # 저장한 프로젝트 실행
    python run_cli.py --examples [--days 120]      # 들어 있는 예시 두 개(영구 주입공·케이슨) 실행
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import export_xlsx as X  # noqa: E402
import metrics as M  # noqa: E402
from engine import Scenario, control_breakthrough_day, eval_start_from, run_scenario  # noqa: E402
from io_3mf import load_mesh_file  # noqa: E402
from params import GRID_PRESETS, Common, suggest_role  # noqa: E402
from voxel import Placement  # noqa: E402


def _new(name):
    return {"name": name, "mechanism": "", "install_text": "", "file_name": "", "file_bytes": None,
            "unit_src": "millimeter", "unit_sel": "millimeter", "objects": [], "objects_raw": [], "title": "",
            "description": "", "role_df": pd.DataFrame(), "placement": dict(x=None, y=None, top_depth=0.0, rot_deg=0.0,
            tilt_deg=0.0, scale=1.0), "events_df": pd.DataFrame(), "allow_refill_wo_opening": False,
            "mass_correction": True, "geom_report": None}


def _load(sc, data, fname):
    d = load_mesh_file(data, fname)
    sc.update(file_bytes=data, file_name=fname, unit_src=d["unit"], unit_sel=d["unit"], objects=d["objects"],
              objects_raw=d["objects"])
    sc["role_df"] = pd.DataFrame([{"객체": o.name, "역할": suggest_role(o.name)} for o in d["objects"]])




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project", nargs="?")
    ap.add_argument("--examples", action="store_true")
    ap.add_argument("--days", type=float, default=None)
    ap.add_argument("--grid", default=None, choices=list(GRID_PRESETS))
    a = ap.parse_args()
    if a.examples:
        sys.path.insert(0, os.path.join(HERE, "verify"))
        from run_verification import _scenario
        import install_events as ie
        c = Common()
        well_ev = [dict(ie.empty_row("구조물 삽입"), **{"소요(분)": 30.0}), dict(ie.empty_row("개구부 열기"), **{"소요(분)": 1.0}),
                   dict(ie.empty_row("시약 주입"), **{"방식": "펌프", "부피(L)": 2.84, "고형분비(v/v)": 0.2, "유량(L/min)": 0.2}),
                   dict(ie.empty_row("정치(대기)"), **{"소요(분)": 60.0}),
                   dict(ie.empty_row("재보충"), **{"방식": "펌프", "부피(L)": 1.0, "고형분비(v/v)": 0.2, "유량(L/min)": 0.2,
                                                 "트리거 η(%)": 50.0, "간격(일)": 7.0, "최대 회차": 5.0, "소요(분)": 20.0})]
        cais_ev = [dict(ie.empty_row("구조물 삽입"), **{"소요(분)": 30.0})]
        scs = [_scenario(os.path.join(HERE, "samples", "injection_well_scale0.05.3mf"), well_ev)[0],
               _scenario(os.path.join(HERE, "samples", "caisson_gate_scale0.05.3mf"), cais_ev)[0]]
        scen_inputs = []
    else:
        import importlib
        txt = open(a.project, encoding="utf-8").read()
        c, scen_inputs = X.project_from_json(txt, _new, _load)
        scs = []
        for s in scen_inputs:
            roles = {}
            for r in s["role_df"].to_dict("records"):
                roles[r["객체"]] = {"role": r["역할"], "K": r.get("K (m/s)"), "phi": r.get("공극률"),
                                   "reactive": r.get("반응", True), "accessible": r.get("교체 가능", False)}
            pl = dict(s["placement"]); pl["x"] = c.Lx / 2 if pl.get("x") is None else pl["x"]
            pl["y"] = c.Ly / 2 if pl.get("y") is None else pl["y"]
            scs.append(Scenario(name=s["name"], mechanism=s["mechanism"], install_text=s["install_text"],
                                objects=s["objects"], roles=roles, placement=Placement(**pl),
                                events=s["events_df"].to_dict("records")))
    if a.days:
        c.days = a.days
    if a.grid:
        c.nx, c.ny, c.nz = GRID_PRESETS[a.grid]
    prog = lambda f, m: print(f"\r  {m}      ", end="", flush=True)
    print("대조군 계산…")
    ctrl = run_scenario(c, None, prog); print()
    bt = eval_start_from(ctrl, c.NO3_in_mgN)
    res = []
    for s in scs:
        print(f"{s.name} 계산…")
        res.append(run_scenario(c, s, prog, eval_start_day=bt)); print()
    R = {"control": ctrl, "scen": res, "skipped": [], "common": c.to_dict(), "eval_start": bt}
    R["metrics"] = [M.compute(r, ctrl, c, bt) for r in res] + [M.compute(ctrl, None, c, bt)]
    head, rows = M.summary_table(R["metrics"])
    print(pd.DataFrame(rows, columns=head).to_string(index=False))
    with open(os.path.join(HERE, "cli_결과.xlsx"), "wb") as fh:
        fh.write(X.build_xlsx(R, c, scen_inputs))
    with open(os.path.join(HERE, "cli_결과.json"), "wb") as fh:
        fh.write(X.results_json(R))
    print("저장: cli_결과.xlsx, cli_결과.json")


if __name__ == "__main__":
    main()
