"""내보내기: Excel(네이티브 차트), 프로젝트 JSON 저장/불러오기, npz/json, PNG 묶음."""
from __future__ import annotations

import base64
import io
import json
import zipfile
from dataclasses import asdict

import numpy as np
import pandas as pd

import metrics as M
from params import Common, REGISTRY, SoilLayer


# ───────────────────────────── 프로젝트 저장 ─────────────────────────────
def project_to_json(c: Common, scen_list) -> str:
    out = {"version": 2, "common": c.to_dict(), "scenarios": []}
    for s in scen_list:
        out["scenarios"].append({
            "name": s["name"], "mechanism": s["mechanism"], "install_text": s["install_text"],
            "file_name": s["file_name"],
            "file_b64": base64.b64encode(s["file_bytes"]).decode() if s.get("file_bytes") else None,
            "unit_sel": s["unit_sel"], "placement": s["placement"],
            "roles": s["role_df"].to_dict("records") if len(s["role_df"]) else [],
            "events": s["events_df"].to_dict("records"),
            "allow_refill_wo_opening": s["allow_refill_wo_opening"], "mass_correction": s["mass_correction"],
        })
    return json.dumps(out, ensure_ascii=False, indent=1, default=_np_default)


def project_from_json(txt, new_scenario, load_file_into):
    from io_3mf import rescale_objects
    d = json.loads(txt)
    c = Common.from_dict(d["common"])
    scs = []
    for s in d.get("scenarios", []):
        sc = new_scenario(s["name"])
        if s.get("file_b64"):
            load_file_into(sc, base64.b64decode(s["file_b64"]), s["file_name"])
            if s.get("unit_sel") and s["unit_sel"] != sc["unit_src"]:
                sc["objects"] = rescale_objects(sc["objects_raw"], sc["unit_src"], s["unit_sel"])
                sc["unit_sel"] = s["unit_sel"]
        sc["mechanism"] = s.get("mechanism", "")
        sc["install_text"] = s.get("install_text", "")
        sc["placement"] = s.get("placement", sc["placement"])
        if s.get("roles"):
            sc["role_df"] = pd.DataFrame(s["roles"])
        sc["events_df"] = pd.DataFrame(s.get("events", []))
        sc["allow_refill_wo_opening"] = s.get("allow_refill_wo_opening", False)
        sc["mass_correction"] = s.get("mass_correction", True)
        scs.append(sc)
    return c, scs


def _np_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


# ───────────────────────────── 결과 저장 ─────────────────────────────────
def results_npz(R) -> bytes:
    arrs = {}
    for i, r in enumerate(R["scen"] + [R["control"]]):
        tag = "control" if r["is_control"] else f"s{i}"
        for k, v in r["daily"].items():
            arrs[f"{tag}__daily__{k}"] = v
        for j, sn in enumerate(r["snaps"]):
            for key in ("NO3", "SO4", "pH", "solid", "Kratio", "theta"):
                arrs[f"{tag}__snap{j:02d}__{key}"] = sn[key]
            arrs[f"{tag}__snap{j:02d}__op_day"] = np.array(sn["op_day"])
        arrs[f"{tag}__role_cell"] = r["role_cell"]
        arrs[f"{tag}__reac0"] = r["reac0"]
    arrs["names"] = np.array([r["name"] for r in R["scen"] + [R["control"]]])
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrs)
    return buf.getvalue()


def results_json(R) -> bytes:
    def strip(m):
        return {k: v for k, v in m.items() if k not in ("profiles", "V_eff_mask")}
    out = {"common": R["common"], "eval_start": R["eval_start"],
           "metrics": [strip(m) for m in R["metrics"]],
           "runs": [{"name": r["name"], "mechanism": r["mechanism"], "install_text": r["install_text"],
                     "events_table": r["events_table"], "events_log": r["events_log"], "refills": r["refills"],
                     "mass_balance": r["mass_balance"], "warnings": r["warnings"], "diag": r["diag"],
                     "mass0_g": r["mass0_g"], "wall_s": r["wall_s"]} for r in R["scen"] + [R["control"]]],
           "skipped": R["skipped"]}
    return json.dumps(out, ensure_ascii=False, indent=1, default=_np_default).encode("utf-8")


# ───────────────────────────── Excel ────────────────────────────────────
def build_xlsx(R, c: Common, scen_inputs) -> bytes:
    from openpyxl import Workbook
    from openpyxl.chart import LineChart, Reference, ScatterChart, Series
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    hdr_font = Font(bold=True, color="FFFFFF")
    hdr_fill = PatternFill("solid", fgColor="2F5D7C")

    def write_table(ws, df, r0=1, c0=1):
        for j, col in enumerate(df.columns):
            cell = ws.cell(row=r0, column=c0 + j, value=str(col))
            cell.font = hdr_font; cell.fill = hdr_fill
            cell.alignment = Alignment(wrap_text=True, vertical="center")
        for i, row in enumerate(df.itertuples(index=False), start=1):
            for j, v in enumerate(row):
                if isinstance(v, (np.floating, float)):
                    v = None if not np.isfinite(v) else float(v)
                elif isinstance(v, (np.integer,)):
                    v = int(v)
                elif isinstance(v, (list, tuple, dict)):
                    v = json.dumps(v, ensure_ascii=False, default=_np_default)
                ws.cell(row=r0 + i, column=c0 + j, value=v)
        for j, col in enumerate(df.columns):
            ws.column_dimensions[get_column_letter(c0 + j)].width = max(12, min(48, len(str(col)) * 1.6))
        return r0 + len(df)

    runs = R["scen"] + [R["control"]]
    names = [r["name"] for r in runs]
    # 1) 요약
    ws = wb.active
    ws.title = "요약"
    head, rows = M.summary_table(R["metrics"])
    ws["A1"] = "비교군 × 지표 (지표 문서 정의)"; ws["A1"].font = Font(bold=True, size=13)
    ws["A2"] = f"η 평가 시작(대조군 50% 돌파일): {R['eval_start']} 일 · 반응재 질량·조건은 모든 비교군 동일 적용"
    end = write_table(ws, pd.DataFrame(rows, columns=head), r0=4)
    ws.cell(row=end + 2, column=1, value="라벨: '가정' 값은 실험 보정 전. 모델 한계는 '입력 파라미터' 시트 하단.")
    # 2) 입력 파라미터
    ws = wb.create_sheet("입력 파라미터")
    cd = R["common"]
    plist = [(k, json.dumps(v, ensure_ascii=False, default=_np_default) if isinstance(v, (list, dict)) else v)
             for k, v in cd.items() if k != "params"]
    end = write_table(ws, pd.DataFrame(plist, columns=["항목", "값"]))
    preg = [{"키": k, "값": cd.get("params", {}).get(k, v[0]), "단위": v[1], "라벨": v[2], "설명·출처": v[3]}
            for k, v in REGISTRY.items()]
    end = write_table(ws, pd.DataFrame(preg), r0=end + 3)
    import texts as TX
    write_table(ws, pd.DataFrame(TX.LIMITS, columns=["모델 한계", "설명"]), r0=end + 3)
    # 3) 설치 단계·메커니즘
    ws = wb.create_sheet("설치 단계·메커니즘")
    r0 = 1
    for r in R["scen"]:
        ws.cell(row=r0, column=1, value=r["name"]).font = Font(bold=True, size=12)
        ws.cell(row=r0 + 1, column=1, value="작용 메커니즘 설명"); ws.cell(row=r0 + 1, column=2, value=r["mechanism"])
        ws.cell(row=r0 + 2, column=1, value="설치 과정 설명"); ws.cell(row=r0 + 2, column=2, value=r["install_text"])
        ev = pd.DataFrame(r["events_table"])
        r0 = write_table(ws, ev, r0=r0 + 4) + 2 if len(ev) else r0 + 5
        log = pd.DataFrame([{k: v for k, v in e.items() if k not in ("inject_cells",)} for e in r["events_log"]])
        if len(log):
            ws.cell(row=r0, column=1, value="실제 재현 기록").font = Font(italic=True)
            r0 = write_table(ws, log, r0=r0 + 1) + 3
    # 4) 지표별 시계열
    ws = wb.create_sheet("지표별 시계열")
    keys = [("eta", "η (%)", 100), ("Cout_mgN", "유출 NO3-N (mg/L)", 1), ("SO4_out_mg", "유출 SO4 (mg/L)", 1),
            ("pH_out", "유출 pH", 1), ("pH_below", "반응재 아래 pH", 1), ("pH_around", "반응재 주변 pH", 1),
            ("Kreac", "반응재 K (cm/s)", 100), ("Keff", "유효 K (cm/s)", 100),
            ("N_to_gw_g", "지하수 도달 누적 (g N)", 1), ("denit_g", "누적 탈질 (g N)", 1)]
    D0 = runs[0]["daily"]
    cols = {"운영 일": D0["day"]}
    for k, lab, sc_ in keys:
        for r in runs:
            v = r["daily"][k] * sc_
            cols[f"{lab} | {r['name']}"] = v[:len(D0["day"])] if len(v) >= len(D0["day"]) else np.pad(v, (0, len(D0["day"]) - len(v)), constant_values=np.nan)
    df = pd.DataFrame(cols)
    write_table(ws, df)
    nrow = len(df) + 1
    for ci, (k, lab, _) in enumerate(keys[:4] + keys[8:9]):
        ch = LineChart()
        ch.title = lab; ch.x_axis.title = "운영 일"; ch.y_axis.title = lab; ch.height = 7.5; ch.width = 16
        first = 2 + [i for i, kk in enumerate(keys) if kk[0] == k][0] * len(runs)
        data = Reference(ws, min_col=first, max_col=first + len(runs) - 1, min_row=1, max_row=nrow)
        ch.add_data(data, titles_from_data=True)
        ch.set_categories(Reference(ws, min_col=1, min_row=2, max_row=nrow))
        ws.add_chart(ch, f"{get_column_letter(len(df.columns) + 2)}{1 + ci * 16}")
    # 5) 깊이별 프로파일
    ws = wb.create_sheet("깊이별 프로파일")
    g = runs[0]["grid"]
    d = (np.arange(g["nz"]) + 0.5) * g["dz"] * 100
    cols = {"깊이 (cm)": d}
    for r, m in zip(R["scen"] + [R["control"]], R["metrics"]):
        if m.get("profiles"):
            p = m["profiles"][-1]
            for key, lab in (("pH", "pH"), ("NO3", "NO3-N mg/L"), ("SO4", "SO4 mg/L")):
                cols[f"{lab} | {r['name']}"] = p[key]
    dfp = pd.DataFrame(cols)
    write_table(ws, dfp)
    for ci, lab in enumerate(("pH", "NO3-N mg/L", "SO4 mg/L")):
        ch = ScatterChart()
        ch.title = f"{lab} 깊이 프로파일(마지막 시점)"; ch.x_axis.title = lab; ch.y_axis.title = "깊이 (cm)"
        ch.y_axis.scaling.orientation = "maxMin"; ch.height = 8; ch.width = 12; ch.style = 13
        yref = Reference(ws, min_col=1, min_row=2, max_row=len(dfp) + 1)
        for j, col in enumerate(dfp.columns):
            if col.startswith(lab):
                xref = Reference(ws, min_col=j + 1, min_row=1, max_row=len(dfp) + 1)
                s_ = Series(yref, xref, title_from_data=True)
                ch.series.append(s_)
        ws.add_chart(ch, f"{get_column_letter(len(dfp.columns) + 2 + ci * 8)}2")
    # 6) 재보충 이벤트
    ws = wb.create_sheet("재보충 이벤트")
    rows = []
    for r, m in zip(R["scen"], R["metrics"]):
        for x in m.get("refill_table", []):
            rows.append({"비교군": r["name"], "회차": x.get("n"), "운영 일": x["day"], "방식": x.get("method"),
                         "성공": x.get("ok"), "η 직전 (%)": (x.get("eta_before") or 0) * 100,
                         "R 24h (%)": x.get("R_24h"), "R 이동시간 보정 (%)": x.get("R_travel"),
                         "주입량 (L)": x.get("delivered_L"), "T_r (분)": x.get("duration_min"), "사유": x.get("why", "")})
        rows.append({"비교군": r["name"], "회차": "요약", "R 이동시간 보정 (%)": m.get("R"), "R 24h (%)": m.get("R_24h"),
                     "사유": m.get("refill_note"), "T_r (분)": m.get("T_r_min")})
    write_table(ws, pd.DataFrame(rows))
    # 7) 지하수 도달량
    ws = wb.create_sheet("지하수 도달량")
    cols = {"운영 일": D0["day"]}
    for r in runs:
        cols[f"누적 도달 NO3-N (g) | {r['name']}"] = r["daily"]["N_to_gw_g"]
        cols[f"지하수층 평균 NO3-N (mg/L) | {r['name']}"] = r["daily"]["gw_NO3_mgN"]
    dfg = pd.DataFrame(cols)
    write_table(ws, dfg)
    ch = LineChart(); ch.title = "지하수 도달 NO3-N 누적"; ch.x_axis.title = "운영 일"; ch.y_axis.title = "g N"
    ch.height = 8; ch.width = 18
    for j, col in enumerate(dfg.columns):
        if col.startswith("누적"):
            ch.add_data(Reference(ws, min_col=j + 1, min_row=1, max_row=len(dfg) + 1), titles_from_data=True)
    ch.set_categories(Reference(ws, min_col=1, min_row=2, max_row=len(dfg) + 1))
    ws.add_chart(ch, f"{get_column_letter(len(dfg.columns) + 2)}2")
    # 8) 대조군 비교
    ws = wb.create_sheet("대조군 비교")
    rows = []
    for r, m in zip(R["scen"], R["metrics"]):
        rows.append({"비교군": r["name"], "유출 NO3-N 평균 (mg/L)": m.get("Cout_mean"),
                     "대조군 대비 유출 농도 감소 (%)": m.get("Cout_improve_pct"),
                     "지하수 도달 누적 (g N)": m.get("N_to_gw_g"),
                     "대조군 대비 지하수 피해 감소 (%)": m.get("gw_improve_pct"),
                     "10 mg/L 도달일": m.get("arrival_10mg_day"),
                     "반응재 통과 유량/투영면 강수": m.get("through_ratio")})
    mc = R["metrics"][-1]
    rows.append({"비교군": "대조군(구조물 없음)", "유출 NO3-N 평균 (mg/L)": mc.get("Cout_mean"),
                 "지하수 도달 누적 (g N)": mc.get("N_to_gw_g"), "10 mg/L 도달일": mc.get("arrival_10mg_day")})
    dfc = pd.DataFrame(rows)
    write_table(ws, dfc)
    from openpyxl.chart import BarChart
    ch = BarChart(); ch.title = "대조군 대비 지하수 피해 감소 (%)"; ch.height = 8; ch.width = 14
    ch.add_data(Reference(ws, min_col=5, min_row=1, max_row=len(dfc)), titles_from_data=True)
    ch.set_categories(Reference(ws, min_col=1, min_row=2, max_row=len(dfc)))
    ws.add_chart(ch, "J2")
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# ───────────────────────────── PNG 묶음(matplotlib) ───────────────────────
def png_zip(R, c: Common) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    for fn in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"):
        if any(fn in f.name for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = fn
            break
    plt.rcParams["axes.unicode_minus"] = False
    runs = R["scen"] + [R["control"]]
    buf = io.BytesIO()
    z = zipfile.ZipFile(buf, "w")
    for key, lab, sc_ in (("eta", "제거효율 η (%)", 100), ("Cout_mgN", "유출 NO3-N (mg/L)", 1),
                          ("SO4_out_mg", "유출 SO4 (mg/L)", 1), ("pH_below", "반응재 아래 pH", 1),
                          ("N_to_gw_g", "지하수 도달 누적 (g N)", 1), ("Kreac", "반응재 영역 K (m/s)", 1)):
        fig, ax = plt.subplots(figsize=(7, 3.6), dpi=150)
        for i, r in enumerate(runs):
            ax.plot(r["daily"]["day"], r["daily"][key] * sc_, label=r["name"],
                    ls="--" if r["is_control"] else "-", color="#555" if r["is_control"] else None)
        ax.set_xlabel("운영 일"); ax.set_ylabel(lab); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        if R["eval_start"]:
            ax.axvspan(0, R["eval_start"], color="#ccc", alpha=0.3)
        fig.tight_layout()
        b = io.BytesIO(); fig.savefig(b, format="png"); plt.close(fig)
        z.writestr(f"{key}.png", b.getvalue())
    # 단면도(마지막 시점, y 중앙)
    for r in runs:
        sn = r["snaps"][-1]
        g = r["grid"]
        fig, axs = plt.subplots(1, 3, figsize=(11, 3.6), dpi=140)
        j = g["ny"] // 2
        ext = [0, g["Lx"], -g["H"], 0]
        for ax, key, cm, lab in zip(axs, ("NO3", "pH", "solid"), ("Reds", "RdBu", "YlOrBr"),
                                    ("NO3-N (mg/L)", "pH", "반응재 고체 부피분율")):
            arr = np.where(r["active"], sn[key], np.nan)[:, j, :].T
            im = ax.imshow(arr, extent=ext, aspect="auto", cmap=cm, origin="upper")
            ax.set_title(lab); ax.set_xlabel("x (m)"); ax.set_ylabel("깊이 (m)")
            fig.colorbar(im, ax=ax, shrink=0.8)
        fig.suptitle(f"{r['name']} — 운영 {sn['op_day']:.0f}일 (y 중앙 단면)")
        fig.tight_layout()
        b = io.BytesIO(); fig.savefig(b, format="png"); plt.close(fig)
        z.writestr(f"단면_{r['name']}.png", b.getvalue())
    z.close()
    return buf.getvalue()
