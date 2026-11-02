"""설치(꽂는) 과정 이벤트 타임라인.

- 표 스키마(한국어 열 이름), 검증, 시간 계산
- 설치 설명 텍스트 → 표 초안 (규칙 기반)
- 선택: Anthropic API 키가 있으면 LLM으로 변환(키가 없어도 앱은 동작)
"""
from __future__ import annotations

import json
import math
import re
import urllib.request
from typing import Dict, List, Optional, Tuple

EVENT_TYPES = ["구조물 삽입", "케이싱 유지", "케이싱 인발", "개구부 열기", "개구부 닫기",
               "시약 주입", "정치(대기)", "재보충", "센서 측정"]
METHODS = ["", "펌프", "가압", "중력", "카트리지 교체"]

COLUMNS = ["이벤트", "시작(일)", "소요(분)", "대상 객체", "x(m)", "y(m)", "깊이(m)", "기울기(°)",
           "회전(°)", "방식", "부피(L)", "고형분비(v/v)", "압력(kPa)", "유량(L/min)",
           "트리거 η(%)", "간격(일)", "최대 회차", "메모"]

NUMERIC = ["시작(일)", "소요(분)", "x(m)", "y(m)", "깊이(m)", "기울기(°)", "회전(°)", "부피(L)",
           "고형분비(v/v)", "압력(kPa)", "유량(L/min)", "트리거 η(%)", "간격(일)", "최대 회차"]


def empty_row(event="구조물 삽입"):
    r = {c: None for c in COLUMNS}
    r["이벤트"] = event
    r["대상 객체"] = "전체"
    r["방식"] = ""
    r["메모"] = ""
    return r


def _num(v):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    return f


def clean_rows(rows: List[dict]) -> List[dict]:
    out = []
    for r in rows:
        if not r or not r.get("이벤트"):
            continue
        rr = {c: r.get(c) for c in COLUMNS}
        for c in NUMERIC:
            rr[c] = _num(rr[c])
        rr["대상 객체"] = (rr.get("대상 객체") or "전체")
        rr["방식"] = rr.get("방식") or ""
        rr["메모"] = rr.get("메모") or ""
        out.append(rr)
    return out


def schedule(rows: List[dict]) -> List[dict]:
    """시작 시각이 비어 있으면 이전 단계 직후. 재보충은 운영 중 규칙이므로 순서 계산에서 제외.
    반환 행에 t_start(일), t_end(일) 추가."""
    t = 0.0
    out = []
    for r in clean_rows(rows):
        r = dict(r)
        dur_d = (r["소요(분)"] or 0.0) / 1440.0
        if r["이벤트"] == "재보충":
            r["t_start"] = r["시작(일)"]
            r["t_end"] = None
            out.append(r)
            continue
        st = r["시작(일)"] if r["시작(일)"] is not None else t
        r["t_start"] = st
        r["t_end"] = st + dur_d
        t = r["t_end"]
        out.append(r)
    return out


def validate(rows: List[dict], object_roles: Dict[str, str]) -> Tuple[List[str], List[str]]:
    """반환 (오류, 경고). object_roles: {객체명: 역할}"""
    errs, warns = [], []
    rows = clean_rows(rows)
    if not rows:
        errs.append("설치 방법을 입력하세요. (설치 단계 표가 비어 있습니다)")
        return errs, warns
    openings = [n for n, r in object_roles.items() if r == "주입 개구부"]
    access = [n for n, r in object_roles.items() if r == "반응재(고체)"]
    has_insert = any(r["이벤트"] == "구조물 삽입" for r in rows)
    if not has_insert and object_roles:
        warns.append("'구조물 삽입' 단계가 없어 구조물을 시작 시점(0일)에 이미 꽂혀 있는 것으로 봅니다.")
    for i, r in enumerate(rows, 1):
        ev = r["이벤트"]
        for c in NUMERIC:
            v = r[c]
            if v is not None and v < 0 and c not in ("x(m)", "y(m)", "기울기(°)", "회전(°)"):
                errs.append(f"{i}행 '{c}' = {v}: 이 값은 계산이 불가능합니다(음수).")
        if ev == "시약 주입":
            if not openings:
                errs.append(f"{i}행 시약 주입: 이 구조에는 '주입 개구부' 역할 객체가 없습니다.")
            if not r["부피(L)"]:
                errs.append(f"{i}행 시약 주입: 부피(L)를 입력하세요.")
            if r["방식"] in ("", None):
                warns.append(f"{i}행 시약 주입: 방식이 비어 있어 '펌프'로 봅니다.")
            if r["방식"] == "가압" and not r["압력(kPa)"]:
                errs.append(f"{i}행 가압 주입: 압력(kPa)을 입력하세요.")
            if r["고형분비(v/v)"] is not None and r["고형분비(v/v)"] >= 1:
                errs.append(f"{i}행 고형분비 {r['고형분비(v/v)']}: 1 이상이면 계산이 불가능합니다.")
            if r["고형분비(v/v)"] is not None and r["고형분비(v/v)"] > 0.6:
                warns.append(f"{i}행 고형분비 {r['고형분비(v/v)']}: 0.6 초과 슬러리는 펌핑이 비현실적일 수 있습니다.")
        if ev == "재보충":
            if r["방식"] == "카트리지 교체":
                if not access:
                    warns.append(f"{i}행 카트리지 교체: 교체할 반응재 객체가 없습니다.")
            elif not openings:
                warns.append(f"{i}행 재보충: 개구부가 없는 구조라 재보충할 수 없습니다(R=0 처리).")
            if r["트리거 η(%)"] is None and r["간격(일)"] is None:
                warns.append(f"{i}행 재보충: 트리거 η와 간격이 모두 비어 '트리거 η 50%'로 봅니다(지표 문서 예시).")
    return errs, warns


# ───────────────────────────── 텍스트 → 표 (규칙 기반) ─────────────────────
_EV_KEYS = [
    ("재보충", ["재보충", "재주입", "보충", "교체", "리필", "refill"]),
    ("케이싱 인발", ["인발", "뽑", "빼낸", "제거"]),
    ("개구부 닫기", ["닫", "밀폐", "마개", "캡"]),
    ("개구부 열기", ["열어", "열고", "연다", "열린", "개방", "오픈"]),
    ("구조물 삽입", ["삽입", "꽂", "관입", "박는", "박아", "매설", "시추", "설치"]),
    ("시약 주입", ["주입", "투입", "넣", "부어", "슬러리"]),
    ("정치(대기)", ["정치", "대기", "방치", "기다", "양생"]),
    ("센서 측정", ["측정", "센서", "샘플"]),
    ("케이싱 유지", ["케이싱 유지", "그대로 둔", "남겨"]),
]

_UNIT_M = {"mm": 1e-3, "cm": 1e-2, "m": 1.0}


def _find(pattern, s, conv=float):
    m = re.search(pattern, s, flags=re.I)
    return conv(m.group(1)) if m else None


def parse_text(text: str, object_names: Optional[List[str]] = None) -> List[dict]:
    """설치 설명을 문장 단위로 읽어 이벤트 행 초안을 만든다."""
    object_names = object_names or []
    # 소수점(0.2)은 자르지 않도록 '.' 뒤가 공백/끝일 때만 문장 경계로 본다
    parts = re.split(r"\n|。|;|→|(?<!\d)\.(?!\d)|\.(?=\s|$)|(?:\s후\s)|(?:\s다음\s)|(?:\s그리고\s)|(?:\d+\)\s)", text)
    rows = []
    for p in parts:
        s = p.strip()
        if len(s) < 2:
            continue
        ev = None
        for name, keys in _EV_KEYS:
            if any(k in s for k in keys):
                ev = name
                break
        if ev is None:
            continue
        r = empty_row(ev)
        # 깊이
        m = re.search(r"(?:깊이|심도|depth)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(mm|cm|m)\b", s, re.I) or \
            re.search(r"(\d+(?:\.\d+)?)\s*(mm|cm|m)\s*(?:깊이|심도|아래|까지)", s, re.I)
        if m:
            r["깊이(m)"] = float(m.group(1)) * _UNIT_M[m.group(2).lower()]
        # 부피
        m = re.search(r"(\d+(?:\.\d+)?)\s*(mL|ml|L|ℓ|리터)", s)
        if m:
            v = float(m.group(1))
            r["부피(L)"] = v / 1000.0 if m.group(2).lower() == "ml" else v
        # 시간
        m = re.search(r"(\d+(?:\.\d+)?)\s*(초|분|시간|일)", s)
        if m:
            v = float(m.group(1)); u = m.group(2)
            if ev == "재보충" and u == "일" and ("마다" in s or "간격" in s or "주기" in s):
                r["간격(일)"] = v
            else:
                r["소요(분)"] = {"초": v / 60, "분": v, "시간": v * 60, "일": v * 1440}[u]
        # 압력·유량·농도·η·회차·기울기
        r["압력(kPa)"] = _find(r"(\d+(?:\.\d+)?)\s*kPa", s)
        if r["압력(kPa)"] is None:
            bar = _find(r"(\d+(?:\.\d+)?)\s*bar", s)
            if bar is not None:
                r["압력(kPa)"] = bar * 100
        r["유량(L/min)"] = _find(r"(\d+(?:\.\d+)?)\s*L\s*/\s*min", s)
        sr = _find(r"(?:고형분|슬러리 비율|고형)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*%", s)
        if sr is not None:
            r["고형분비(v/v)"] = sr / 100
        eta = _find(r"[ηn효율]\s*[<≤]?\s*(\d+(?:\.\d+)?)\s*%", s) or _find(r"(\d+(?:\.\d+)?)\s*%\s*(?:미만|이하|아래)", s)
        if eta is not None and ev == "재보충":
            r["트리거 η(%)"] = eta
        nmax = _find(r"(\d+)\s*회", s)
        if nmax is not None and ev == "재보충":
            r["최대 회차"] = nmax
        tilt = _find(r"(\d+(?:\.\d+)?)\s*(?:°|도)\s*(?:기울|경사)", s)
        if tilt is not None:
            r["기울기(°)"] = tilt
        # 방식
        if ev in ("시약 주입", "재보충"):
            if "카트리지" in s:
                r["방식"] = "카트리지 교체"
            elif "가압" in s or "압력" in s:
                r["방식"] = "가압"
            elif "중력" in s or "자연" in s or "부어" in s:
                r["방식"] = "중력"
            elif "펌프" in s or r["유량(L/min)"]:
                r["방식"] = "펌프"
            else:
                r["방식"] = "펌프"
        # 대상 객체(이름 언급)
        hit = [n for n in object_names if n and n.lower() in s.lower()]
        if hit:
            r["대상 객체"] = ", ".join(hit)
        r["메모"] = s[:80]
        rows.append(r)
    return rows


def parse_text_llm(text: str, object_names: List[str], api_key: str,
                   model: str = "claude-sonnet-5-5", timeout: float = 60.0) -> List[dict]:
    """Anthropic Messages API로 텍스트를 표로 변환. 실패하면 예외."""
    schema_hint = {c: "숫자 또는 null" if c in NUMERIC else "문자열" for c in COLUMNS}
    prompt = (
        "다음은 흙수조에 구조물을 꽂고 시약을 넣는 설치 과정 설명이다. 각 단계를 JSON 배열의 행으로 바꿔라.\n"
        f"이벤트 값은 {EVENT_TYPES} 중 하나, 방식은 {METHODS} 중 하나.\n"
        f"사용 가능한 객체 이름: {object_names}\n"
        f"열 정의: {json.dumps(schema_hint, ensure_ascii=False)}\n"
        "단위: 깊이 m, 부피 L, 소요 분, 압력 kPa, 유량 L/min, 고형분비 0~1, 트리거 η %.\n"
        "설명에 없는 값은 null. JSON 배열만 출력.\n\n설명:\n" + text)
    body = json.dumps({"model": model, "max_tokens": 4000,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body, method="POST",
                                 headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                                          "content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
    txt = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    m = re.search(r"\[.*\]", txt, re.S)
    if not m:
        raise ValueError("LLM 응답에서 표를 찾지 못했습니다.")
    rows = json.loads(m.group(0))
    out = []
    for r in rows:
        rr = empty_row(r.get("이벤트", "구조물 삽입"))
        for c in COLUMNS:
            if c in r:
                rr[c] = r[c]
        if rr["이벤트"] not in EVENT_TYPES:
            continue
        out.append(rr)
    return out
