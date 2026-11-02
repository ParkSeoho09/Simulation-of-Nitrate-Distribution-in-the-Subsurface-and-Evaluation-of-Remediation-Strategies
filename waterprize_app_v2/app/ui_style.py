"""화면 디자인(지층 단면 팔레트). 앱 첫머리에서 inject()를 한 번 부른다.

팔레트 — 흙수조의 재료에서 가져옴:
  석회석 바탕 #F2F4F3 · 잉크 #1B2B33 · 흙 #6E4F35 · 지하수 #1D6A86 · 황(반응재) #E0B53A · 경고 #A8323E · 통과 #2F7D5B
기억에 남는 요소는 하나: 머리말의 '코어 샘플' 띠(흙층 → 지하수면 → 지하수층, 반응재 렌즈). 나머지는 조용하게.
"""
PALETTE = {"paper": "#F2F4F3", "ink": "#1B2B33", "soil": "#6E4F35", "water": "#1D6A86", "sulfur": "#E0B53A",
           "alert": "#A8323E", "ok": "#2F7D5B", "line": "#C9D2D1", "mute": "#5B6B70"}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stTextInput, .stNumberInput, .stSelectbox, button, input, textarea {
  font-family: 'IBM Plex Sans KR', 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif !important;
}
.stApp { background: #F2F4F3; word-break: keep-all; }
.block-container { padding-top: 1.2rem; max-width: 1480px; }
h1, h2, h3 { color: #1B2B33; letter-spacing: -0.01em; }
h3 { font-weight: 600; font-size: 1.15rem; margin-top: 1.4rem; }
/* 숫자는 고정폭 숫자(표·지표 비교가 쉬움) */
[data-testid="stDataFrame"], [data-testid="stMetricValue"], .stNumberInput input { font-variant-numeric: tabular-nums; }
/* 탭: 지층 경계선처럼 아래쪽 굵은 선 하나로만 구분. 전환 시 밑줄이 미끄러지듯 이동 */
.stTabs [data-baseweb="tab-list"] { gap: 2px; border-bottom: 2px solid #C9D2D1; }
.stTabs [data-baseweb="tab"] { padding: 10px 14px; color: #5B6B70; font-weight: 500; transition: color .15s ease; }
.stTabs [data-baseweb="tab"]:hover { color: #1B2B33; }
.stTabs [aria-selected="true"] { color: #1B2B33; }
.stTabs [data-baseweb="tab-highlight"] { background-color: #1D6A86; height: 3px; transition: left .22s cubic-bezier(.3,.1,.2,1), width .22s cubic-bezier(.3,.1,.2,1); }
/* 버튼: 주 실행만 지하수색으로 채움. 호버·클릭에 짧은 되돌림만 — 들썩임 없음 */
.stButton > button { border-radius: 6px; border: 1px solid #C9D2D1; background: #FFFFFF; color: #1B2B33;
                      transition: border-color .15s ease, color .15s ease, background-color .15s ease, transform .08s ease; }
.stButton > button:hover { border-color: #1D6A86; color: #1D6A86; }
.stButton > button:active { transform: scale(.98); }
.stButton > button[kind="primary"] { background: #1D6A86; border-color: #1D6A86; color: #FFFFFF; }
.stButton > button[kind="primary"]:hover { background: #185777; border-color: #185777; color: #FFFFFF; }
.stButton > button:focus-visible { outline: 3px solid #E0B53A; outline-offset: 2px; }
/* 경고·오류 상자는 색 막대 하나로, 나타날 때만 살짝 가라앉듯 */
[data-testid="stAlert"] { border-radius: 4px; }
/* 펼침(계산 방법 보기 등): 화살표만 회전, 내용은 순간 전환(레이아웃 애니메이션 비용 없음) */
[data-testid="stExpander"] summary svg, [data-testid="stExpanderToggleIcon"] { transition: transform .18s ease; }
[data-testid="stExpander"] details[open] summary svg { transform: rotate(180deg); }
/* 코어 샘플 머리말 */
.core-head { display: grid; grid-template-columns: minmax(260px, 1fr) minmax(320px, 1.4fr); gap: 28px; align-items: end;
             margin: 0 0 14px 0; }
.core-head h1 { font-size: 1.75rem; font-weight: 700; margin: 0; line-height: 1.2; }
.core-head p { word-break: keep-all; margin: 6px 0 0 0; color: #5B6B70; font-size: 0.92rem; max-width: 62ch; }
.core { position: relative; height: 84px; border-radius: 3px; overflow: hidden; border: 1px solid #C9D2D1;
        background: linear-gradient(180deg,
          #8B6A4C 0 14%, #7A5B40 14% 36%, #6E4F35 36% 58%, #5E432C 58% 70%,
          #1D6A86 70% 100%); }
.core .wt { position: absolute; left: 0; right: 0; top: 70%; border-top: 2px dashed #F2F4F3; }
.core .lens { position: absolute; left: 41%; top: 20%; width: 22%; height: 34%; border-radius: 50%;
              background: radial-gradient(ellipse at center, #F1CF5E 0%, #E0B53A 58%, rgba(224,181,58,0) 72%); }
.core .lab { position: absolute; font-size: 11px; color: #F2F4F3; opacity: .92; }
@media (max-width: 760px) { .core-head { grid-template-columns: 1fr; } }
@media (prefers-reduced-motion: no-preference) {
  .core .lens { animation: lens 1.2s ease-out 1; }
  @keyframes lens { from { opacity: 0; transform: scale(.5); } to { opacity: 1; transform: none; } }
}
@media (prefers-reduced-motion: reduce) {
  .stTabs [data-baseweb="tab"], .stTabs [data-baseweb="tab-highlight"], .stButton > button,
  [data-testid="stExpander"] summary svg, [data-testid="stExpanderToggleIcon"] { transition: none !important; }
  .stButton > button:active { transform: none; }
}
/* 결과 신뢰성 띠: 다섯 칸, 각 칸은 이름(작게) + 상태 칩. 근거가 서로 다르다는 걸 칸 구분으로 보여 준다 */
.trust { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); border: 1px solid #C9D2D1; border-radius: 4px;
         background: #FFFFFF; margin: 4px 0 14px 0; }
.trust-cell { padding: 8px 12px; display: flex; flex-direction: column; gap: 4px; border-left: 1px solid #E4E9E8; min-width: 0; }
.trust-cell:first-child { border-left: none; }
.trust-cell > span:first-child { font-size: 0.78rem; color: #5B6B70; }
.trust-cell .chip { align-self: flex-start; max-width: 100%; word-break: keep-all; overflow-wrap: break-word; border-radius: 10px; }
@media (max-width: 900px) { .trust { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .trust-cell { border-left: none; border-top: 1px solid #E4E9E8; } .trust-cell:nth-child(-n+2) { border-top: none; } }
/* 상태 표시(통과·미검증·FAILED) */
.chip { display: inline-block; padding: 1px 9px; border-radius: 999px; font-size: 0.82rem; font-weight: 600; }
.chip.ok { background: #DCEEE5; color: #2F7D5B; } .chip.warn { background: #F6ECCB; color: #7A5A06; }
.chip.fail { background: #F3DADD; color: #A8323E; }
</style>
"""

HEADER = """
<div class="core-head">
  <div>
    <h1>SLAD 반응재 배치 비교 시뮬레이터</h1>
    <p>흙수조에 황:석회석 반응재를 어떤 구조로, 어떻게 넣느냐에 따라 질산염이 지하수에 닿는 양이 얼마나 달라지는지 계산합니다.
       비교군은 3MF와 설치 방법으로 직접 넣습니다.</p>
  </div>
  <div class="core" role="img" aria-label="흙수조 단면: 흙층 안의 반응재, 지하수면, 지하수층">
    <div class="lens"></div><div class="wt"></div>
    <span class="lab" style="left:8px; top:6px">흙(불포화대)</span>
    <span class="lab" style="left:calc(63% + 8px); top:26%">반응재</span>
    <span class="lab" style="right:8px; top:calc(70% - 16px)">지하수면</span>
    <span class="lab" style="left:8px; bottom:5px">지하수층</span>
  </div>
</div>
"""


def chip(text, kind="ok"):
    return f'<span class="chip {kind}">{text}</span>'


def inject(st):
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(HEADER, unsafe_allow_html=True)
