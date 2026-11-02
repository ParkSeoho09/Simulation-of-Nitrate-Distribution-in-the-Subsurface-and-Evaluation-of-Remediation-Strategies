"""3MF 입력 (object 단위).

- 3MF Core Spec: ZIP 안의 *.model XML. <model unit="..."> 단위, <object name=...>,
  <components>(재귀), <build><item transform=...> 를 모두 반영한다.
- 각 build item(=배치된 object 인스턴스)을 하나의 "객체"로 돌려준다.
  객체마다 이름, 원래 단위, 꼭짓점(미터), 삼각형, 표시색, 닫힘 여부, 체적, 최소 두께 추정을 담는다.
- 객체를 합치지 않는다(합친 뒤 짝홀 판정하면 겹친 객체끼리 내부가 상쇄되는 오류가 생긴다).
- 단위 변환 외의 축소/확대는 하지 않는다.
"""
from __future__ import annotations

import io
import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Optional, Dict

import numpy as np

UNIT_TO_M = {
    "micron": 1e-6, "millimeter": 1e-3, "centimeter": 1e-2,
    "inch": 0.0254, "foot": 0.3048, "meter": 1.0,
}
UNIT_LABEL = {"micron": "µm", "millimeter": "mm", "centimeter": "cm",
              "inch": "inch", "foot": "ft", "meter": "m"}


@dataclass
class MeshObject:
    name: str
    vertices_m: np.ndarray          # (N,3) 미터, 3MF 좌표계(z 위쪽, 보통 z=0 지표)
    triangles: np.ndarray           # (M,3) int
    color: str = "#888888"
    source_unit: str = "millimeter"
    closed: bool = True
    open_edges: int = 0
    volume_m3: float = 0.0
    min_thickness_m: float = float("nan")
    median_thickness_m: float = float("nan")
    warnings: List[str] = field(default_factory=list)

    @property
    def bounds(self):
        return self.vertices_m.min(axis=0), self.vertices_m.max(axis=0)


def _parse_transform(s: Optional[str]) -> np.ndarray:
    """3MF transform 문자열(12개) -> 4x4 (열벡터 기준) 행렬."""
    M = np.eye(4)
    if not s:
        return M
    v = [float(t) for t in s.split()]
    if len(v) != 12:
        return M
    # 3MF: p' = [x y z 1] * [[m00 m01 m02],[m10 m11 m12],[m20 m21 m22],[m30 m31 m32]]
    A = np.array(v).reshape(4, 3)
    M[:3, :3] = A[:3, :].T
    M[:3, 3] = A[3, :]
    return M


def _apply(M: np.ndarray, V: np.ndarray) -> np.ndarray:
    return V @ M[:3, :3].T + M[:3, 3]


def _mesh_volume(V: np.ndarray, T: np.ndarray) -> float:
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def _edge_check(T: np.ndarray):
    e = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    e = np.sort(e, axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    bad = int((cnt != 2).sum())
    return bad == 0, bad


def _ray_hits(orig: np.ndarray, d: np.ndarray, V: np.ndarray, T: np.ndarray) -> np.ndarray:
    """광선(orig[k], d[k])마다 모든 삼각형과의 교차 거리 t (교차 없으면 inf). (K, M)"""
    v0, v1, v2 = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    e1, e2 = v1 - v0, v2 - v0
    out = np.full((len(orig), len(T)), np.inf)
    for s in range(0, len(orig), 64):
        o = orig[s:s + 64, None, :]
        dd = d[s:s + 64, None, :]
        p = np.cross(dd, e2[None])
        det = np.einsum("kmj,mj->km", p, e1)
        ok = np.abs(det) > 1e-18
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        tv = o - v0[None]
        u = np.einsum("kmj,kmj->km", tv, p) * inv
        q = np.cross(tv, e1[None])
        v = np.einsum("kmj,kmj->km", dd, q) * inv
        t = np.einsum("kmj,mj->km", q, e2) * inv
        hit = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > 0)
        out[s:s + 64] = np.where(hit, t, np.inf)
    return out


def estimate_thickness(V: np.ndarray, T: np.ndarray, n_samples: int = 300, seed: int = 0):
    """표면 삼각형 중심에서 안쪽으로 쏜 광선이 반대편 면에 닿는 거리 = 국소 두께.
    5번째 백분위를 '최소 두께', 중앙값을 '대표 두께'로 돌려준다."""
    if len(T) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(T), size=min(n_samples, len(T)), replace=False)
    a, b, c = V[T[idx, 0]], V[T[idx, 1]], V[T[idx, 2]]
    n = np.cross(b - a, c - a)
    nn = np.linalg.norm(n, axis=1)
    keep = nn > 0
    if not keep.any():
        return float("nan"), float("nan")
    n = n[keep] / nn[keep, None]
    cen = (a + b + c)[keep] / 3.0
    sign = 1.0 if _mesh_volume(V, T) >= 0 else -1.0   # 바깥 법선 방향 보정
    inward = -sign * n
    span = float(np.max(V.max(0) - V.min(0)))
    eps = 1e-7 * max(span, 1e-9)
    t = _ray_hits(cen + eps * inward, inward, V, T)
    tmin = t.min(axis=1)
    tmin = tmin[np.isfinite(tmin)]
    if len(tmin) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(tmin, 5)), float(np.median(tmin))


def merge_vertices(pts: np.ndarray, tol: float):
    """허용오차 기반 꼭짓점 병합(A13): KD-tree로 tol 안의 점들을 같은 점으로 묶는다(합집합-찾기).
    반환 (고유 꼭짓점, 원래→고유 인덱스)."""
    from scipy.spatial import cKDTree
    n = len(pts)
    if n == 0:
        return pts, np.zeros(0, np.int64)
    tree = cKDTree(pts)
    pairs = tree.query_pairs(r=tol, output_type="ndarray")
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([find(i) for i in range(n)])
    uniq_roots, inv = np.unique(roots, return_inverse=True)
    return pts[uniq_roots], inv.astype(np.int64)


def _hex_color(c: Optional[str]) -> str:
    if not c:
        return "#888888"
    c = c.strip()
    if re.fullmatch(r"#[0-9A-Fa-f]{6}([0-9A-Fa-f]{2})?", c):
        return c[:7]
    return "#888888"


def load_3mf(data: bytes, name_hint: str = "") -> Dict:
    """3MF 바이트를 읽어 객체 목록과 메타데이터를 돌려준다.

    반환: {"unit": str, "objects": [MeshObject], "title": str, "description": str,
           "warnings": [str]}
    """
    z = zipfile.ZipFile(io.BytesIO(data))
    model_names = [n for n in z.namelist() if n.lower().endswith(".model")]
    if not model_names:
        raise ValueError("3MF 안에 .model 파일이 없습니다.")
    # 3D/3dmodel.model 을 우선
    model_names.sort(key=lambda n: (0 if n.lower().endswith("3dmodel.model") else 1, n))
    root = ET.fromstring(z.read(model_names[0]))
    ns = root.tag.split("}")[0] + "}" if root.tag.startswith("{") else ""
    unit = root.get("unit", "millimeter")
    if unit not in UNIT_TO_M:
        unit = "millimeter"
    scale = UNIT_TO_M[unit]

    meta = {m.get("name", ""): (m.text or "") for m in root.iter(ns + "metadata")}

    # 재질 색
    colors = {}
    for bm in root.iter(ns + "basematerials"):
        pid = bm.get("id")
        colors[pid] = [_hex_color(b.get("displaycolor")) for b in bm.findall(ns + "base")]

    objs = {}
    for o in root.iter(ns + "object"):
        oid = o.get("id")
        entry = {"name": o.get("name") or f"object_{oid}", "mesh": None, "components": [],
                 "color": "#888888"}
        pid, pindex = o.get("pid"), o.get("pindex")
        if pid in colors and pindex is not None:
            try:
                entry["color"] = colors[pid][int(pindex)]
            except (ValueError, IndexError):
                pass
        m = o.find(ns + "mesh")
        if m is not None:
            vs = m.find(ns + "vertices")
            ts = m.find(ns + "triangles")
            V = np.array([[float(v.get("x")), float(v.get("y")), float(v.get("z"))]
                          for v in vs.findall(ns + "vertex")], dtype=float) if vs is not None else np.zeros((0, 3))
            tris = ts.findall(ns + "triangle") if ts is not None else []
            T = np.array([[int(t.get("v1")), int(t.get("v2")), int(t.get("v3"))] for t in tris],
                         dtype=np.int64) if tris else np.zeros((0, 3), np.int64)
            # 3MF Core 규격: 꼭짓점 인덱스는 0..n−1 범위, 한 삼각형의 세 인덱스는 서로 달라야 함 → 위반 시 오류(A14)
            if len(T):
                bad = (T < 0) | (T >= len(V))
                if bad.any():
                    raise ValueError(f"3MF 객체 '{entry['name']}': 삼각형 꼭짓점 인덱스 범위 초과 {int(bad.any(axis=1).sum())}개 "
                                     f"(꼭짓점 {len(V)}개). 규격 위반 파일이라 읽지 않습니다.")
                deg = (T[:, 0] == T[:, 1]) | (T[:, 1] == T[:, 2]) | (T[:, 0] == T[:, 2])
                if deg.any():
                    raise ValueError(f"3MF 객체 '{entry['name']}': 같은 꼭짓점을 두 번 쓴 삼각형 {int(deg.sum())}개(규격 위반).")
            # 삼각형 단위 재질(pid/p1): 객체 기본 재질을 덮어쓴다. 범위를 벗어난 재질 인덱스는 경고 후 객체 기본값 사용.
            tri_col = []
            nbad = 0
            for t in tris:
                tp = t.get("pid", pid)
                ti = t.get("p1", pindex if t.get("pid") is None else None)
                col = entry["color"]
                if tp is not None and ti is not None and tp in colors:
                    try:
                        k = int(ti)
                        if 0 <= k < len(colors[tp]):
                            col = colors[tp][k]
                        else:
                            nbad += 1
                    except ValueError:
                        nbad += 1
                tri_col.append(col)
            entry["tri_color"] = tri_col
            entry["tri_bad_material"] = nbad
            if tri_col:
                vals, cnts = np.unique(tri_col, return_counts=True)
                entry["color"] = str(vals[np.argmax(cnts)])
                entry["n_materials"] = int(len(vals))
            entry["mesh"] = (V, T)
        comps = o.find(ns + "components")
        if comps is not None:
            for c in comps.findall(ns + "component"):
                entry["components"].append((c.get("objectid"), _parse_transform(c.get("transform"))))
        objs[oid] = entry

    def flatten(oid, M, prefix, depth=0):
        if depth > 16 or oid not in objs:
            return []
        e = objs[oid]
        nm = f"{prefix}/{e['name']}" if prefix else e["name"]
        out = []
        if e["mesh"] is not None and len(e["mesh"][1]):
            V, T = e["mesh"]
            out.append((nm, _apply(M, V), T, e["color"], e.get("n_materials", 1), e.get("tri_bad_material", 0)))
        for cid, CM in e["components"]:
            out += flatten(cid, M @ CM, nm, depth + 1)
        return out

    items = []
    build = root.find(ns + "build")
    if build is not None:
        for it in build.findall(ns + "item"):
            items.append((it.get("objectid"), _parse_transform(it.get("transform"))))
    if not items:  # build가 없으면 모든 메시 객체를 그대로
        items = [(oid, np.eye(4)) for oid, e in objs.items() if e["mesh"] is not None]

    out_objs: List[MeshObject] = []
    seen = {}
    warnings = []
    for oid, M in items:
        for nm, V, T, col, nmat, nbadmat in flatten(oid, M, ""):
            k = seen.get(nm, 0)
            seen[nm] = k + 1
            label = nm if k == 0 else f"{nm}#{k + 1}"
            Vm = V * scale
            closed, bad = _edge_check(T)
            vol = _mesh_volume(Vm, T)
            tmin, tmed = estimate_thickness(Vm, T)
            mo = MeshObject(name=label, vertices_m=Vm, triangles=T, color=col,
                            source_unit=unit, closed=closed, open_edges=bad,
                            volume_m3=abs(vol), min_thickness_m=tmin, median_thickness_m=tmed)
            if not closed:
                mo.warnings.append(f"닫히지 않은 메시(열린/중복 모서리 {bad}개): 내부 판정이 부정확할 수 있습니다.")
            if vol < 0:
                mo.warnings.append("면 방향이 안쪽을 향합니다(체적 음수). 체적은 절댓값으로 처리합니다.")
            if nmat > 1:
                mo.warnings.append(f"삼각형별 재질 {nmat}종(pid/p1). 표시색은 가장 많은 재질, 체적 판정은 객체 전체로 합니다.")
            if nbadmat:
                mo.warnings.append(f"재질 인덱스 범위 초과 삼각형 {nbadmat}개 → 객체 기본 재질로 처리.")
            out_objs.append(mo)
    if not out_objs:
        raise ValueError("3MF에서 메시 객체를 찾지 못했습니다.")
    return {"unit": unit, "objects": out_objs, "title": meta.get("Title", name_hint),
            "description": meta.get("Description", ""), "warnings": warnings}


def load_stl(data: bytes, name: str = "stl") -> Dict:
    """STL(바이너리/ASCII)을 하나의 객체로 읽는다. STL에는 단위가 없으므로 mm로 가정한다."""
    if data[:5].lower() == b"solid" and b"facet" in data[:1000]:
        txt = data.decode("utf-8", "replace")
        pts = np.array([[float(x) for x in l.split()[1:4]] for l in txt.splitlines()
                        if l.strip().startswith("vertex")])
    else:
        n = int(np.frombuffer(data[80:84], dtype=np.uint32)[0])
        rec = np.frombuffer(data[84:84 + 50 * n], dtype=np.dtype([
            ("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
        pts = rec["v"].reshape(-1, 3).astype(float)
    # 중복 꼭짓점 병합: 반올림(경계에 걸치면 분리·과병합) 대신 허용오차 기반(A13). tol = 경계상자 대각선 × 1e-6
    span = float(np.linalg.norm(pts.max(0) - pts.min(0))) if len(pts) else 1.0
    tol = max(span * 1e-6, 1e-12)
    uniq, inv = merge_vertices(pts, tol)
    T = inv.reshape(-1, 3).astype(np.int64)
    deg = (T[:, 0] == T[:, 1]) | (T[:, 1] == T[:, 2]) | (T[:, 0] == T[:, 2])
    T = T[~deg]
    Vm = uniq * UNIT_TO_M["millimeter"]
    closed, bad = _edge_check(T)
    tmin, tmed = estimate_thickness(Vm, T)
    mo = MeshObject(name=name, vertices_m=Vm, triangles=T, source_unit="millimeter",
                    closed=closed, open_edges=bad, volume_m3=abs(_mesh_volume(Vm, T)),
                    min_thickness_m=tmin, median_thickness_m=tmed)
    mo.warnings.append("STL은 단위 정보가 없어 mm로 가정했습니다(가정). 단위 선택에서 바꿀 수 있습니다.")
    return {"unit": "millimeter", "objects": [mo], "title": name, "description": "",
            "warnings": []}


def load_mesh_file(data: bytes, filename: str) -> Dict:
    fn = filename.lower()
    if fn.endswith(".3mf"):
        return load_3mf(data, filename)
    if fn.endswith(".stl"):
        return load_stl(data, filename)
    raise ValueError("3MF 또는 STL 파일만 읽을 수 있습니다.")


def rescale_objects(objs: List[MeshObject], from_unit: str, to_unit: str) -> List[MeshObject]:
    """사용자가 단위를 바꿨을 때(예: 3MF에는 mm로 적혔지만 실제는 cm) 다시 스케일."""
    f = UNIT_TO_M[to_unit] / UNIT_TO_M[from_unit]
    out = []
    for o in objs:
        out.append(MeshObject(name=o.name, vertices_m=o.vertices_m * f, triangles=o.triangles,
                              color=o.color, source_unit=to_unit, closed=o.closed,
                              open_edges=o.open_edges, volume_m3=o.volume_m3 * f ** 3,
                              min_thickness_m=o.min_thickness_m * f,
                              median_thickness_m=o.median_thickness_m * f,
                              warnings=list(o.warnings)))
    return out
