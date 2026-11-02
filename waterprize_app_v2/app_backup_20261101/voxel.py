"""흙수조 격자, 구조물 배치, 객체별 복셀화, 역할 합성.

좌표계(흙수조): x∈[0,Lx], y∈[0,Ly], 깊이 d∈[0,H] (지표 d=0, 아래로 +).
3MF 좌표(z 위쪽, z=0 지표)는 배치(위치·상단 깊이·회전·기울기·배율)로 흙수조 좌표로 옮긴다.
배율은 사용자가 명시할 때만 1이 아니다(자동 축소 없음).

얇은 벽(불투수·저투수 실링)은 셀을 차지시키지 않고 '셀 사이 면'에 장벽으로 넣는다.
(유한체적법에서 셀보다 얇은 벽을 표현하는 표준 방법. 벽에 둘러싸인 반응재 셀이 사라지지 않는다.)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from params import ROLE_DEFAULTS, ROLES

BARRIER_ROLES = ("불투수", "저투수 실링")


@dataclass
class Grid:
    nx: int
    ny: int
    nz: int
    Lx: float
    Ly: float
    H: float            # 전체 깊이 (토양층 + 지하수층)
    H_soil: float

    @property
    def dx(self): return self.Lx / self.nx
    @property
    def dy(self): return self.Ly / self.ny
    @property
    def dz(self): return self.H / self.nz
    @property
    def h(self): return np.array([self.dx, self.dy, self.dz])
    @property
    def shape(self): return (self.nx, self.ny, self.nz)
    @property
    def N(self): return self.nx * self.ny * self.nz
    @property
    def vcell(self): return self.dx * self.dy * self.dz
    @property
    def xc(self): return (np.arange(self.nx) + 0.5) * self.dx
    @property
    def yc(self): return (np.arange(self.ny) + 0.5) * self.dy
    @property
    def dc(self): return (np.arange(self.nz) + 0.5) * self.dz
    @property
    def k_interface(self) -> int:
        """토양층/지하수층 경계면 바로 아래 셀 인덱스(경계면 = 이 셀의 윗면). 지하수층이 없으면 nz."""
        if self.H_soil >= self.H - 1e-12:
            return self.nz
        return int(min(self.nz - 1, max(1, round(self.H_soil / self.dz))))
    @property
    def interface_depth(self) -> float:
        return self.k_interface * self.dz


def make_grid(c) -> Grid:
    H = c.H_soil + max(0.0, c.H_aq)
    return Grid(int(c.nx), int(c.ny), int(c.nz), float(c.Lx), float(c.Ly), float(H), float(c.H_soil))


@dataclass
class Placement:
    x: float = 0.25          # 3MF 원점(보통 구조물 축)의 흙수조 x
    y: float = 0.25
    top_depth: float = 0.0   # 3MF z=0 평면의 깊이(0 = 지표)
    rot_deg: float = 0.0     # 연직축 회전
    tilt_deg: float = 0.0    # x축 기준 기울기
    scale: float = 1.0       # 사용자가 명시한 배율만 적용


def to_tank(V: np.ndarray, pl: Placement) -> np.ndarray:
    """3MF(미터) 꼭짓점 -> 흙수조 좌표 (x, y, depth)."""
    P = V * pl.scale
    t = np.deg2rad(pl.tilt_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    r = np.deg2rad(pl.rot_deg)
    Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    P = P @ (Rz @ Rx).T
    out = np.empty_like(P)
    out[:, 0] = pl.x + P[:, 0]
    out[:, 1] = pl.y + P[:, 1]
    out[:, 2] = pl.top_depth - P[:, 2]
    return out


def _line_hits(Vt, T, axis, pu, pv):
    """axis 방향 직선들(다른 두 축 좌표 pu, pv)과 메시 교차점.
    반환: (line_index, 교차 좌표) — 좌표는 axis 방향 값."""
    u_ax, v_ax = [a for a in range(3) if a != axis]
    v0, v1, v2 = Vt[T[:, 0]], Vt[T[:, 1]], Vt[T[:, 2]]
    det = (v1[:, u_ax] - v0[:, u_ax]) * (v2[:, v_ax] - v0[:, v_ax]) - \
          (v2[:, u_ax] - v0[:, u_ax]) * (v1[:, v_ax] - v0[:, v_ax])
    good = np.abs(det) > 1e-22
    v0, v1, v2, det = v0[good], v1[good], v2[good], det[good]
    if len(det) == 0 or len(pu) == 0:
        return np.zeros(0, int), np.zeros(0)
    umin = np.minimum(np.minimum(v0[:, u_ax], v1[:, u_ax]), v2[:, u_ax])
    umax = np.maximum(np.maximum(v0[:, u_ax], v1[:, u_ax]), v2[:, u_ax])
    vmin = np.minimum(np.minimum(v0[:, v_ax], v1[:, v_ax]), v2[:, v_ax])
    vmax = np.maximum(np.maximum(v0[:, v_ax], v1[:, v_ax]), v2[:, v_ax])
    out_l, out_x = [], []
    chunk = max(1, int(3_000_000 // max(1, len(det))))
    for s in range(0, len(pu), chunk):
        a = pu[s:s + chunk, None]
        b = pv[s:s + chunk, None]
        cand = (a >= umin) & (a <= umax) & (b >= vmin) & (b <= vmax)
        li, ti = np.nonzero(cand)
        if len(li) == 0:
            continue
        pa, pb = pu[s + li], pv[s + li]
        A0, A1, A2, D = v0[ti], v1[ti], v2[ti], det[ti]
        l1 = ((pa - A0[:, u_ax]) * (A2[:, v_ax] - A0[:, v_ax]) - (A2[:, u_ax] - A0[:, u_ax]) * (pb - A0[:, v_ax])) / D
        l2 = ((A1[:, u_ax] - A0[:, u_ax]) * (pb - A0[:, v_ax]) - (pa - A0[:, u_ax]) * (A1[:, v_ax] - A0[:, v_ax])) / D
        l0 = 1.0 - l1 - l2
        ins = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
        if not ins.any():
            continue
        xh = l0[ins] * A0[ins, axis] + l1[ins] * A1[ins, axis] + l2[ins] * A2[ins, axis]
        out_l.append(s + li[ins])
        out_x.append(xh)
    if not out_l:
        return np.zeros(0, int), np.zeros(0)
    return np.concatenate(out_l), np.concatenate(out_x)


def _count_below(hl, hx, nL, q):
    """각 직선 l에서 좌표 q[l, :]보다 작은 교차점 개수."""
    if len(hl) == 0:
        return np.zeros(q.shape, dtype=np.int64)
    lo = min(hx.min(), q.min()) - 1.0
    span = max(hx.max(), q.max()) - lo + 1.0
    key = np.sort(hl * span + (hx - lo))
    starts = np.searchsorted(key, np.arange(nL) * span)
    qk = np.arange(nL)[:, None] * span + (q - lo)
    return np.searchsorted(key, qk.ravel()).reshape(q.shape) - starts[:, None]


def parity_fraction(Vt, T, g: Grid, sub: int = 2) -> np.ndarray:
    """객체 하나의 셀별 체적분율(+x 방향 짝홀 판정, 셀당 sub^3 표본)."""
    frac = np.zeros(g.shape, dtype=np.float32)
    lo, hi = Vt.min(0), Vt.max(0)
    i0 = max(0, int(np.floor(lo[0] / g.dx))); i1 = min(g.nx, int(np.ceil(hi[0] / g.dx)))
    j0 = max(0, int(np.floor(lo[1] / g.dy))); j1 = min(g.ny, int(np.ceil(hi[1] / g.dy)))
    k0 = max(0, int(np.floor(lo[2] / g.dz))); k1 = min(g.nz, int(np.ceil(hi[2] / g.dz)))
    if i1 <= i0 or j1 <= j0 or k1 <= k0:
        return frac
    offs = (np.arange(sub) + 0.5) / sub
    ys = ((np.arange(j0, j1)[:, None] + offs[None]) * g.dy).ravel() + 1.234567e-7 * g.dy
    ds = ((np.arange(k0, k1)[:, None] + offs[None]) * g.dz).ravel() + 2.345678e-7 * g.dz
    xs = ((np.arange(i0, i1)[:, None] + offs[None]) * g.dx).ravel()
    YY, DD = np.meshgrid(ys, ds, indexing="ij")
    pu, pv = YY.ravel(), DD.ravel()
    hl, hx = _line_hits(Vt, T, 0, pu, pv)
    cnt = _count_below(hl, hx, len(pu), np.broadcast_to(xs, (len(pu), len(xs))))
    inside = (cnt % 2 == 1).astype(np.float32)
    nj, nk, ni = j1 - j0, k1 - k0, i1 - i0
    f = inside.reshape(nj, sub, nk, sub, ni, sub).mean(axis=(1, 3, 5))
    frac[i0:i1, j0:j1, k0:k1] = np.transpose(f, (2, 0, 1))
    return frac


def face_crossings(Vt, T, g: Grid):
    """메시 표면이 이웃 셀 중심을 잇는 선분을 가로지르는 면(=장벽 면).
    반환: (bx[nx-1,ny,nz], by[nx,ny-1,nz], bz[nx,ny,nz-1], btop[nx,ny]) bool"""
    lo, hi = Vt.min(0), Vt.max(0)
    cs = [g.xc, g.yc, g.dc]
    res = []
    for axis in range(3):
        u_ax, v_ax = [a for a in range(3) if a != axis]
        cu, cv = cs[u_ax], cs[v_ax]
        su = np.where((cu >= lo[u_ax] - g.h[u_ax]) & (cu <= hi[u_ax] + g.h[u_ax]))[0]
        sv = np.where((cv >= lo[v_ax] - g.h[v_ax]) & (cv <= hi[v_ax] + g.h[v_ax]))[0]
        shape = list(g.shape); shape[axis] -= 1
        B = np.zeros(shape, dtype=bool)
        top = None
        if axis == 2:
            top = np.zeros((g.nx, g.ny), dtype=bool)
        if len(su) == 0 or len(sv) == 0:
            res.append(B)
            if axis == 2:
                res.append(top)
            continue
        UU, VV = np.meshgrid(cu[su] + 1.1e-7 * g.h[u_ax], cv[sv] + 2.3e-7 * g.h[v_ax], indexing="ij")
        pu, pv = UU.ravel(), VV.ravel()
        hl, hx = _line_hits(Vt, T, axis, pu, pv)
        c = cs[axis]
        q = np.broadcast_to(c, (len(pu), len(c)))
        cnt = _count_below(hl, hx, len(pu), q)
        cross = (cnt[:, 1:] - cnt[:, :-1]) > 0          # (nL, n_axis-1)
        cross = cross.reshape(len(su), len(sv), -1)
        if axis == 0:
            B[:, su[:, None], sv[None, :]] = np.transpose(cross, (2, 0, 1))
        elif axis == 1:
            B[su[:, None], :, sv[None, :]] = cross  # 고급 인덱스가 떨어져 있으면 (su, sv, ny-1) 순서
        else:
            B[su[:, None], sv[None, :], :] = cross
            # 지표면(깊이 0)과 첫 셀 중심 사이
            ctop = cnt[:, 0] > _count_below(hl, hx, len(pu), np.full((len(pu), 1), -1e-9))[:, 0]
            top[su[:, None], sv[None, :]] = ctop.reshape(len(su), len(sv))
        res.append(B)
        if axis == 2:
            res.append(top)
    return res[0], res[1], res[2], res[3]


def surface_cells(Vt, T, g: Grid) -> np.ndarray:
    """메시 표면이 지나가는 셀(얇은 개구부·반응재 보존용)."""
    mask = np.zeros(g.shape, dtype=bool)
    hstep = min(g.dx, g.dy, g.dz) / 3.0
    v0, v1, v2 = Vt[T[:, 0]], Vt[T[:, 1]], Vt[T[:, 2]]
    L = np.maximum(np.maximum(np.linalg.norm(v1 - v0, axis=1), np.linalg.norm(v2 - v0, axis=1)),
                   np.linalg.norm(v2 - v1, axis=1))
    m_all = np.clip(np.ceil(L / hstep).astype(int), 1, 400)
    for m in np.unique(m_all):
        sel = m_all == m
        a = np.arange(m + 1)
        I, J = np.meshgrid(a, a, indexing="ij")
        keep = (I + J) <= m
        u = (I[keep] / m)[None, :, None]
        w = (J[keep] / m)[None, :, None]
        P = (v0[sel][:, None, :] * (1 - u - w) + v1[sel][:, None, :] * u + v2[sel][:, None, :] * w).reshape(-1, 3)
        ii = np.floor(P[:, 0] / g.dx).astype(int)
        jj = np.floor(P[:, 1] / g.dy).astype(int)
        kk = np.floor(P[:, 2] / g.dz).astype(int)
        ok = (ii >= 0) & (ii < g.nx) & (jj >= 0) & (jj < g.ny) & (kk >= 0) & (kk < g.nz)
        mask[ii[ok], jj[ok], kk[ok]] = True
    return mask


@dataclass
class ObjectVoxel:
    name: str
    role: str
    frac: np.ndarray              # 체적분율 (float32)
    member: np.ndarray            # 셀 소속(bool)
    faces: Optional[tuple]        # 장벽 면 (bx, by, bz, btop) — 장벽 역할만
    n_cells: int
    n_faces: int
    used_surface: bool
    clipped: bool
    thin: bool
    warnings: List[str] = field(default_factory=list)


def voxelize_objects(objects, roles: Dict[str, dict], placement: Placement, g: Grid,
                     sub: int = 2) -> Tuple[List[ObjectVoxel], List[str]]:
    """objects: io_3mf.MeshObject 목록, roles: {이름: {role, thin_mode, ...}}"""
    out, warns = [], []
    hmax = float(max(g.dx, g.dy, g.dz))
    for o in objects:
        r = roles.get(o.name, {})
        role = r.get("role", "불투수")
        Vt = to_tank(o.vertices_m, placement)
        lo, hi = Vt.min(0), Vt.max(0)
        clipped = bool((lo[0] < -1e-9) or (lo[1] < -1e-9) or (hi[0] > g.Lx + 1e-9) or (hi[1] > g.Ly + 1e-9)
                       or (lo[2] < -1e-6) or (hi[2] > g.H + 1e-9))
        frac = parity_fraction(Vt, o.triangles, g)
        member = frac >= 0.5
        thick = o.min_thickness_m * placement.scale
        thin = bool(np.isfinite(thick) and thick < hmax)
        w = []
        if thin:
            w.append(f"'{o.name}': 최소 두께 {thick*1000:.2f} mm < 격자 {hmax*1000:.1f} mm.")
        faces, n_faces, used_surface = None, 0, False
        thin_mode = r.get("thin_mode", "자동")
        if role in BARRIER_ROLES:
            faces = face_crossings(Vt, o.triangles, g)
            n_faces = int(sum(int(f.sum()) for f in faces))
            if thin:
                w.append(f"'{o.name}': 얇은 벽을 셀 사이 면 장벽 {n_faces}개로 넣었습니다(셀은 차지하지 않음).")
        elif role != "토양":
            if thin_mode == "면 래스터화" or (thin_mode == "자동" and (thin or member.sum() == 0)):
                surf = surface_cells(Vt, o.triangles, g)
                if surf.any() and (thin or member.sum() == 0):
                    member = member | surf
                    used_surface = True
                    w.append(f"'{o.name}': 표면이 지나는 셀로 보존(실제보다 두껍게 표현됨 → 질량은 메시 체적 기준으로 보정).")
        if clipped:
            w.append(f"'{o.name}': 일부가 흙수조 밖에 있어 잘렸습니다(자동 축소 안 함).")
        ov = ObjectVoxel(name=o.name, role=role, frac=frac, member=member, faces=faces,
                         n_cells=int(member.sum()), n_faces=n_faces, used_surface=used_surface,
                         clipped=clipped, thin=thin, warnings=w)
        out.append(ov)
        warns += w
    return out, warns


def compose(obj_vox: List[ObjectVoxel], roles: Dict[str, dict], g: Grid):
    """역할 우선순위로 겹침을 정리.

    반환 dict:
      role_grid(int8, -1=토양), obj_grid(int16, -1=없음), counts{이름: 최종 셀수},
      barrier(bx,by,bz,btop: float, 0=개방, inf=불투수, 그 외=직렬 저항 t/K [s])
    """
    role_grid = np.full(g.shape, -1, dtype=np.int8)
    obj_grid = np.full(g.shape, -1, dtype=np.int16)
    prio_grid = np.zeros(g.shape, dtype=np.int16)
    for idx, ov in enumerate(obj_vox):
        r = roles.get(ov.name, {})
        role = r.get("role", ov.role)
        prio = int(r.get("prio", ROLE_DEFAULTS[role]["prio"]))
        if role == "토양":
            continue
        sel = ov.member & (prio > prio_grid)
        role_grid[sel] = ROLES.index(role)
        obj_grid[sel] = idx
        prio_grid[sel] = prio
    nx, ny, nz = g.shape
    bx = np.zeros((nx - 1, ny, nz)); by = np.zeros((nx, ny - 1, nz))
    bz = np.zeros((nx, ny, nz - 1)); bt = np.zeros((nx, ny))
    for ov in obj_vox:
        if ov.faces is None:
            continue
        r = roles.get(ov.name, {})
        role = r.get("role", ov.role)
        if role == "불투수":
            R = np.inf
        else:
            K = float(r.get("K", ROLE_DEFAULTS[role]["K"]) or 1e-10)
            t = float(r.get("thickness_m", 2e-3))
            R = t / max(K, 1e-30)
        for B, F in zip((bx, by, bz, bt), ov.faces):
            B[F] = np.maximum(B[F], R)
    counts = {ov.name: int((obj_grid == i).sum()) for i, ov in enumerate(obj_vox)}
    return {"role_grid": role_grid, "obj_grid": obj_grid, "counts": counts,
            "barrier": (bx, by, bz, bt)}
