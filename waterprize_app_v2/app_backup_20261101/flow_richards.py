"""3D 불포화 흐름: Richards 방정식(혼합형), van Genuchten–Mualem, 셀 중심 유한체적.

  ∂θ(h)/∂t = ∇·[K(h) ∇(h + z)] + 원천

- 면 투수계수: 두 셀 K(h)의 조화평균(직렬 저항 합) + 얇은 벽 장벽 저항(직렬).
- 시간 적분: 후진 오일러 + 수정 Picard 반복(Celia et al. 1990, 질량 보존형).
- 선형계: scipy.sparse, Jacobi 전처리 켤레기울기법(실패 시 직접법).
- 경계: 상단 = 지정 플럭스(지표 포화 시 h=0 고정으로 전환), 하단 = 지하수 수위(고정 수두) 또는
  자유배수(단위 동수경사), 측면 = 무유동(지하수층에서 지하수 유속을 주면 x 방향 고정 수두 기울기).
- 정상 상태는 의사-과도 연속법(Δt를 키워 가며 수렴)으로 푼다.
불투수 셀은 계산에서 빠진다(흐름 0).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from scipy.sparse.csgraph import connected_components

SS = 1e-5          # 포화 비저류 [1/m] (수치 안정용, 가정)
KMIN = 1e-30


def vg_theta(h, thr, ths, alpha, n):
    m = 1.0 - 1.0 / n
    ah = np.abs(alpha * np.minimum(h, 0.0))
    Se = (1.0 + ah ** n) ** (-m)
    return thr + (ths - thr) * Se


def vg_Se(h, alpha, n):
    m = 1.0 - 1.0 / n
    ah = np.abs(alpha * np.minimum(h, 0.0))
    return (1.0 + ah ** n) ** (-m)


def vg_kr(h, alpha, n):
    m = 1.0 - 1.0 / n
    Se = np.clip(vg_Se(h, alpha, n), 1e-12, 1.0)
    return np.sqrt(Se) * (1.0 - (1.0 - Se ** (1.0 / m)) ** m) ** 2


def vg_C(h, thr, ths, alpha, n):
    m = 1.0 - 1.0 / n
    hn = np.minimum(h, 0.0)
    ah = np.abs(alpha * hn)
    C = (ths - thr) * alpha * n * m * ah ** (n - 1.0) * (1.0 + ah ** n) ** (-m - 1.0)
    return np.where(h < 0, C, 0.0) + SS


def vg_h_from_Se(Se, alpha, n):
    m = 1.0 - 1.0 / n
    Se = np.clip(Se, 1e-6, 1.0)
    return -((Se ** (-1.0 / m) - 1.0) ** (1.0 / n)) / alpha


def h_for_flux(q, Ks, alpha, n):
    """K(h) = q 를 만족하는 압력수두(단위 동수경사 정상해). q>=Ks면 0."""
    q = np.broadcast_to(np.asarray(q, float), np.shape(Ks))
    lo = np.full(np.shape(Ks), -1e4); hi = np.zeros(np.shape(Ks))
    for _ in range(60):
        mid = -np.sqrt(-lo * np.maximum(-hi, 1e-12)) if False else 0.5 * (lo + hi)
        k = Ks * vg_kr(mid, alpha, n)
        big = k > q
        hi = np.where(big, mid, hi)
        lo = np.where(big, lo, mid)
    return np.where(q >= Ks, 0.0, 0.5 * (lo + hi))


@dataclass
class FlowBC:
    q_top: object = 0.0                # m/s, 지표 침투 플럭스(아래로 +). 스칼라 또는 지표 셀별 배열(nx*ny)
    bottom: str = "지하수 수위"          # 또는 "자유배수"
    z_wt: float = 0.0                  # 지하수면 표고 m (표고 0 = 도메인 바닥)
    gw_grad: float = 0.0               # 지하수층 x 방향 동수경사(+ = x 증가 방향으로 흐름)
    wt_k: int = 0                      # 측면 고정수두를 줄 첫 셀 층(지하수면 아래)
    src: Optional[np.ndarray] = None   # 셀별 유입량 m3/s (주입 펌프)
    fixed_h: Optional[np.ndarray] = None   # 셀별 고정 압력수두(nan=자유)


@dataclass
class FlowState:
    h: np.ndarray
    theta: np.ndarray
    Fx: np.ndarray        # (nx-1,ny,nz) m3/s, +x
    Fy: np.ndarray        # (nx,ny-1,nz) +y
    Fz: np.ndarray        # (nx,ny,nz-1) +깊이(아래로)
    Ftop: np.ndarray      # (nx,ny) 지표에서 들어오는 양(+)
    Fbot: np.ndarray      # (nx,ny) 바닥으로 나가는 양(+)
    Fleft: np.ndarray     # (ny,nz) x=0 면을 통해 들어오는 양(+)
    Fright: np.ndarray    # (ny,nz) x=Lx 면을 통해 나가는 양(+)
    Fsrc: np.ndarray      # (nx,ny,nz) 셀 원천(+)
    ponded: np.ndarray    # (nx,ny) 지표 포화(유출) 셀
    runoff: float = 0.0   # m3/s 지표에서 못 들어간 양
    info: Dict = field(default_factory=dict)


class RichardsFV:
    def __init__(self, g, active, Ks_h, Ks_v, thr, ths, alpha, n, barrier):
        self.g = g
        self.shape = g.shape
        self.active = active.astype(bool)
        self.set_material(Ks_h, Ks_v, thr, ths, alpha, n)
        self.barrier = barrier
        self.verbose = False
        self._build_topology()

    # ── 재료 ─────────────────────────────────────────────────────────
    def set_material(self, Ks_h, Ks_v, thr, ths, alpha, n):
        f = lambda a: np.asarray(a, dtype=float).ravel()
        self.Ks_h, self.Ks_v = f(Ks_h), f(Ks_v)
        self.thr, self.ths, self.alpha, self.n = f(thr), f(ths), f(alpha), f(n)

    def update_Ks(self, Ks_h, Ks_v, ths=None):
        self.Ks_h = np.asarray(Ks_h, float).ravel()
        self.Ks_v = np.asarray(Ks_v, float).ravel()
        if ths is not None:
            self.ths = np.maximum(np.asarray(ths, float).ravel(), self.thr + 1e-3)

    # ── 위상 ─────────────────────────────────────────────────────────
    def _build_topology(self):
        g = self.g
        nx, ny, nz = self.shape
        idx = np.arange(g.N).reshape(self.shape)
        act = self.active
        self.cmap = -np.ones(g.N, dtype=np.int64)
        self.cells = np.where(act.ravel())[0]
        self.cmap[self.cells] = np.arange(len(self.cells))
        self.nA = len(self.cells)
        bx, by, bz, bt = self.barrier
        faces = []   # (a_full, b_full, area, dist, axis, R, face_flat_index_in_axis_array)
        geo = [(g.dy * g.dz, g.dx), (g.dx * g.dz, g.dy), (g.dx * g.dy, g.dz)]
        for ax, B in enumerate((bx, by, bz)):
            sa = [slice(None)] * 3; sb = [slice(None)] * 3
            sa[ax] = slice(0, -1); sb[ax] = slice(1, None)
            a = idx[tuple(sa)].ravel(); b = idx[tuple(sb)].ravel()
            R = B.ravel()
            ok = act.ravel()[a] & act.ravel()[b] & ~np.isinf(R)
            fid = np.arange(len(a))
            faces.append(dict(a=a[ok], b=b[ok], A=geo[ax][0], d=geo[ax][1], ax=ax, R=R[ok], fid=fid[ok],
                              shape=B.shape))
        self.faces = faces
        self.fa = np.concatenate([self.cmap[f["a"]] for f in faces])
        self.fb = np.concatenate([self.cmap[f["b"]] for f in faces])
        self.fA = np.concatenate([np.full(len(f["a"]), f["A"]) for f in faces])
        self.fd = np.concatenate([np.full(len(f["a"]), f["d"]) for f in faces])
        self.fR = np.concatenate([f["R"] for f in faces])
        self.fax = np.concatenate([np.full(len(f["a"]), f["ax"]) for f in faces])
        self.nF = len(self.fa)
        # 경계 셀
        top = idx[:, :, 0].ravel()
        self.top_ok = act.ravel()[top] & ~np.isinf(bt.ravel())
        self.top_cells = top
        self.top_R = np.where(np.isinf(bt.ravel()), 0.0, bt.ravel())
        bot = idx[:, :, -1].ravel()
        self.bot_cells = bot
        self.bot_ok = act.ravel()[bot]
        self.left_cells = idx[0, :, :].ravel()
        self.right_cells = idx[-1, :, :].ravel()
        self.z = (g.H - (np.arange(nz) + 0.5) * g.dz)[None, None, :] * np.ones(self.shape)
        self.z = self.z.ravel()
        self.V = g.vcell
        # 행렬 패턴
        # 연결 성분(벽으로 막힌 구역 판별)
        adj = sp.csr_matrix((np.ones(self.nF), (self.fa, self.fb)), shape=(self.nA, self.nA))
        self.ncomp, self.comp = connected_components(adj, directed=False)
        self.comp_top = np.zeros(self.ncomp, bool)
        tcm = self.cmap[self.top_cells]
        self.comp_top[self.comp[tcm[self.top_ok]]] = True
        self.comp_bot = np.zeros(self.ncomp, bool)
        bcm = self.cmap[self.bot_cells]
        self.comp_bot[self.comp[bcm[self.bot_ok]]] = True
        self.comp_size = np.bincount(self.comp, minlength=self.ncomp)

    # ── 보조 ─────────────────────────────────────────────────────────
    def K_cells(self, h, axis):
        Ks = self.Ks_v if axis == 2 else self.Ks_h
        return np.maximum(Ks * vg_kr(h, self.alpha, self.n), KMIN)

    def _cell(self, arr):
        return arr[self.cells]

    # ── 뉴턴 계 조립 ────────────────────────────────────────────────
    def _K_and_dK(self, h, axis):
        c = self.cells
        Ks = (self.Ks_v if axis == 2 else self.Ks_h)[c]
        al, n = self.alpha[c], self.n[c]
        eps = 1e-6 * np.maximum(1.0, np.abs(h))
        K = np.maximum(Ks * vg_kr(h, al, n), KMIN)
        Kp = Ks * vg_kr(h + eps, al, n)
        Km = Ks * vg_kr(h - eps, al, n)
        dK = np.where(h < 0, (Kp - Km) / (2 * eps), 0.0)
        return K, dK

    def _system(self, h, theta_old, dt, bc: FlowBC, ponded, want_J=True, picard=False):
        """잔차 G(h) = 순유입 − 저류변화 (=0이 해)와 야코비안 J = ∂G/∂h."""
        g = self.g
        c = self.cells
        thr, ths, al, n = self.thr[c], self.ths[c], self.alpha[c], self.n[c]
        th = vg_theta(h, thr, ths, al, n)
        H = h + self.z[c]
        G = np.zeros(self.nA)
        dg = np.zeros(self.nA)
        rows, cols, vals = [], [], []
        KD = [self._K_and_dK(h, ax) for ax in range(3)]
        for ax in range(3):
            sel = self.fax == ax
            if not sel.any():
                continue
            a, b = self.fa[sel], self.fb[sel]
            K, dK = KD[ax]
            Ka, Kb, dKa, dKb = K[a], K[b], dK[a], dK[b]
            d, A, Rb = self.fd[sel], self.fA[sel], self.fR[sel]
            den = 0.5 * d / Ka + 0.5 * d / Kb + Rb
            T = A / den
            dHab = H[a] - H[b]
            flux = T * dHab
            G += np.bincount(b, weights=flux, minlength=self.nA) - np.bincount(a, weights=flux, minlength=self.nA)
            if want_J:
                dTa = A * (0.5 * d / Ka ** 2) / den ** 2
                dTb = A * (0.5 * d / Kb ** 2) / den ** 2
                if picard:
                    dfa = T; dfb = -T
                else:
                    dfa = T + dHab * dTa * dKa
                    dfb = -T + dHab * dTb * dKb
                dg += np.bincount(b, weights=dfb, minlength=self.nA) - np.bincount(a, weights=dfa, minlength=self.nA)
                rows += [a, b]; cols += [b, a]; vals += [-dfb, dfa]
        if np.isfinite(dt):
            G -= self.V * (th - theta_old) / dt
            dg -= self.V * vg_C(h, thr, ths, al, n) / dt
        A_top = g.dx * g.dy
        Kz, dKz = KD[2]
        tc = self.cmap[self.top_cells]
        okt = self.top_ok
        fl = okt & ~ponded
        qv = np.broadcast_to(np.asarray(bc.q_top, float), (len(tc),))
        G[tc[fl]] += qv[fl] * A_top
        pd = okt & ponded
        if pd.any():
            ii = tc[pd]
            den = 0.5 * g.dz / Kz[ii] + self.top_R[pd]
            Tb = A_top / den
            dH = g.H - H[ii]
            G[ii] += Tb * dH
            dg[ii] += -Tb + (0.0 if picard else dH * A_top * (0.5 * g.dz / Kz[ii] ** 2) / den ** 2 * dKz[ii])
        bcells = self.cmap[self.bot_cells]
        okb = self.bot_ok
        ii = bcells[okb]
        if bc.bottom == "자유배수":
            G[ii] -= Kz[ii] * A_top
            if not picard:
                dg[ii] -= dKz[ii] * A_top
        else:
            xb = (np.arange(g.nx) + 0.5) * g.dx
            Hb = np.repeat(bc.z_wt - bc.gw_grad * (xb - g.Lx / 2.0), g.ny)[okb]
            Tb = 2.0 * A_top * Kz[ii] / g.dz
            dH = Hb - H[ii]
            G[ii] += Tb * dH
            dg[ii] += -Tb + (0.0 if picard else dH * 2.0 * A_top / g.dz * dKz[ii])
        if bc.gw_grad != 0.0:
            Ax = g.dy * g.dz
            Kx, dKx = KD[0]
            kk = np.tile(np.arange(g.nz), g.ny)
            for cells_full, x_face in ((self.left_cells, 0.0), (self.right_cells, g.Lx)):
                cm = self.cmap[cells_full]
                ok = (cm >= 0) & (kk >= bc.wt_k)
                ii = cm[ok]
                Hl = bc.z_wt - bc.gw_grad * (x_face - g.Lx / 2.0)
                Tb = 2.0 * Ax * Kx[ii] / g.dx
                dH = Hl - H[ii]
                G[ii] += Tb * dH
                dg[ii] += -Tb + (0.0 if picard else dH * 2.0 * Ax / g.dx * dKx[ii])
        if bc.src is not None:
            G += bc.src.ravel()[c]
        J = None
        if want_J:
            rows.append(np.arange(self.nA)); cols.append(np.arange(self.nA)); vals.append(dg)
            J = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                              shape=(self.nA, self.nA))
        return G, J, th

    def _linsolve(self, A, b, spd=False):
        """A x = b. 대칭 대각 스케일링 후 (대칭이면 CG) BiCGSTAB → LGMRES → 직접법."""
        if not np.any(b):
            return np.zeros_like(b)
        d = np.abs(A.diagonal())
        d = np.where(d > 0, d, 1.0)
        sc = 1.0 / np.sqrt(d)
        D = sp.diags(sc)
        As = (D @ A @ D).tocsr()
        bs = b * sc
        if spd:
            y, info = spl.cg(As, bs, rtol=1e-10, atol=0.0, maxiter=3000)
            if info == 0 and np.all(np.isfinite(y)):
                return y * sc
        y, info = spl.bicgstab(As, bs, rtol=1e-9, atol=0.0, maxiter=2000)
        if info != 0 or not np.all(np.isfinite(y)):
            y, info = spl.lgmres(As, bs, rtol=1e-9, atol=0.0, maxiter=200)
        if info != 0 or not np.all(np.isfinite(y)):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                x = spl.spsolve(A.tocsc(), b)
            if not np.all(np.isfinite(x)):
                # 특이 행렬(고립된 마른 셀 등): 아주 작은 대각 이동으로 정칙화
                eps = 1e-10 * float(np.max(np.abs(A.diagonal())) or 1.0)
                Ar = (A + sp.diags(np.full(A.shape[0], eps))).tocsc()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    x = spl.spsolve(Ar, b)
                if not np.all(np.isfinite(x)):
                    x = np.zeros_like(b)
            return x
        return y * sc

    def region_status(self, bc: FlowBC):
        """성분별 배출구 여부. 반환 (outlet[ncomp], dead_cells[nA], closed_top_cells[ntop])"""
        outlet = self.comp_bot.copy()
        if bc.gw_grad != 0.0:
            kk = np.tile(np.arange(self.g.nz), self.g.ny)
            for cells_full in (self.left_cells, self.right_cells):
                cm = self.cmap[cells_full]
                ok = (cm >= 0) & (kk >= bc.wt_k)
                outlet[self.comp[cm[ok]]] = True
        if bc.fixed_h is not None:
            fx = np.isfinite(bc.fixed_h.ravel()[self.cells])
            outlet[np.unique(self.comp[fx])] = True
        if bc.src is not None:
            sx = bc.src.ravel()[self.cells] < 0
            outlet[np.unique(self.comp[sx])] = True
        dead_comp = ~outlet & ~self.comp_top
        dead = dead_comp[self.comp]
        tcm = self.cmap[self.top_cells]
        closed_top = np.zeros(len(tcm), bool)
        okt = self.top_ok
        closed_top[okt] = ~outlet[self.comp[tcm[okt]]]
        return outlet, dead, closed_top

    def step(self, h_full, dt, bc: FlowBC, ponded=None, max_iter=25, tol=1e-7):
        """Δt 한 단계(Δt=inf면 정상해). 뉴턴 + 선 탐색. 반환 (h_new, 수렴, 반복수, ponded)."""
        c = self.cells
        h = h_full.ravel()[c].copy()
        theta_old = vg_theta(h, self.thr[c], self.ths[c], self.alpha[c], self.n[c])
        if ponded is None:
            ponded = np.zeros(len(self.top_cells), dtype=bool)
        free = np.ones(self.nA, dtype=bool)
        if bc.fixed_h is not None:
            fh = bc.fixed_h.ravel()[c]
            fx = np.isfinite(fh)
            h[fx] = fh[fx]
            free = ~fx
        locked_ponded = np.zeros(len(ponded), bool)
        if not np.isfinite(dt):
            # 정상해: 배출구 없는 구역 = 물이 가득 찬 정수압 상태(지표에서 넘침, 통과 흐름 0)는 정확해로 고정.
            outlet, dead, closed_top = self.region_status(bc)
            closed = ~outlet[self.comp]
            topc = self.comp_top[self.comp]
            fill = closed & topc
            h[fill] = self.g.H - self.z[c][fill]          # H = 지표 표고 (h = 깊이)
            free = free & ~closed
            ponded = ponded | closed_top
            locked_ponded = closed_top
        conv, it = False, 0
        total_it = 0
        h_start = h.copy()
        use_picard = np.isfinite(dt)        # 과도: 수정 Picard(대칭, 빠름) → 정체 시 뉴턴
        for sw in range(6):          # 지표 포화 전환 반복
            G, J, _ = self._system(h, theta_old, dt, bc, ponded, picard=use_picard)
            for it in range(1, max_iter + 1):
                total_it += 1
                if use_picard and it == 10:
                    use_picard = False
                    G, J, _ = self._system(h, theta_old, dt, bc, ponded, picard=False)
                Jf = J[free][:, free] if not free.all() else J
                dh = np.zeros(self.nA)
                dh[free] = self._linsolve(-Jf, G[free], spd=use_picard)
                mx = float(np.max(np.abs(dh))) if len(dh) else 0.0
                if mx > 1.0:
                    dh *= 1.0 / mx
                g0 = np.linalg.norm(G[free])
                lam = 1.0
                for _ls in range(8 if not use_picard else 1):
                    hn = h + lam * dh
                    Gn, Jn, _ = self._system(hn, theta_old, dt, bc, ponded, picard=use_picard)
                    if use_picard or np.linalg.norm(Gn[free]) <= (1 - 1e-4 * lam) * g0 or g0 == 0:
                        break
                    lam *= 0.5
                h, G, J = hn, Gn, Jn
                if self.verbose:
                    print(f'    newton it={it} |dh|max={mx:.2e} lam={lam:.3f} |G|={np.linalg.norm(G[free]):.2e} ponded={int(ponded.sum())}', flush=True)
                if mx * lam < tol:
                    conv = True
                    break
            if not conv:
                break
            tc = self.cmap[self.top_cells]
            okt = self.top_ok
            ht = np.full(len(tc), -1.0)
            ht[okt] = h[tc[okt]]
            Kt = np.zeros(len(tc))
            Kt[okt] = self.K_cells_compact(h, 2)[tc[okt]]
            A_top = self.g.dx * self.g.dy
            infil = np.zeros(len(tc))
            Tb = np.zeros(len(tc))
            Tb[okt] = A_top / (0.5 * self.g.dz / np.maximum(Kt[okt], KMIN) + self.top_R[okt])
            infil[okt] = Tb[okt] * (self.g.H - (ht[okt] + self.z[self.top_cells][okt]))
            turn_on = okt & ~ponded & (ht > 0.0)
            qv = np.broadcast_to(np.asarray(bc.q_top, float), (len(tc),))
            turn_off = okt & ponded & (infil > qv * A_top * (1 + 1e-9)) & ~locked_ponded
            if not turn_on.any() and not turn_off.any():
                break
            ponded = ponded.copy()
            ponded[turn_on] = True
            ponded[turn_off] = False
            conv = False
        out = h_full.ravel().copy()
        out[c] = h
        return out.reshape(self.shape), conv, total_it, ponded

    def _face_T_compact(self, hc):
        T = np.empty(self.nF)
        for ax in range(3):
            sel = self.fax == ax
            if not sel.any():
                continue
            Ks = (self.Ks_v if ax == 2 else self.Ks_h)[self.cells]
            Kc = np.maximum(Ks * vg_kr(hc, self.alpha[self.cells], self.n[self.cells]), KMIN)
            Ka = Kc[self.fa[sel]]; Kb = Kc[self.fb[sel]]
            d = self.fd[sel]
            T[sel] = self.fA[sel] / (0.5 * d / Ka + 0.5 * d / Kb + self.fR[sel])
        return T

    def K_cells_compact(self, hc, axis):
        Ks = (self.Ks_v if axis == 2 else self.Ks_h)[self.cells]
        return np.maximum(Ks * vg_kr(hc, self.alpha[self.cells], self.n[self.cells]), KMIN)

    def initial_guess(self, bc: FlowBC, h_fallback=-1.0):
        """정상해 초기 추정: 지하수면 근처는 정수압, 그 위는 K(h)=q 인 단위경사 수두."""
        c = self.cells
        h = np.full(self.g.N, h_fallback)
        qm = float(np.mean(bc.q_top))
        if qm > 0:
            hq = h_for_flux(qm, np.maximum(self.Ks_v[c], 1e-20), self.alpha[c], self.n[c])
        else:
            hq = np.full(len(c), -2.0)
        hs = bc.z_wt - self.z[c]          # 정수압
        h[c] = np.where(hs >= 0, hs, np.maximum(hq, hs))
        return h.reshape(self.shape)

    def solve_steady(self, h0, bc: FlowBC, dt0=None, max_steps=40, progress=None):
        """정상해: 물리 기반 초기 추정에서 바로 뉴턴. 실패하면 의사-과도 연속법(Δt를 키워 감)."""
        if h0 is None:
            h0 = self.initial_guess(bc)
        info = {"steps": 0, "newton": 0, "method": "뉴턴(정상)"}
        hn, conv, it, pond = self.step(h0, np.inf, bc, None, max_iter=40)
        info["newton"] += it
        if conv:
            info["final_conv"] = True
            return self.fluxes(hn, bc, pond, info=info)
        info["method"] = "의사-과도 연속법"
        h, ponded, dt = h0.copy(), None, (dt0 or 3600.0)
        for s_ in range(max_steps):
            hn, conv, it, pond = self.step(h, dt, bc, ponded)
            info["newton"] += it
            if not conv:
                dt *= 0.25
                if dt < 1.0:
                    break
                continue
            h, ponded = hn, pond
            info["steps"] = s_ + 1
            if progress:
                progress(min(1.0, np.log10(max(dt, 1)) / 10.0))
            hs, c2, it2, p2 = self.step(h, np.inf, bc, ponded, max_iter=25)
            info["newton"] += it2
            if c2:
                h, ponded = hs, p2
                info["final_conv"] = True
                return self.fluxes(h, bc, ponded, info=info)
            dt *= 8.0
        info["final_conv"] = False
        return self.fluxes(h, bc, ponded, info=info)

    # ── 플럭스 ───────────────────────────────────────────────────────
    def fluxes(self, h_full, bc: FlowBC, ponded=None, info=None) -> FlowState:
        g = self.g
        nx, ny, nz = self.shape
        c = self.cells
        h = h_full.ravel()[c]
        if ponded is None:
            ponded = np.zeros(len(self.top_cells), dtype=bool)
        thr, ths, al, n = self.thr[c], self.ths[c], self.alpha[c], self.n[c]
        th = vg_theta(h, thr, ths, al, n)
        H = h + self.z[c]
        T = self._face_T_compact(h)
        flux = T * (H[self.fa] - H[self.fb])
        F = [np.zeros(f["shape"]).ravel() for f in self.faces]
        off = 0
        for k, f in enumerate(self.faces):
            m = len(f["a"])
            F[k][f["fid"]] = flux[off:off + m]
            off += m
        Fx, Fy, Fz = (F[0].reshape(self.faces[0]["shape"]), F[1].reshape(self.faces[1]["shape"]),
                      F[2].reshape(self.faces[2]["shape"]))
        A_top = g.dx * g.dy
        Kt = self.K_cells_compact(h, 2)
        Ftop = np.zeros(nx * ny)
        tc = self.cmap[self.top_cells]
        okt = self.top_ok
        fl = okt & ~ponded
        qv = np.broadcast_to(np.asarray(bc.q_top, float), (nx * ny,))
        Ftop[fl] = qv[fl] * A_top
        pd = okt & ponded
        if pd.any():
            ii = tc[pd]
            Tb = A_top / (0.5 * g.dz / Kt[ii] + self.top_R[pd])
            Ftop[pd] = Tb * (g.H - H[ii])
        runoff = float(qv.sum() * A_top - Ftop.sum())
        Fbot = np.zeros(nx * ny)
        bcells = self.cmap[self.bot_cells]
        okb = self.bot_ok
        ii = bcells[okb]
        if bc.bottom == "자유배수":
            Fbot[okb] = Kt[ii] * A_top
        else:
            xb = (np.arange(nx) + 0.5) * g.dx
            Hb = np.repeat(bc.z_wt - bc.gw_grad * (xb - g.Lx / 2.0), ny)[okb]
            Tb = A_top / (0.5 * g.dz / Kt[ii])
            Fbot[okb] = Tb * (H[ii] - Hb)
        Fleft = np.zeros(ny * nz); Fright = np.zeros(ny * nz)
        if bc.gw_grad != 0.0:
            Ax = g.dy * g.dz
            Kx = self.K_cells_compact(h, 0)
            kk = np.tile(np.arange(nz), ny)
            for cells_full, x_face, arr, sgn in ((self.left_cells, 0.0, Fleft, 1), (self.right_cells, g.Lx, Fright, -1)):
                cm = self.cmap[cells_full]
                ok = (cm >= 0) & (kk >= bc.wt_k)
                ii = cm[ok]
                Hl = bc.z_wt - bc.gw_grad * (x_face - g.Lx / 2.0)
                Tb = Ax / (0.5 * g.dx / Kx[ii])
                q = Tb * (Hl - H[ii])          # 셀로 들어오는 양
                arr[ok] = q if sgn == 1 else -q  # right: 나가는 양(+)
        Fsrc = np.zeros(g.N)
        if bc.src is not None:
            Fsrc += bc.src.ravel()
        if bc.fixed_h is not None:
            # 고정 수두 셀의 원천 = 셀 수지 불균형
            fh = bc.fixed_h.ravel()[c]
            fixed = np.isfinite(fh)
            if fixed.any():
                net = np.zeros(self.nA)
                net += np.bincount(self.fb, weights=flux, minlength=self.nA) - np.bincount(self.fa, weights=flux, minlength=self.nA)
                tcm = self.cmap[self.top_cells]
                net[tcm[okt]] += Ftop[okt]
                net[bcells[okb]] -= Fbot[okb]
                Fsrc[c[fixed]] += -net[fixed]
        theta_full = np.zeros(g.N)
        theta_full[c] = th
        hfull = h_full.ravel().copy()
        return FlowState(h=hfull.reshape(self.shape), theta=theta_full.reshape(self.shape),
                         Fx=Fx, Fy=Fy, Fz=Fz, Ftop=Ftop.reshape(nx, ny), Fbot=Fbot.reshape(nx, ny),
                         Fleft=Fleft.reshape(ny, nz), Fright=Fright.reshape(ny, nz),
                         Fsrc=Fsrc.reshape(self.shape), ponded=ponded.reshape(nx, ny),
                         runoff=runoff, info=info or {})

    def injection_perturbation(self, src=None, fixed_dH=None, bottom_dirichlet=True):
        """주입에 의한 압력 섭동 흐름(중첩): ∇·(Ks∇ΔH) + Q = 0, 지표 ΔH=0, 바닥 ΔH=0(수위) 또는 무유동.
        포화 투수계수 사용(주입구 주변 포화 가정). 반환: 섭동 면 유량들과 원천 유량."""
        g = self.g
        c = self.cells
        nA = self.nA
        T = np.empty(self.nF)
        for ax in range(3):
            sel = self.fax == ax
            Ks = (self.Ks_v if ax == 2 else self.Ks_h)[c]
            Ka, Kb = np.maximum(Ks[self.fa[sel]], KMIN), np.maximum(Ks[self.fb[sel]], KMIN)
            d = self.fd[sel]
            T[sel] = self.fA[sel] / (0.5 * d / Ka + 0.5 * d / Kb + self.fR[sel])
        diag = np.bincount(self.fa, weights=T, minlength=nA) + np.bincount(self.fb, weights=T, minlength=nA)
        rhs = np.zeros(nA)
        A_top = g.dx * g.dy
        Ksv = np.maximum(self.Ks_v[c], KMIN)
        tcm = self.cmap[self.top_cells]
        okt = self.top_ok
        Tt = np.zeros(len(tcm))
        Tt[okt] = A_top / (0.5 * g.dz / Ksv[tcm[okt]] + self.top_R[okt])
        diag[tcm[okt]] += Tt[okt]
        bcm = self.cmap[self.bot_cells]
        okb = self.bot_ok
        Tb = np.zeros(len(bcm))
        if bottom_dirichlet:
            Tb[okb] = 2 * A_top * Ksv[bcm[okb]] / g.dz
            diag[bcm[okb]] += Tb[okb]
        if src is not None:
            rhs += src.ravel()[c]
        free = np.ones(nA, bool)
        x = np.zeros(nA)
        if fixed_dH is not None:
            fd = fixed_dH.ravel()[c]
            fx = np.isfinite(fd)
            x[fx] = fd[fx]
            free = ~fx
        A = sp.csr_matrix((np.concatenate([-T, -T, diag]),
                           (np.concatenate([self.fa, self.fb, np.arange(nA)]),
                            np.concatenate([self.fb, self.fa, np.arange(nA)]))), shape=(nA, nA))
        if not free.all():
            rhs = rhs - A[:, ~free] @ x[~free]
        Af = A[free][:, free]
        x[free] = self._linsolve(Af, rhs[free], spd=True)
        flux = T * (x[self.fa] - x[self.fb])
        F = []
        off = 0
        for f in self.faces:
            arr = np.zeros(f["shape"]).ravel()
            m = len(f["a"])
            arr[f["fid"]] = flux[off:off + m]
            off += m
            F.append(arr.reshape(f["shape"]))
        nx, ny, nz = self.shape
        Ftop = np.zeros(nx * ny)
        Ftop[okt] = -Tt[okt] * x[tcm[okt]]              # 지표로 나가면 음수(들어오는 쪽 +)
        Fbot = np.zeros(nx * ny)
        Fbot[okb] = Tb[okb] * x[bcm[okb]]
        net = np.bincount(self.fb, weights=flux, minlength=nA) - np.bincount(self.fa, weights=flux, minlength=nA)
        net[tcm[okt]] += Ftop[okt]
        net[bcm[okb]] -= Fbot[okb]
        Fsrc = np.zeros(g.N)
        Fsrc[c] = -net                                   # 셀 수지 = 원천
        dH = np.zeros(g.N); dH[c] = x
        return {"Fx": F[0], "Fy": F[1], "Fz": F[2], "Ftop": Ftop.reshape(nx, ny),
                "Fbot": Fbot.reshape(nx, ny), "Fsrc": Fsrc.reshape(self.shape), "dH": dH.reshape(self.shape)}

    def mass_residual(self, st: FlowState) -> float:
        """정상 상태 수지: (유입-유출)/유입."""
        qin = st.Ftop.sum() + st.Fleft.sum() + max(0.0, st.Fsrc.sum()) - min(0.0, st.Fbot.sum())
        qout = max(0.0, st.Fbot.sum()) + st.Fright.sum()
        return float((qin - qout) / max(qin, 1e-30)) if qin > 0 else 0.0

    # ── 정수두 시험(유효 K) ──────────────────────────────────────────
    def constant_head_Keff(self, k_bottom: int):
        """토양층(k < k_bottom) 포화 정수두 시험: 상단 h=0, 경계면 h=0 → 단위 동수경사.
        K_eff = Q / A (m/s). 지표 문서의 '간이 정수두법'을 계산으로 재현."""
        g = self.g
        nx, ny, nz = self.shape
        kk = np.tile(np.arange(nz), nx * ny)
        sub = (kk[self.cells] < k_bottom)
        if not sub.any():
            return 0.0
        smap = -np.ones(self.nA, dtype=np.int64)
        smap[sub] = np.arange(sub.sum())
        okf = sub[self.fa] & sub[self.fb]
        fa, fb = smap[self.fa[okf]], smap[self.fb[okf]]
        T = np.empty(okf.sum())
        for ax in range(3):
            sel = self.fax[okf] == ax
            Ks = (self.Ks_v if ax == 2 else self.Ks_h)[self.cells][sub]
            Ka, Kb = Ks[fa[sel]], Ks[fb[sel]]
            d = self.fd[okf][sel]
            T[sel] = self.fA[okf][sel] / (0.5 * d / np.maximum(Ka, KMIN) + 0.5 * d / np.maximum(Kb, KMIN) + self.fR[okf][sel])
        nS = int(sub.sum())
        diag = np.zeros(nS); R = np.zeros(nS)
        diag += np.bincount(fa, weights=T, minlength=nS) + np.bincount(fb, weights=T, minlength=nS)
        z = self.z[self.cells][sub]
        A_top = g.dx * g.dy
        Ksv = self.Ks_v[self.cells][sub]
        # 상단 h=0 (H = g.H)
        tcm = self.cmap[self.top_cells]
        okt = self.top_ok & (tcm >= 0)
        it = smap[tcm[okt]]
        okt2 = it >= 0
        it = it[okt2]
        Rt = self.top_R[okt][okt2]
        Tt = A_top / (0.5 * g.dz / np.maximum(Ksv[it], KMIN) + Rt)
        diag[it] += Tt; R[it] += Tt * g.H
        # 경계면 h=0 (H = 경계면 표고)
        zb = g.H - k_bottom * g.dz
        kc = kk[self.cells][sub]
        ib = np.where(kc == k_bottom - 1)[0]
        Tb = A_top / (0.5 * g.dz / np.maximum(Ksv[ib], KMIN))
        diag[ib] += Tb; R[ib] += Tb * zb
        A = sp.csr_matrix((np.concatenate([-T, -T, diag]),
                           (np.concatenate([fa, fb, np.arange(nS)]), np.concatenate([fb, fa, np.arange(nS)]))),
                          shape=(nS, nS))
        d = A.diagonal(); d = np.where(d > 0, d, 1.0)
        M = spl.LinearOperator(A.shape, matvec=lambda v: v / d)
        H, info = spl.cg(A, R, rtol=1e-12, M=M, maxiter=6000)
        if info != 0:
            H = spl.spsolve(A.tocsc(), R)
        Q = float((Tb * (H[ib] - zb)).sum())
        L = k_bottom * g.dz
        dH = g.H - zb
        return Q * L / (g.Lx * g.Ly * dH)
