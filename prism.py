from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import lfilter, welch, fftconvolve
from scipy.ndimage import median_filter


# ----------------------------------------------------------------------------
# Базовые кирпичи: автокорреляция, Левинсон–Дарбин, адаптивный остаток
# ----------------------------------------------------------------------------

def autocorr(x: np.ndarray, p: int) -> np.ndarray:
    n = len(x)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    X = np.fft.rfft(x, nfft)
    r = np.fft.irfft(X * np.conj(X), nfft)[: p + 1]
    return r / n


def levinson(r: np.ndarray) -> tuple[np.ndarray, float]:
    p = len(r) - 1
    r = r.copy()
    r[0] *= 1.0 + 1e-9
    r[0] += 1e-20
    a = np.zeros(p + 1)
    a[0] = 1.0
    err = r[0]
    for k in range(1, p + 1):
        acc = r[k] + np.dot(a[1:k], r[k - 1:0:-1])
        refl = -acc / err
        a[1:k] = a[1:k] + refl * a[k - 1:0:-1]
        a[k] = refl
        err *= 1.0 - refl * refl
    return a, err


def adaptive_residual(x: np.ndarray, p: int = 16, block: int = 512,
                      ctx: int = 2048) -> np.ndarray:
    n = len(x)
    e = np.zeros(n)
    for b in range(0, n, block):
        c0 = max(0, min(n - ctx, b + block // 2 - ctx // 2))
        seg = x[c0:c0 + ctx]
        # устойчивая подстройка: сильный удар в контексте не «перетягивает» модель
        a, _ = levinson(robust_autocorr(seg[None, :], p, 8)[0])
        s0 = max(0, b - p)
        y = lfilter(a, [1.0], x[s0:b + block])
        e[b:b + block] = y[b - s0:]
    return e


def clip_outliers(e: np.ndarray, k: float = 4.0, block: int = 2048) -> np.ndarray:
    out = e.copy()
    for b in range(0, len(e), block):
        seg = e[b:b + block]
        sig = np.median(np.abs(seg)) * 1.4826 + 1e-30
        out[b:b + block] = np.clip(seg, -k * sig, k * sig)
    return out


def median_psd(S: np.ndarray, parts: int = 8) -> np.ndarray:
    N, n = S.shape
    sub = n // parts
    Sb = S[:, : sub * parts].reshape(N, parts, sub) * np.hanning(sub)
    P = np.abs(np.fft.rfft(Sb, 2 * sub, axis=2)) ** 2
    return np.median(P, axis=1) / sub + 1e-30  # (N, sub+1)


def robust_autocorr(S: np.ndarray, p: int, parts: int = 8) -> np.ndarray:
    P = median_psd(S, parts)
    return np.fft.irfft(P, axis=1)[:, : p + 1]


def frame(x: np.ndarray, win: int, hop: int) -> np.ndarray:
    nfr = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(nfr)[:, None]
    return x[idx]


def robust_z(v: np.ndarray, floor: float = 0.0) -> np.ndarray:
    med = np.nanmedian(v)
    mad = np.nanmedian(np.abs(v - med)) * 1.4826
    return (v - med) / max(mad, floor, 1e-12)


# ----------------------------------------------------------------------------
# Механизм 1: отпечаток тракта
# # ----------------------------------------------------------------------------

FEATURE_NAMES = {
    "flatness": "«белизна» спектра остатка",
    "whiteness": "связи соседних отсчётов",
    "flat_top": "срезанные пики (плато на максимуме)",
    "hf_share": "доля ВЧ в остатке",
    "floor": "уровень шумового пола",
    "comb": "гребёнка уровней квантования (громкость меняли после оцифровки)",
    "hum_amp": "амплитуда фона сети",
    "hum_phase": "скачок фазы фона сети",
}


def _detect_quantization(x: np.ndarray) -> float | None:
    for bits in (16, 24, 8):
        q = 2.0 ** (bits - 1)
        v = x[: 200000] * q
        if np.max(np.abs(v - np.round(v))) < 1e-6 and np.std(np.round(v)) > 3:
            return q
    return None


def _detect_hum(x: np.ndarray, fs: float) -> float | None:
    nper = int(fs * 2)
    f, P = welch(x, fs, nperseg=min(nper, len(x)))
    best = None
    for f0 in (50.0, 60.0):
        band = (f > f0 - 2) & (f < f0 + 2)
        neigh = (f > f0 - 15) & (f < f0 + 15) & ~band
        if band.any() and neigh.any():
            ratio = P[band].max() / (np.median(P[neigh]) + 1e-30)
            if ratio > 30 and (best is None or ratio > best[1]):
                best = (f[band][np.argmax(P[band])], ratio)
    return None if best is None else best[0]


def fingerprint_features(x: np.ndarray, e: np.ndarray, fs: float,
                         win: int, hop: int) -> dict[str, np.ndarray]:
    X = frame(x, win, hop)
    E_raw = frame(e, win, hop)
    nfr = len(X)
    feats: dict[str, np.ndarray] = {}

    vx = X.var(axis=1) + 1e-30
    ve = E_raw.var(axis=1) + 1e-30
    # «Выигрыш предсказания» log10(vx/ve) сильно зависит от содержания (паузы/речь)
    # и в тестах давал ложные тревоги, поэтому в решение не входит.
    # «цвет» остатка — по устойчивому (медианному) спектру окна
    P = median_psd(E_raw, 8)[:, 1:]
    feats["flatness"] = np.exp(np.log(P).mean(axis=1)) / P.mean(axis=1)
    k_hf = int(P.shape[1] * 0.80)
    feats["hf_share"] = np.log10(P[:, k_hf:].sum(axis=1) / P.sum(axis=1))

    # белизна: сумма квадратов нормированных автокорреляций остатка, лаги 1..16
    r = robust_autocorr(E_raw, 16, 8)
    rho = r[:, 1:] / r[:, :1]
    feats["whiteness"] = np.log10((rho ** 2).sum(axis=1) * win)

    # срезанные пики: сколько отсчётов «упёрлось» в максимум окна
    ax = np.abs(X)
    mx = ax.max(axis=1, keepdims=True)
    feats["flat_top"] = np.log10((ax >= mx * 0.999).sum(axis=1).astype(float))

    # шумовой пол: 10-й перцентиль энергии коротких кадров остатка.
    # Меряем только в тихих окнах (паузах): там остаток — почти чистый шум тракта.
    sh = 128
    Esh = (E_raw[:, : (win // sh) * sh].reshape(nfr, -1, sh) ** 2).mean(axis=2)
    fl = np.log10(np.percentile(Esh, 10, axis=1) + 1e-30)
    lv = np.log10(vx)
    quiet = lv <= np.percentile(lv, 5) + 0.5  # не громче самых тихих мест на 5 дБ
    feats["floor"] = np.where(quiet, fl, np.nan)

    qstep = _detect_quantization(x)
    if qstep is not None:
        comb = np.zeros(nfr)
        for i in range(nfr):
            v = np.round(X[i] * qstep).astype(np.int64)
            v -= v.min()
            nfft = 1 << int(np.ceil(np.log2(2 * (v.max() + 1))))
            H = np.fft.rfft(np.bincount(v, minlength=nfft)[:nfft])
            f = np.arange(len(H)) / nfft
            band = (f > 0.02) & (f <= 0.5)
            comb[i] = (np.abs(H[band]) ** 2).max() / len(v)
        feats["comb"] = np.log10(comb)

    zfeats = {}
    f0 = _detect_hum(x, fs)
    if f0 is not None:
        Lh = 8192
        w = np.hanning(Lh)
        w /= w.sum()
        t = np.arange(len(x)) / fs
        centers = np.arange(nfr) * hop + win // 2

        def demod(f):
            z = x * np.exp(-2j * np.pi * f * t)
            c = fftconvolve(z, w, mode="same")
            return c[np.clip(centers, 0, len(x) - 1)]

        Z0 = demod(f0)
        side = np.array([demod(f0 + d) for d in (-14, -11, -8, 8, 11, 14)])
        sn = np.sqrt((np.abs(side) ** 2).mean(axis=0)) + 1e-30
        A = np.abs(Z0)
        Am = np.median(A)
        zfeats["hum_amp"] = (A - Am) / np.sqrt(sn ** 2 + (0.1 * Am) ** 2)
        ph = np.unwrap(np.angle(Z0))
        d = np.diff(ph, prepend=ph[0])
        d = d - median_filter(d, 15, mode="nearest")
        zfeats["hum_phase"] = d / (sn / (A + 1e-30) + 0.03)
    return feats, zfeats


# минимальный разброс признаков (в их собственных единицах)
FEATURE_FLOOR = {"flatness": 0.02, "whiteness": 0.1,
                 "flat_top": 0.15, "hf_share": 0.05, "floor": 0.1,
                 "comb": 0.1}


# ----------------------------------------------------------------------------
# Механизм 2: матрица взаимного предсказания
# ----------------------------------------------------------------------------

def segment_models(x: np.ndarray, n: int, hop: int, p: int, robust=False):
    if robust:
        R = robust_autocorr(frame(x, n, hop), p, 8)  # (N, p+1)
    else:
        S = frame(x, n, hop) * np.hanning(n)
        R = np.array([autocorr(s, p) for s in S])  # (N, p+1)
    A = np.array([levinson(r)[0] for r in R])  # (N, p+1)
    return R, A


def mutual_matrix(x: np.ndarray, n: int, hop: int, p: int,
                  robust: bool = False) -> np.ndarray:
    R, A = segment_models(x, n, hop, p, robust)
    N = len(A)
    C = np.zeros((N, p + 1))
    for m in range(p + 1):
        C[:, m] = (A[:, : p + 1 - m] * A[:, m:]).sum(axis=1) * (1 if m == 0 else 2)
    E = C @ R.T  # E[i,j]
    own = np.diag(E).copy()
    Rat = np.maximum(E / own[None, :], 1.0)
    L = np.log(Rat)
    return 0.5 * (L + L.T)


def foote_novelty(D: np.ndarray, L: int) -> np.ndarray:
    N = len(D)
    g = np.exp(-0.5 * ((np.arange(L) + 0.5) / (L / 2)) ** 2)
    W = np.outer(g, g)
    nov = np.zeros(N)
    for t in range(N):
        l = min(L, t, N - t)
        if l < L // 2:  # у краёв слишком мало сегментов для надёжной оценки
            continue
        w = W[:l, :l]
        before = slice(t - l, t)
        after = slice(t, t + l)
        cross = (D[before, after] * w[::-1, :]).sum() / w.sum()
        within = 0.5 * ((D[before, before] * w[::-1, ::-1]).sum()
                        + (D[after, after] * w).sum()) / w.sum()
        nov[t] = cross - within
    return nov


def find_copies(e: np.ndarray, n: int, hop: int, thr: float = 0.5):
    sig = np.median(np.abs(e)) * 1.4826 + 1e-30
    e = np.clip(e, -3 * sig, 3 * sig)  # одиночные выбросы не должны «совпадать»
    N_tot = len(e)
    nfft = 1 << int(np.ceil(np.log2(N_tot + n)))
    Ef = np.fft.rfft(e, nfft)
    csum = np.concatenate([[0.0], np.cumsum(e * e)])
    loc_en = np.sqrt(np.maximum(csum[n:] - csum[:-n], 1e-30))
    hits = []
    for s in range(0, N_tot - n, hop):
        seg = e[s:s + n]
        en = np.sqrt((seg * seg).sum()) + 1e-30
        c = np.fft.irfft(Ef * np.conj(np.fft.rfft(seg, nfft)), nfft)[: N_tot - n + 1]
        c = c / (en * loc_en)
        lo, hi = max(0, s - n), min(len(c), s + n)
        c[lo:hi] = 0
        j = int(np.argmax(c))
        if c[j] > thr:
            hits.append((s, j, float(c[j])))
    # копия должна держаться минимум на двух соседних сегментах с тем же сдвигом
    keep = []
    for h in hits:
        if any(abs(g[0] - h[0]) == hop and abs((g[1] - g[0]) - (h[1] - h[0])) <= 2
               for g in hits):
            keep.append(h)
    return keep


# ----------------------------------------------------------------------------
# Сборка
# ----------------------------------------------------------------------------

@dataclass
class PrismResult:
    tampered: bool
    events: list = field(default_factory=list)  # (тип, t0, t1, описание)
    natural: list = field(default_factory=list)  # естественные смены содержания
    # сырые данные для графиков
    fp_time: np.ndarray | None = None
    fp_score: np.ndarray | None = None
    fp_feature: list | None = None
    seg_time: np.ndarray | None = None
    D_content: np.ndarray | None = None
    D_tract: np.ndarray | None = None
    nov_content: np.ndarray | None = None
    nov_tract: np.ndarray | None = None
    row_tract: np.ndarray | None = None
    fp_Z: np.ndarray | None = None
    fp_names: list | None = None

    def summary(self) -> str:
        if not self.tampered:
            s = "Изменений не найдено."
        else:
            s = "СИГНАЛ ИЗМЕНЁН:\n" + "\n".join(
                f"  • {k}: {a:.2f}–{b:.2f} с — {d}" for k, a, b, d in self.events)
        if self.natural:
            s += "\nЕстественные смены содержания: " + ", ".join(
                f"{t:.2f} с" for t in self.natural)
        return s


@dataclass
class Prism:
    fs: float
    # механизм 1
    fp_win: int = 2048
    res_order: int = 16
    fp_thr: float = 7.5  # одиночное окно
    fp_thr_run: float = 4.5  # серия из ≥3 окон подряд
    # механизм 2
    seg_len: int = 4096
    content_order: int = 12
    tract_order: int = 24
    foote_L: int = 8
    nov_thr: float = 12.0
    row_thr: float = 15.0
    copy_thr: float = 0.5

    def analyze(self, x: np.ndarray) -> PrismResult:
        x = np.asarray(x, float)
        fs = self.fs
        e = adaptive_residual(x, self.res_order)
        res = PrismResult(tampered=False)

        # ---------- механизм 1: отпечаток тракта ----------
        hop = self.fp_win // 2
        feats, zfeats = fingerprint_features(x, e, fs, self.fp_win, hop)
        names = list(feats)
        names = [k for k in names if np.isfinite(feats[k]).sum() >= 10]
        Z = [robust_z(feats[k], FEATURE_FLOOR[k]) for k in names]
        Z += [zfeats[k] for k in zfeats]
        names += list(zfeats)
        Z = np.abs(np.array(Z))
        Z = np.nan_to_num(Z, nan=0.0)  #
        Z[:, :1] = 0;
        Z[:, -1:] = 0
        score = Z.max(axis=0)
        which = [names[i] for i in Z.argmax(axis=0)]
        t_fp = (np.arange(len(score)) * hop + self.fp_win / 2) / fs
        res.fp_time, res.fp_score, res.fp_feature = t_fp, score, which
        res.fp_Z, res.fp_names = Z, names
        flag = score > self.fp_thr
        run = score > self.fp_thr_run
        k = 0
        while k < len(run):  # серии ≥3 окон выше мягкого порога
            if run[k]:
                m = k
                while m < len(run) and run[m]:
                    m += 1
                if m - k >= 3:
                    flag[k:m] = True
                k = m
            else:
                k += 1
        for a, b in _runs(flag):
            seg_feats = [which[i] for i in range(a, b) if score[i] > self.fp_thr_run]
            top = max(set(seg_feats), key=seg_feats.count) if seg_feats else which[a]
            t0 = a * hop / fs
            t1 = ((b - 1) * hop + self.fp_win) / fs
            res.events.append(("отпечаток", t0, t1, FEATURE_NAMES[top]))

        # ---------- механизм 2: матрицы взаимного предсказания ----------
        n, shop = self.seg_len, self.seg_len // 2
        Dc = mutual_matrix(x, n, shop, self.content_order)
        Dt = mutual_matrix(e, n, shop, self.tract_order, robust=True)
        t_seg = (np.arange(len(Dc)) * shop) / fs  # начало сегмента t = граница
        nc = robust_z(foote_novelty(Dc, self.foote_L))
        nt = robust_z(foote_novelty(Dt, self.foote_L))
        # «крест»: сегмент, непохожий ни на кого по тракту
        row = robust_z(np.median(Dt, axis=1))
        res.seg_time, res.D_content, res.D_tract = t_seg, Dc, Dt
        res.nov_content, res.nov_tract, res.row_tract = nc, nt, row

        for i in _peaks(nt, self.nov_thr):
            res.events.append(("граница тракта", t_seg[i], t_seg[i],
                               "резкая смена тракта (матрица остатков)"))
        for a, b in _runs(row > self.row_thr):
            res.events.append(("чужой участок", t_seg[a], t_seg[b - 1] + n / fs,
                               "непохож на остальные по тракту"))
        tract_b = [t_seg[i] for i in _peaks(nt, self.nov_thr * 0.6)]
        for i in _peaks(nc, self.nov_thr):
            if all(abs(t_seg[i] - tb) > 2 * shop / fs for tb in tract_b):
                res.natural.append(t_seg[i])

        # копии: подозрительно идеальное взаимное предсказание
        copies = find_copies(e, n, shop, self.copy_thr)
        for (s, j, c) in _merge_copies(copies, n):
            res.events.append(("копия", j / fs, (j + (s[1] - s[0])) / fs,
                               f"совпадает с {s[0] / fs:.2f}–{s[1] / fs:.2f} с "
                               f"(корр. {c:.2f})"))

        res.events.sort(key=lambda ev: ev[1])
        res.tampered = len(res.events) > 0
        return res


def _runs(mask: np.ndarray):
    out, k = [], 0
    while k < len(mask):
        if mask[k]:
            m = k
            while m < len(mask) and mask[m]:
                m += 1
            out.append((k, m))
            k = m
        else:
            k += 1
    return out


def _peaks(v: np.ndarray, thr: float):
    idx = [i for i in range(1, len(v) - 1)
           if v[i] > thr and v[i] >= v[i - 1] and v[i] >= v[i + 1]]
    # подавление соседних пиков
    out = []
    for i in sorted(idx, key=lambda i: -v[i]):
        if all(abs(i - j) > 3 for j in out):
            out.append(i)
    return sorted(out)


def _merge_copies(hits, n):
    if not hits:
        return []
    hits = sorted(hits)
    groups, cur = [], [hits[0]]
    for h in hits[1:]:
        prev = cur[-1]
        if h[0] - prev[0] <= n and abs((h[1] - h[0]) - (prev[1] - prev[0])) < 64:
            cur.append(h)
        else:
            groups.append(cur)
            cur = [h]
    groups.append(cur)
    out = []
    for g in groups:
        s0, s1 = g[0][0], g[-1][0] + n
        j0 = g[0][1]
        # «источник» — более ранний из двух
        src, dst = (s0, j0) if s0 < j0 else (j0, s0)
        c = max(h[2] for h in g)
        dup = [k for k, o in enumerate(out) if abs(o[0][0] - src) < n and abs(o[1] - dst) < n]
        if dup:  # та же пара, найденная «с другой стороны»
            k = dup[0]
            (a0, a1), d0, c0 = out[k]
            out[k] = ((min(a0, src), max(a1, src + s1 - s0)), min(d0, dst), max(c0, c))
        else:
            out.append(((src, src + (s1 - s0)), dst, c))
    return out
