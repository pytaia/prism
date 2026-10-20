from __future__ import annotations

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import signals as S
from prism import Prism

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
SURF, GRID = "#fcfcfb", "#e6e5e0"
C_CONTENT, C_TRACT = "#2a78d6", "#eb6834"
C_ALERT = "#d03b3b"
SEQ = LinearSegmentedColormap.from_list(
    "seq", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 10, "axes.titleweight": "bold", "figure.facecolor": SURF,
    "axes.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False,
})


def cases():
    x = S.make("audio", seed=1)
    xr = S.make("radio", seed=1)
    return [
        ("01_clean_audio", "Чистый речеподобный сигнал", (x, [])),
        ("02_natural_change", "Естественная смена содержания (без вмешательства)", S.natural_change(1)),
        ("03_splice", "Склейка: вставка 8–11 с с другого устройства", S.tamper_splice(x, "audio", 1)),
        ("04_lowpass", "Фильтр НЧ 4 кГц на 5–8 с", S.tamper_lowpass(x)),
        ("05_gain", "Усиление ×1,5 после оцифровки на 12–15 с", S.tamper_gain(x)),
        ("06_clip", "Перегруз (клиппинг) на 4–6 с", S.tamper_clip(x)),
        ("07_copy", "Копирование: 3–5 с вставлено в 15–17 с", S.tamper_copy(x)),
        ("08_resample", "Ресемплинг (+3 %) на 9–12 с", S.tamper_resample(x)),
        ("09_radio_splice", "Радиосигнал: склейка 8–11 с", S.tamper_splice(xr, "radio", 1)),
    ]


def draw(name, title, y, truth, r, outdir):
    fig = plt.figure(figsize=(12, 7.2))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 2.1], hspace=0.55, wspace=0.28)
    t = np.arange(len(y)) / S.FS
    dur = t[-1]
    verdict = "ИЗМЕНЁН" if r.tampered else "изменений не найдено"
    fig.suptitle(f"{title}  →  вердикт: {verdict}", x=0.06, ha="left",
                 fontsize=12, fontweight="bold", color=C_ALERT if r.tampered else INK)

    def shade(ax):
        for a, b in truth:
            ax.axvspan(a, b, color="#d9d8d3", alpha=0.6, lw=0, zorder=0)

    # 1. сигнал + найденные события
    ax = fig.add_subplot(gs[0, :])
    shade(ax)
    step = max(1, len(y) // 20000)
    ax.plot(t[::step], y[::step], color=MUTED, lw=0.5)
    lim = np.max(np.abs(y)) * 1.15
    for k, a, b, d in r.events:
        if b - a < 0.05:
            ax.axvline(a, color=C_ALERT, lw=1.5)
        else:
            ax.plot([a, b], [lim * 0.95] * 2, color=C_ALERT, lw=4, solid_capstyle="butt")
    for tn in r.natural:
        ax.axvline(tn, color=C_CONTENT, lw=1.2, ls="--")
    ax.set_xlim(0, dur)
    ax.set_ylim(-lim, lim)
    ax.set_title("Сигнал  (серая полоса — истинное изменение, красное — найдено, "
                 "синий пунктир — естественная смена)", loc="left", fontsize=9)
    ax.set_yticks([])

    # 2. отпечаток тракта
    ax = fig.add_subplot(gs[1, :])
    shade(ax)
    ax.plot(r.fp_time, r.fp_score, color=C_TRACT, lw=1.2)
    ax.axhline(7.5, color=INK2, lw=0.8, ls=":")
    ax.text(dur, 7.5, " порог", va="center", fontsize=8, color=INK2)
    ax.set_xlim(0, dur)
    ax.set_ylim(0, max(12, min(np.max(r.fp_score) * 1.1, 60)))
    ax.set_title("Механизм 1 — отпечаток тракта: max |z| признаков остатка по окнам",
                 loc="left", fontsize=9)
    ax.grid(axis="y", color=GRID, lw=0.6)

    # 3–4. матрицы
    ext = [0, dur, dur, 0]
    for j, (D, lab) in enumerate([(r.D_content, "Матрица содержания (по сигналу)"),
                                  (r.D_tract, "Матрица тракта (по остатку)")]):
        ax = fig.add_subplot(gs[2, j])
        v = np.percentile(D, 98)
        ax.imshow(D, cmap=SEQ, vmin=0, vmax=v, extent=ext, interpolation="nearest")
        ax.set_title(lab, loc="left", fontsize=9)
        ax.set_xlabel("сегмент j, с")
        ax.set_ylabel("модель сегмента i, с")
        ax.spines[:].set_visible(False)

    # 5. новизна
    ax = fig.add_subplot(gs[2, 2])
    shade(ax)
    ax.plot(r.seg_time, r.nov_content, color=C_CONTENT, lw=1.2, label="содержание")
    ax.plot(r.seg_time, r.nov_tract, color=C_TRACT, lw=1.6, label="тракт")
    ax.axhline(12, color=INK2, lw=0.8, ls=":")
    ax.set_xlim(0, dur)
    top = max(20, min(np.nanmax(r.nov_tract) * 1.1, 200))
    ax.set_ylim(min(-5, np.nanmin(r.nov_tract)), top)
    ax.set_title("Детектор «шахматной доски» (z)", loc="left", fontsize=9)
    ax.set_xlabel("время, с")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    ax.grid(axis="y", color=GRID, lw=0.6)

    fig.savefig(os.path.join(outdir, name + ".png"), dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    outdir = "figures"
    os.makedirs(outdir, exist_ok=True)
    P = Prism(fs=S.FS)
    for name, title, (y, truth) in cases():
        r = P.analyze(y)
        print(f"\n=== {title}  (истина: {truth or 'нет изменений'})")
        print(r.summary())
        draw(name, title, y, truth, r, outdir)


if __name__ == "__main__":
    main()
