from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfilt, lfilter, resample_poly

FS = 16000


@dataclass
class Device:
    noise_db: float = -62.0       # шумовой пол, дБ от полной шкалы
    noise_tilt: float = 0.3       # «цвет» шума: 0 — белый, >0 — розоватый
    aa_cut: float = 7200.0        # частота среза антиалиасингового фильтра
    hum_db: float | None = -58.0  # фон сети 50 Гц (None — нет)
    hum_phase: float = 0.0


DEV_A = Device()
DEV_B = Device(noise_db=-56.0, noise_tilt=0.0, aa_cut=6300.0, hum_db=None)


def content_audio(dur, rng, style=0):
    """Речеподобный сигнал: дрейфующие резонансы, паузы, огибающая."""
    n = int(dur * FS)
    out = np.zeros(n)
    blk = 256
    f1, f2 = (500, 1500) if style == 0 else (300, 2600)
    bw = 120 if style == 0 else 60
    state = np.zeros(4)
    env = 0.0
    talking = True
    for b in range(0, n, blk):
        if rng.random() < 0.012:
            talking = not talking
        target = (0.25 + 0.15 * rng.random()) if talking else 0.0
        env = 0.9 * env + 0.1 * target
        f1 = np.clip(f1 + rng.normal(0, 15), 250, 900)
        f2 = np.clip(f2 + rng.normal(0, 30), 1000, 3200)
        poles = []
        for f in (f1, f2):
            r = np.exp(-np.pi * bw / FS)
            poles += [r * np.exp(2j * np.pi * f / FS), r * np.exp(-2j * np.pi * f / FS)]
        a = np.real(np.poly(poles))
        if style == 0:
            exc = rng.normal(0, 1, blk)
        else:   # «звонкий» стиль: импульсы основного тона
            exc = np.zeros(blk)
            exc[:: rng.integers(80, 140)] = 6.0
            exc += rng.normal(0, 0.3, blk)
        y, state = lfilter([1.0], a, exc, zi=state)
        out[b:b + blk] = y[: min(blk, n - b)] * env
    return out / (np.std(out) + 1e-12) * 0.08


def content_radio(dur, rng):
    n = int(dur * FS)
    sps = 16
    nsym = n // sps + 8
    sym = rng.choice([-1.0, 1.0], nsym)
    up = np.zeros(nsym * sps)
    up[::sps] = sym
    t = np.arange(-4 * sps, 4 * sps + 1) / sps
    rc = np.sinc(t) * np.cos(np.pi * 0.35 * t) / (1 - (0.7 * t) ** 2 + 1e-9)
    bb = np.convolve(up, rc, mode="same")[:n]
    tt = np.arange(n) / FS
    return 0.12 * bb * np.cos(2 * np.pi * 3000 * tt + 0.0002 * np.cumsum(rng.normal(0, 1, n)))


def content_sensor(dur, rng):
    n = int(dur * FS)
    tt = np.arange(n) / FS
    f = 120 + np.cumsum(rng.normal(0, 0.002, n))
    ph = 2 * np.pi * np.cumsum(f) / FS
    y = 0.05 * np.sin(ph) + 0.02 * np.sin(2 * ph + 1) + 0.01 * np.sin(3 * ph + 2)
    y += 0.03 * np.cumsum(rng.normal(0, 1, n)) / np.sqrt(n)
    hits = rng.random(n) < 2 / FS
    y += lfilter([1.0], [1, -1.6, 0.9], hits * rng.normal(0, 0.3, n))
    y += 0.01 * rng.normal(0, 1, n) * (1 + 0.5 * np.sin(2 * np.pi * 0.2 * tt))
    return y


CONTENT = {"audio": content_audio, "radio": content_radio, "sensor": content_sensor}


# --------------------------- тракт устройства ---------------------------

def record(content, rng, dev: Device = DEV_A, quantize=True):
    n = len(content)
    w = rng.normal(0, 1, n)
    if dev.noise_tilt > 0:
        w = lfilter([1.0], [1, -dev.noise_tilt], w)
        w /= np.std(w)
    x = content + 10 ** (dev.noise_db / 20) * w
    if dev.hum_db is not None:
        t = np.arange(n) / FS
        drift = np.cumsum(rng.normal(0, 2e-5, n))
        x += 10 ** (dev.hum_db / 20) * np.sin(2 * np.pi * 50 * t + dev.hum_phase + drift)
    sos = butter(10, dev.aa_cut, fs=FS, output="sos")
    x = sosfilt(sos, x)
    if quantize:
        x = quantize16(x)
    return x


def quantize16(x):
    return np.clip(np.round(x * 32768), -32768, 32767) / 32768


def make(kind="audio", dur=20.0, seed=0, dev=DEV_A, style=0):
    rng = np.random.default_rng(seed)
    c = CONTENT[kind](dur, rng) if kind != "audio" else content_audio(dur, rng, style)
    return record(c, rng, dev)


# --------------------------- вмешательства ---------------------------

def _sl(t0, t1):
    return slice(int(t0 * FS), int(t1 * FS))


def tamper_splice(x, kind, seed, t0=8.0, t1=11.0):
    y = x.copy()
    other = make(kind, dur=t1 - t0 + 0.01, seed=seed + 1000, dev=DEV_B)
    s = _sl(t0, t1)
    y[s] = other[: s.stop - s.start]
    return y, [(t0, t1)]


def tamper_lowpass(x, t0=5.0, t1=8.0, cut=4000):
    y = x.copy()
    s = _sl(t0, t1)
    sos = butter(8, cut, fs=FS, output="sos")
    y[s] = quantize16(sosfilt(sos, x)[s])
    return y, [(t0, t1)]


def tamper_gain(x, t0=12.0, t1=15.0, g=1.5):
    y = x.copy()
    s = _sl(t0, t1)
    y[s] = quantize16(x[s] * g)
    return y, [(t0, t1)]


def tamper_clip(x, t0=4.0, t1=6.0, g=4.0):
    y = x.copy()
    s = _sl(t0, t1)
    lim = 0.5 * np.max(np.abs(x[s]))   # «перегруз»: усилили в g раз, упёрлись в предел
    y[s] = quantize16(np.clip(x[s] * g, -lim * g / 3, lim * g / 3) / g)
    return y, [(t0, t1)]


def tamper_copy(x, src=3.0, dst=15.0, length=2.0):
    y = x.copy()
    y[_sl(dst, dst + length)] = x[_sl(src, src + length)]
    return y, [(dst, dst + length)]


def tamper_resample(x, t0=9.0, t1=12.0, up=100, down=103):
    s = _sl(t0, t1)
    pad = 400
    seg = x[s.start - pad:s.stop + pad]
    r = resample_poly(seg, up, down)
    need = s.stop - s.start
    r = r[int(pad * up / down): int(pad * up / down) + need]
    y = x.copy()
    y[s.start:s.start + len(r)] = quantize16(r)
    return y, [(t0, t1)]


def natural_change(seed, dur=20.0, t_change=10.0):
    """Без вмешательства: то же устройство, но содержание резко меняет характер."""
    rng = np.random.default_rng(seed)
    c1 = content_audio(t_change, rng, style=0)
    c2 = content_audio(dur - t_change, rng, style=1)
    return record(np.concatenate([c1, c2]), rng, DEV_A), []
