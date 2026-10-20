from __future__ import annotations

import sys
import json
from multiprocessing import Pool

import numpy as np

import signals as S
from prism import Prism

MARGIN = 0.35   # допуск локализации, с


def scenarios(seeds):
    out = []
    for kind in ("audio", "radio", "sensor"):
        for sd in seeds:
            out.append((kind, "чистый", sd))
    for sd in seeds:
        out.append(("audio", "естественная смена", sd))
    for kind in ("audio", "radio", "sensor"):
        for t in ("склейка", "фильтр НЧ", "усиление", "клиппинг", "копия", "ресемплинг"):
            for sd in seeds:
                out.append((kind, t, sd))
    return out


def build(kind, case, sd):
    if case == "естественная смена":
        return S.natural_change(sd)
    x = S.make(kind, seed=sd)
    return {
        "чистый": lambda: (x, []),
        "склейка": lambda: S.tamper_splice(x, kind, sd),
        "фильтр НЧ": lambda: S.tamper_lowpass(x),
        "усиление": lambda: S.tamper_gain(x),
        "клиппинг": lambda: S.tamper_clip(x),
        "копия": lambda: S.tamper_copy(x),
        "ресемплинг": lambda: S.tamper_resample(x),
    }[case]()


def run_one(args):
    kind, case, sd = args
    y, truth = build(kind, case, sd)
    r = Prism(fs=S.FS).analyze(y)

    def near(ev):
        return any(ev[1] <= b + MARGIN and ev[2] >= a - MARGIN for a, b in truth)

    hit = any(near(ev) for ev in r.events) if truth else None
    false_ev = [ev for ev in r.events if not near(ev)]
    return dict(kind=kind, case=case, seed=sd, detected=r.tampered, hit=hit,
                n_false=len(false_ev),
                false=[(ev[0], round(ev[1], 2), ev[3]) for ev in false_ev],
                kinds=sorted({ev[0] for ev in r.events if near(ev)}))


def main():
    nseeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    seeds = list(range(100, 100 + nseeds))
    sc = scenarios(seeds)
    with Pool(2) as pool:
        res = pool.map(run_one, sc, chunksize=4)
    json.dump(res, open("results.json", "w"), ensure_ascii=False, indent=1)

    print(f"\n{'природа':8} {'случай':20} {'обнаружено':>11} {'место верно':>12} {'ложн. событий':>14}")
    keys = []
    for r in res:
        k = (r["kind"], r["case"])
        if k not in keys:
            keys.append(k)
    for k in keys:
        g = [r for r in res if (r["kind"], r["case"]) == k]
        det = np.mean([r["detected"] for r in g])
        hits = [r["hit"] for r in g if r["hit"] is not None]
        loc = f"{np.mean(hits):.0%}" if hits else "—"
        fe = np.mean([r["n_false"] for r in g])
        print(f"{k[0]:8} {k[1]:20} {det:>11.0%} {loc:>12} {fe:>14.2f}")
    print("\nЛожные события:")
    for r in res:
        for f in r["false"]:
            print(f"  {r['kind']}/{r['case']}/{r['seed']}: {f}")


if __name__ == "__main__":
    main()
