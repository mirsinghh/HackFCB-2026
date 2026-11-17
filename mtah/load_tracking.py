"""Parse the tactical ATD txt into dense numpy arrays (cached as .npz next to the data)."""
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import asset_paths, source_frames  # noqa: E402


def load(source="tactical", use_cache=True):
    """Returns players (T,46,2) normalised x,y ; speed (T,46); ball (T,2). NaN where missing."""
    paths = asset_paths(source=source)
    raw = paths["atd_raw"]
    cache = os.path.join(os.path.dirname(raw), "atd_cache.npz")
    if use_cache and os.path.isfile(cache):
        z = np.load(cache)
        return z["players"], z["speed"], z["ball"]
    n = source_frames(source)
    players = np.full((n, 46, 2), np.nan, dtype=np.float32)
    speed = np.full((n, 46), np.nan, dtype=np.float32)
    ball = np.full((n, 2), np.nan, dtype=np.float32)
    with open(raw) as fh:
        for line in fh:
            p = line.rstrip("\n").split(":")
            if len(p) != 3:
                continue
            f = int(p[0]) - 1
            if not 0 <= f < n:
                continue
            for s, g in enumerate(p[1].split(";")):
                if g[0] == "N":
                    continue
                a = g.split(",")
                players[f, s, 0] = float(a[0]); players[f, s, 1] = float(a[1])
                if a[2][0] != "N":
                    speed[f, s] = float(a[2])
            if p[2][0] != "N":
                a = p[2].split(",")
                ball[f, 0] = float(a[0]); ball[f, 1] = float(a[1])
    np.savez(cache, players=players, speed=speed, ball=ball)
    return players, speed, ball


if __name__ == "__main__":
    pl, sp, b = load(use_cache=False)
    print(pl.shape, np.isfinite(b[:, 0]).mean())
