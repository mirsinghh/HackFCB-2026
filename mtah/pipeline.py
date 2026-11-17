"""Possession labelling from the tactical ATD only (no video, no ground truth of any kind).

Pipeline
  1. parse tracking -> metres, team of each slot from the ATD metadata (first team = home)
  2. per-frame *team evidence*  e_t  (>0 favours H): who is closer to the ball, weighted by how
     close the ball is to anyone (a ball in flight tells us little).  Missing ball -> 0.
  3. per-frame *dead-ball evidence* from (a) how slowly the 22 players move (walking = stoppage),
     (b) ball outside the pitch, (c) ball moving fast (=> alive).
     All thresholds are physical constants (walking pace, distance to a line...), not fitted.
  4. 3-state Viterbi (H / A / D) per half with switching penalties  -> minimum-duration possessions.
  5. frames outside the two graded halves -> X.
"""
import os
import sys
import xml.etree.ElementTree as ET
import numpy as np
from scipy.ndimage import uniform_filter1d, median_filter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import asset_paths, half_video_blocks, GAME, GAMES  # noqa: E402
from mtah.load_tracking import load  # noqa: E402
from mtah.player_only import player_only_logit  # noqa: E402

FPS = 25
N = 199319
PITCH = GAMES[GAME]["pitch"]
H, A, D = 0, 1, 2


def team_of_slots():
    root = ET.parse(asset_paths()["atd_meta"]).getroot()
    ids = [t.get("id") for t in root.iter("Team")]
    return np.array([0 if p.get("teamId") == ids[0] else 1 for p in root.iter("Player")])


def interp_short(ball, max_gap):
    """Linear interpolation of the ball across gaps of <= max_gap frames."""
    out = ball.copy()
    vis = np.isfinite(ball[:, 0])
    idx = np.flatnonzero(vis)
    if len(idx) < 2:
        return out
    gaps = np.diff(idx)
    for a, g in zip(idx[:-1][gaps > 1], gaps[gaps > 1]):
        if g - 1 <= max_gap:
            t = np.arange(1, g)
            for c in (0, 1):
                out[a + 1:a + g, c] = ball[a, c] + (ball[a + g, c] - ball[a, c]) * t / g
    return out


def team_evidence(P, B, team, p):
    """Soft evidence (log-odds-like, >0 = home) from ball/player geometry."""
    d = np.sqrt(((P - B[:, None, :]) ** 2).sum(-1))
    d[~np.isfinite(d)] = np.inf
    dH = np.where(team[None, :] == 0, d, np.inf).min(1)
    dA = np.where(team[None, :] == 1, d, np.inf).min(1)
    dmin = np.minimum(dH, dA)
    ok = np.isfinite(B[:, 0]) & np.isfinite(dmin) & (dmin < 1e6)
    diff = np.where(np.isfinite(dA) & np.isfinite(dH), dA - dH, 0.0)
    diff = np.where(np.isfinite(diff), diff, 0.0)
    e = np.clip(diff / p["scale"], -p["clip"], p["clip"])
    w = np.exp(-np.where(ok, dmin, 99.0) / p["near"])
    e = np.where(ok, e * w, 0.0)
    return e, np.where(ok, dmin, np.nan)


def player_speed(P):
    v = np.hypot(*(np.diff(P, axis=0).transpose(2, 0, 1))) * FPS
    v = np.vstack([v[:1], v])
    v[v > 12] = np.nan
    with np.errstate(all="ignore"):
        return np.nanmedian(v, axis=1)


def viterbi(emit, lam_ha, lam_ad):
    """emit: (T,3) log-likelihood-like scores (higher better). Switching costs between states."""
    T = emit.shape[0]
    pen = np.array([[0, lam_ha, lam_ad], [lam_ha, 0, lam_ad], [lam_ad, lam_ad, 0]], dtype=np.float64)
    score = emit[0].copy()
    back = np.zeros((T, 3), dtype=np.int8)
    for t in range(1, T):
        cand = score[:, None] - pen  # from i to j
        b = cand.argmax(0)
        back[t] = b
        score = cand[b, np.arange(3)] + emit[t]
    path = np.zeros(T, dtype=np.int8)
    path[-1] = score.argmax()
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def run_lengths(mask):
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def dead_evidence(Bn, P, sspd, p):
    """Per-frame log-odds-like evidence that the ball is dead (>0) vs alive (<0). Rules are
    physical, not fitted: stoppages mean players walk, the ball sits near a line or leaves the
    pitch, and the tracker often loses the ball while a restart is set up."""
    L, W = PITCH
    vis = np.isfinite(Bn[:, 0])
    x, y = Bn[:, 0] * L, Bn[:, 1] * W
    outside = vis & ((Bn[:, 0] < 0) | (Bn[:, 0] > 1) | (Bn[:, 1] < 0) | (Bn[:, 1] > 1))
    edge = vis & (np.minimum.reduce([x, L - x, y, W - y]) < p["edge_m"])
    # ball speed over ~1 s on the interpolated track
    xs = np.where(vis, x, np.nan); ys = np.where(vis, y, np.nan)
    k = 25
    dx = np.r_[np.full(k, np.nan), xs[k:] - xs[:-k]]; dy = np.r_[np.full(k, np.nan), ys[k:] - ys[:-k]]
    bspd = np.hypot(dx, dy)  # m per second
    slow_ball = vis & (bspd < p["ball_slow"])
    fast_ball = vis & (bspd > p["ball_fast"])
    # long missing runs
    s, e = run_lengths(~vis)
    longmiss = np.zeros(len(vis), bool)
    for a, b in zip(s, e):
        if b - a >= p["miss_min"]:
            longmiss[a:b] = True
    slow_players = sspd < p["walk_ms"]
    # local fraction smoothing so single frames do not flip evidence
    sm = lambda m, w: uniform_filter1d(m.astype(np.float32), w)
    ev = (-p["d_bias"]
          + p["d_walk"] * slow_players
          + p["d_out"] * (sm(outside, 25) > 0)
          + p["d_edge"] * (sm(edge & slow_ball, 50) > 0.5)
          + p["d_miss"] * (longmiss & (sspd < p["walk_ms"] + 0.4))
          - p["d_fast"] * (sm(fast_ball, 25) > 0.3))
    return ev


# Published by the organisers (FAQ.md / starter/load_smart_tagging.py): the 2nd half is graded from
# its first tracked frame but the kick-off is only at 113742; the frames before it are dead ball.
KICKOFF_2ND_HALF = 113742

PARAMS = dict(
    interp_gap=25,        # interpolate ball over <=1 s
    scale=2.0, clip=3.0,  # metres: distance difference that yields one unit of evidence
    near=6.0,             # evidence decays with distance of the ball to the nearest player (m)
    speed_win=100,        # 4 s window for the "everyone is walking" detector
    walk_ms=1.5,          # median player speed (m/s) under which the 22 players are "walking"
    edge_m=3.0, ball_slow=2.5, ball_fast=9.0, miss_min=100,
    d_bias=2.5, d_walk=2.0, d_out=3.0, d_edge=2.0, d_miss=1.5, d_fast=1.5,
    ctrl_m=2.0,           # a player this close to a visible ball is taken as controlling it (pseudo-label)
    po_gain=0.15, po_clip=2.0,   # weight of the player-only cue on frames where the ball is lost
    lam_ha=60.0,          # cost of a hand-over (evidence units, ~ 2-3 s of clear evidence)
    lam_ad=60.0,          # cost of entering / leaving a stoppage
)


def run(params=None, debug=None):
    """Returns the label array. If `debug` is a dict it is filled with the intermediate signals."""
    p = dict(PARAMS, **(params or {}))
    pl, _, ball = load()
    scale = np.array(PITCH, dtype=np.float32)
    P = pl * scale
    Bn = interp_short(ball, p["interp_gap"])
    B = Bn * scale
    team = team_of_slots()
    e, dmin = team_evidence(P, B, team, p)

    spd = player_speed(P)
    spd = np.where(np.isfinite(spd), spd, np.nanmedian(spd))
    sspd = uniform_filter1d(spd, p["speed_win"])

    blocks = [(a, b) for _, a, b in half_video_blocks(GAME)]
    graded = np.zeros(N, bool)
    for a, b in blocks:
        graded[a:b + 1] = True
    lr = dead_evidence(Bn, P, sspd, p)

    # pseudo-labels: team of the nearest player when it is within ctrl_m of a visible ball
    near = np.isfinite(dmin) & (dmin < p["ctrl_m"])
    pseudo = np.where(near, np.where(e > 0, 1, -1), 0)
    logit, _, _, _, _ = player_only_logit(pl, team, blocks, pseudo, graded)
    po = np.clip(np.nan_to_num(logit, nan=0.0), -p["po_clip"], p["po_clip"]) * p["po_gain"]
    e = np.where(np.isfinite(Bn[:, 0]), e, po)

    labels = np.full(N, "X", dtype="<U1")
    names = np.array(["H", "A", "D"])
    for a, b in blocks:
        emit = np.zeros((b - a + 1, 3))
        emit[:, H] = e[a:b + 1] - 0.5 * lr[a:b + 1]
        emit[:, A] = -e[a:b + 1] - 0.5 * lr[a:b + 1]
        emit[:, D] = 0.5 * lr[a:b + 1]
        pre = min(max(KICKOFF_2ND_HALF - a, 0), b - a + 1) if a > 100000 else 0
        if pre:  # published: everything before the 2nd-half kick-off is dead ball
            emit[:pre, D] += 1e4
        path = viterbi(emit, p["lam_ha"], p["lam_ad"])
        labels[a:b + 1] = names[path]
    if debug is not None:
        debug.update(team_evidence=e, dead_evidence=lr, ball=Bn, player_speed=sspd)
    return labels


def write_csv(labels, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, l) for i, l in enumerate(labels))


def summarize(labels):
    out = {}
    seg = []
    y = labels
    s = 0
    for i in range(1, len(y) + 1):
        if i == len(y) or y[i] != y[s]:
            seg.append((y[s], i - s))
            s = i
    pos = [d for l, d in seg if l in "HA"]
    dead = [d for l, d in seg if l == "D"]
    return dict(H=int((y == "H").sum()), A=int((y == "A").sum()), D=int((y == "D").sum()),
                X=int((y == "X").sum()), n_poss=len(pos), mean_poss_s=float(np.mean(pos) / FPS),
                n_dead=len(dead), mean_dead_s=float(np.mean(dead) / FPS) if dead else 0.0)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="submission/submission.csv")
    a = ap.parse_args()
    lab = run()
    write_csv(lab, a.out)
    print(summarize(lab))
