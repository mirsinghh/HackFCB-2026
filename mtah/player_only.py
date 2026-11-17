"""Self-supervised, player-only possession cue for frames where the tracker has lost the ball.

Pseudo-labels come from our own geometric rule on frames where the ball is visible AND a player is
within CTRL_M metres of it (an algorithmic label, no human annotation, no ground truth).
A logistic regression on team-shape features is trained on those frames and then applied to
frames with no ball.
"""
import numpy as np
from scipy.ndimage import uniform_filter1d

FPS = 25


def _half_sign(P, team, a, b):
    """+1 if the home team is, on the median frame of the block, on the left of the away team."""
    seg = P[a:b + 1]
    with np.errstate(all="ignore"):
        xh = np.nanmedian(np.nanmean(seg[:, team == 0, 0], 1))
        xa = np.nanmedian(np.nanmean(seg[:, team == 1, 0], 1))
    return 1.0 if xh < xa else -1.0


def attack_signs(P, team, blocks):
    """+1 if the home team attacks towards +x in that half.

    The sides are read from the whole first half (median over frames of each team's mean x) and the
    second half is its mirror image, because teams change ends at half-time. We do NOT read the
    start of the second block: it begins ~2 minutes before kick-off, while players are still
    walking onto the pitch, and gives the wrong side. The second half is checked against its own
    data and a warning is raised if it disagrees with the mirror rule."""
    import warnings
    s1 = _half_sign(P, team, *blocks[0])
    signs = [s1 if i % 2 == 0 else -s1 for i in range(len(blocks))]
    for (a, b), sg in zip(blocks[1:], signs[1:]):
        if _half_sign(P, team, a, b) != sg:
            warnings.warn("second-half side disagrees with the mirror of the first half")
    return signs


def features(Pn, team, sign_per_frame):
    """Pn normalised (0-1) positions (T,46,2). Returns (T,k) feature matrix (NaN allowed)."""
    X = Pn[:, :, 0]; Y = Pn[:, :, 1]
    mh = team == 0; ma = team == 1
    with np.errstate(all="ignore"):
        xh = np.nanmean(X[:, mh], 1); xa = np.nanmean(X[:, ma], 1)
        yh = np.nanstd(Y[:, mh], 1); ya = np.nanstd(Y[:, ma], 1)
        xall = np.nanmean(X, 1)
        sh = np.nanstd(X[:, mh], 1); sa = np.nanstd(X[:, ma], 1)
    s = sign_per_frame
    uh = s * (xh - 0.5); ua = -s * (xa - 0.5)
    cu = s * (xall - 0.5)
    k = 25
    def vel(x):
        v = np.r_[np.full(k, np.nan), x[k:] - x[:-k]]
        return v
    vh = s * vel(xh) * FPS / k * 25; va = -s * vel(xa) * FPS / k * 25
    F = np.stack([uh, ua, uh - ua, yh - ya, sh - sa, cu, vh, va, vh - va], 1)
    return F


def player_only_logit(Pn, team, blocks, pseudo, usable):
    """pseudo: (T,) +1 home / -1 away / 0 unknown. usable: bool mask of frames to fit on.
    Returns logit (T,) for every frame (NaN where features undefined)."""
    from sklearn.linear_model import LogisticRegression
    T = len(Pn)
    signs = attack_signs(Pn, team, blocks)
    s = np.zeros(T)
    for (a, b), sg in zip(blocks, signs):
        s[a:b + 1] = sg
    F = features(Pn, team, s)
    # smooth features over 2 s so single-frame tracker noise does not dominate
    Fs = np.where(np.isfinite(F), F, np.nan)
    for j in range(F.shape[1]):
        col = Fs[:, j]
        m = np.isfinite(col)
        fill = np.where(m, col, 0.0)
        num = uniform_filter1d(fill, 50); den = uniform_filter1d(m.astype(float), 50)
        Fs[:, j] = np.where(den > 0.3, num / np.maximum(den, 1e-9), np.nan)
    ok = np.isfinite(Fs).all(1) & (s != 0)
    fit = ok & usable & (pseudo != 0)
    mu = Fs[fit].mean(0); sd = Fs[fit].std(0) + 1e-9
    Z = (Fs - mu) / sd
    clf = LogisticRegression(C=0.5, class_weight="balanced", max_iter=500)
    idx = np.flatnonzero(fit)
    clf.fit(Z[idx], (pseudo[idx] > 0).astype(int))
    logit = np.full(T, np.nan)
    logit[ok] = clf.decision_function(Z[ok])
    return logit, clf, Z, ok, fit
