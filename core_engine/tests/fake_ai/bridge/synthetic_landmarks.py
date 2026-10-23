"""Synthetic MediaPipe-style landmarks for the tests (numpy only, no MediaPipe).

make_face(): 478 points, with the 12 eye points + 2 iris points placed so that the Eye Aspect Ratio
             and the iris gaze ratios that gaze_intent.py computes behave like a real face.
make_hand(): 21 points with a controllable index fingertip, curled fingers and thumb position.
make_hand_world() / pack_hand(): the matching MediaPipe WORLD landmarks (metres, wrist-centred) and the
             42-row array the real HandLandmarkExtractor returns (21 image rows + 21 world rows).
All values are float32 in normalised image coordinates, exactly what MediaPipe returns.
"""
import numpy as np

N_FACE, N_HAND = 478, 21

_BASE_RNG = np.random.default_rng(1234)
_BASE_XY = _BASE_RNG.uniform(-1.0, 1.0, (N_FACE, 2)).astype(np.float64)
_BASE_Z = _BASE_RNG.uniform(-0.04, 0.04, N_FACE)

# (corner_a, top_a, top_b, corner_b, bottom_b, bottom_a)  == the order of LEFT_EYE / RIGHT_EYE in gaze_intent.py
_LEFT_EYE = (33, 160, 158, 133, 153, 144)
_RIGHT_EYE = (362, 385, 387, 263, 373, 380)
_LEFT_VERT, _RIGHT_VERT = (159, 145), (386, 374)
_LEFT_IRIS, _RIGHT_IRIS = 468, 473

_HALF_W, _APERTURE = 0.035, 0.012


def _eye(pts, ids, vert, iris_idx, cx, cy, k, iris):
    ca, ta, tb, cb, bb, ba = ids
    ap = _APERTURE * k
    pts[ca] = (cx - _HALF_W, cy)
    pts[cb] = (cx + _HALF_W, cy)
    pts[ta] = (cx - 0.4 * _HALF_W, cy - ap)
    pts[ba] = (cx - 0.4 * _HALF_W, cy + ap)
    pts[tb] = (cx + 0.4 * _HALF_W, cy - ap)
    pts[bb] = (cx + 0.4 * _HALF_W, cy + ap)
    pts[vert[0]] = (cx, cy - 1.1 * ap)           # upper lid centre
    pts[vert[1]] = (cx, cy + 1.1 * ap)           # lower lid centre
    pts[iris_idx] = (cx - _HALF_W + 2 * _HALF_W * iris[0], cy - 1.1 * ap + 2.2 * ap * iris[1])


def make_face(k_left=1.0, k_right=1.0, iris=(0.5, 0.5), center=(0.5, 0.5), size=0.25, rng=None, noise=0.0):
    """k_* : eye openness factor (1.0 = open, ~0.25 = closed). iris: position inside the eye, 0..1."""
    pts = center + _BASE_XY * size
    _eye(pts, _LEFT_EYE, _LEFT_VERT, _LEFT_IRIS, center[0] - 0.08, center[1] - 0.05, k_left, iris)
    _eye(pts, _RIGHT_EYE, _RIGHT_VERT, _RIGHT_IRIS, center[0] + 0.08, center[1] - 0.05, k_right, iris)
    z = _BASE_Z.copy()
    if rng is not None and noise > 0:
        pts = pts + rng.normal(0.0, noise, pts.shape)
        z = z + rng.normal(0.0, noise, z.shape)
    out = np.empty((N_FACE, 3), dtype=np.float32)
    out[:, :2] = pts
    out[:, 2] = z
    return out


# hand layout: offsets from the wrist of the four knuckles (index, middle, ring, pinky) and the fingertip rise
_MCP_OFF = {5: (-0.06, -0.19), 9: (0.0, -0.20), 13: (0.05, -0.19), 17: (0.09, -0.17)}
_TIP_OF = {5: 8, 9: 12, 13: 16, 17: 20}
_CHAIN = {5: (6, 7), 9: (10, 11), 13: (14, 15), 17: (18, 19)}


def make_hand(tip=(0.5, 0.4), curled=(False, False, False, False), thumb=None, pinch=False,
              scale=1.0, rng=None, noise=0.0):
    """tip: where the index fingertip is when the index finger is extended.
    curled: (index, middle, ring, pinky). thumb: explicit thumb-tip (x, y) or None.
    pinch: if thumb is None, put the thumb tip next to the index tip (True) or far away (False)."""
    s = scale
    wrist = np.array([tip[0] + 0.06 * s, tip[1] + 0.34 * s])
    lm = np.zeros((N_HAND, 3), dtype=np.float64)
    lm[0, :2] = wrist
    for finger, mcp_idx in enumerate((5, 9, 13, 17)):
        mcp = wrist + np.array(_MCP_OFF[mcp_idx]) * s
        if curled[finger]:
            tip_pt = wrist + 0.55 * (mcp - wrist)                   # fingertip closer to the wrist than its knuckle
        else:
            tip_pt = mcp + np.array([0.0, -0.15 * s])
        lm[mcp_idx, :2] = mcp
        lm[_TIP_OF[mcp_idx], :2] = tip_pt
        p1, p2 = _CHAIN[mcp_idx]
        lm[p1, :2] = mcp + 0.33 * (tip_pt - mcp)
        lm[p2, :2] = mcp + 0.66 * (tip_pt - mcp)
    index_tip = lm[8, :2]
    if thumb is not None:
        thumb_pt = np.array(thumb, dtype=np.float64)
    elif pinch:
        thumb_pt = index_tip + np.array([0.004, 0.004])
    else:
        thumb_pt = wrist + np.array([-0.20, -0.12]) * s
    lm[4, :2] = thumb_pt
    for i, f in zip((1, 2, 3), (0.25, 0.5, 0.75)):
        lm[i, :2] = wrist + f * (thumb_pt - wrist)
    if rng is not None and noise > 0:
        lm = lm + rng.normal(0.0, noise, lm.shape)
    return lm.astype(np.float32)


def make_hand_world(img, rng=None, noise=0.0, z_sigma=0.01):
    """World landmarks matching an image-landmark hand (wrist-centred, ~0.3 m per image unit, small depth noise)."""
    w = (img.astype(np.float64) - img[0].astype(np.float64)) * 0.30
    if rng is not None:
        w[:, 2] = rng.normal(0.0, z_sigma, N_HAND)
        if noise > 0:
            w = w + rng.normal(0.0, noise, w.shape)
    else:
        w[:, 2] = 0.0
    return w.astype(np.float32)


def pack_hand(img, world):
    """(42, 3) float32: rows 0..20 image landmarks, rows 21..41 world landmarks (HandLandmarkExtractor contract)."""
    return np.concatenate([img.astype(np.float32), world.astype(np.float32)])
