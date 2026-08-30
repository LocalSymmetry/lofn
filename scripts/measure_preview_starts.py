#!/usr/bin/env python3
"""measure_preview_starts.py — where does a stranger enter this song?

DistroKid has one submission field, "Preview clip start time (TikTok, Apple
Music, iTunes)", with one value per track, in mm:ss. That single integer second
decides which fifteen seconds of a record a TikTok editor is handed, and which
thirty seconds Apple plays. Spotify is NOT controlled by it.

This script measures, per track, WHERE THE MOST-REUSED, LOUDEST, FULLEST,
CLEANEST-EDGED eight seconds of the song begins, snaps that point to a 4/4
downbeat, and rounds it to the integer second one beat earlier — so the hook
lands rather than starts cold. See output/distribution/PREVIEW_STARTS.md for the
panel that derived the rule and for the honest limits.

    python3 scripts/measure_preview_starts.py --root output/distribution/DELIVER \\
        --out output/distribution/preview_starts.json

Every number here is a PROXY for a perceptual judgement nobody in the loop can
make: neither the harness nor the author of this file can hear the audio. The
per-track `confidence` and the `conflicts` list are therefore the most important
outputs, not the timestamps.

MEMORY: audio is streamed, never fully resident. float32 throughout, one
channel at a time through the K-weighting recursion, band reductions applied to
each FFT block before it is released. Peak RSS per track is a few tens of MB
regardless of track length. (A 178 MiB float64 allocation has failed on this
machine before; see compare_masters.py.)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import find_peaks, lfilter, lfilter_zi

# ---------------------------------------------------------------------------
# analysis constants  (all times in seconds unless suffixed _f = frames)
# ---------------------------------------------------------------------------
NFFT_A, HOP_A = 1024, 512        # 21.3 ms window / 10.67 ms hop — onsets, beats
NFFT_B, HOP_B = 8192, 4096       # 170 ms / 85 ms — chroma + timbre for structure
STRUCT_HOP = 0.5                 # structure/scoring grid
CLIP_S = 8.0                     # the window a TikTok user actually keeps
LUFS_HOP = 0.1                   # short-term loudness hop
LUFS_WIN = 3.0                   # short-term loudness window (BS.1771 "S")
MIN_REPEAT_LAG = 15.0            # a chorus does not recur sooner than this
TAIL_GUARD = 40.0                # leave >= this much audio after the start
READ_CHUNK = 480000              # 10 s at 48 kHz; a whole number of 0.1 s hops

# ITU-R BS.1770-4 K-weighting (48 kHz). float32 coefficients so scipy does not
# silently promote the whole signal to float64. Lifted from compare_masters.py.
SHELF_B = np.array([1.53512485958697, -2.69169618940638, 1.19839281085285], dtype=np.float32)
SHELF_A = np.array([1.0, -1.69065929318241, 0.73248077421585], dtype=np.float32)
HPF_B = np.array([1.0, -2.0, 1.0], dtype=np.float32)
HPF_A = np.array([1.0, -1.99004745483398, 0.99007225036621], dtype=np.float32)

BANDS = (("sub", 20, 80), ("low", 80, 300), ("mid", 300, 2000),
         ("high", 2000, 8000), ("air", 8000, 16000))


# ---------------------------------------------------------------------------
# streaming framer
# ---------------------------------------------------------------------------
class Framer:
    """Accumulates mono samples and emits |STFT|^2 blocks, releasing as it goes."""

    def __init__(self, nfft: int, hop: int):
        self.nfft, self.hop = nfft, hop
        self.win = np.hanning(nfft).astype(np.float32)
        self.buf = np.zeros(0, dtype=np.float32)

    def push(self, mono: np.ndarray):
        self.buf = mono if self.buf.size == 0 else np.concatenate([self.buf, mono])
        if self.buf.size < self.nfft:
            return None
        nf = 1 + (self.buf.size - self.nfft) // self.hop
        fr = sliding_window_view(self.buf, self.nfft)[:: self.hop][:nf]
        X = np.fft.rfft(fr * self.win, axis=1)
        P = (X.real.astype(np.float32) ** 2 + X.imag.astype(np.float32) ** 2)
        del X, fr
        self.buf = self.buf[nf * self.hop:].copy()
        return P


def band_matrix(freqs: np.ndarray, edges) -> np.ndarray:
    """(n_bands, n_bins) 0/1 selection matrix."""
    M = np.zeros((len(edges), len(freqs)), dtype=np.float32)
    for i, (lo, hi) in enumerate(edges):
        M[i] = ((freqs >= lo) & (freqs < hi)).astype(np.float32)
    return M


def log_edges(lo: float, hi: float, n: int):
    e = np.geomspace(lo, hi, n + 1)
    return [(float(e[i]), float(e[i + 1])) for i in range(n)]


# ---------------------------------------------------------------------------
# pass 1 — K-weighted loudness + fine-hop onset/band features
# ---------------------------------------------------------------------------
def pass_one(path: Path):
    info = sf.info(str(path))
    sr = info.samplerate
    freqs_a = np.fft.rfftfreq(NFFT_A, 1 / sr)
    Mband = band_matrix(freqs_a, [(lo, hi) for _, lo, hi in BANDS])          # 5 x bins
    Mflux = band_matrix(freqs_a, log_edges(40.0, min(16000.0, sr / 2 - 1), 40))  # 40 x bins

    hop_n = int(round(LUFS_HOP * sr))
    fr = Framer(NFFT_A, HOP_A)
    band_blocks, flux_blocks, hop_sums = [], [], []
    kcarry = None          # leftover K-weighted samples not filling a 0.1 s hop
    zi1 = zi2 = None

    with sf.SoundFile(str(path)) as f:
        nch = f.channels
        while True:
            x = f.read(READ_CHUNK, dtype="float32", always_2d=True)
            if x.shape[0] == 0:
                break
            # --- K-weighting, one channel at a time (bounds peak RSS) ---
            if zi1 is None:
                z1 = lfilter_zi(SHELF_B, SHELF_A).astype(np.float32)
                z2 = lfilter_zi(HPF_B, HPF_A).astype(np.float32)
                zi1 = np.stack([z1 * x[0, c] for c in range(nch)], axis=1)
                zi2 = np.stack([z2 * 0.0 for _ in range(nch)], axis=1)
            kf = np.empty_like(x)
            for c in range(nch):
                col = np.ascontiguousarray(x[:, c], dtype=np.float32)
                y, zi1[:, c] = lfilter(SHELF_B, SHELF_A, col, zi=zi1[:, c])
                y, zi2[:, c] = lfilter(HPF_B, HPF_A, y.astype(np.float32), zi=zi2[:, c])
                kf[:, c] = y.astype(np.float32)
                del col, y
            k = kf if kcarry is None else np.concatenate([kcarry, kf])
            del kf
            nh = k.shape[0] // hop_n
            if nh:
                hop_sums.append(
                    np.square(k[: nh * hop_n], dtype=np.float32)
                    .reshape(nh, hop_n, nch)
                    .sum(axis=1, dtype=np.float64)
                )
            kcarry = k[nh * hop_n:].copy()
            del k

            # --- spectral features ---
            P = fr.push(x.mean(1))
            del x
            if P is not None:
                band_blocks.append(P @ Mband.T)
                flux_blocks.append(P @ Mflux.T)
                del P

    band = np.concatenate(band_blocks) if band_blocks else np.zeros((0, 5), np.float32)
    flux_bands = np.concatenate(flux_blocks) if flux_blocks else np.zeros((0, 40), np.float32)
    hs = np.concatenate(hop_sums) if hop_sums else np.zeros((0, 1))
    return sr, band, flux_bands, hs, hop_n, nch, info.frames / sr


# ---------------------------------------------------------------------------
# pass 2 — chroma + timbre for the self-similarity matrix
# ---------------------------------------------------------------------------
def pass_two(path: Path, sr: int):
    freqs_b = np.fft.rfftfreq(NFFT_B, 1 / sr)
    # chroma: bins between A2 and A7, folded to 12 pitch classes
    sel = (freqs_b >= 110.0) & (freqs_b <= 3520.0)
    pc = np.zeros(len(freqs_b), dtype=np.int64) - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        pc[sel] = np.mod(np.round(12 * np.log2(freqs_b[sel] / 440.0)).astype(np.int64), 12)
    Mchroma = np.zeros((12, len(freqs_b)), dtype=np.float32)
    for i in range(12):
        Mchroma[i] = (pc == i).astype(np.float32)
    Mtimbre = band_matrix(freqs_b, log_edges(50.0, min(16000.0, sr / 2 - 1), 12))

    fr = Framer(NFFT_B, HOP_B)
    ch_blocks, tb_blocks = [], []
    with sf.SoundFile(str(path)) as f:
        while True:
            x = f.read(READ_CHUNK, dtype="float32", always_2d=True)
            if x.shape[0] == 0:
                break
            P = fr.push(x.mean(1))
            del x
            if P is not None:
                A = np.sqrt(P)                       # magnitude, less peaky than power
                ch_blocks.append(A @ Mchroma.T)
                tb_blocks.append(A @ Mtimbre.T)
                del P, A
    chroma = np.concatenate(ch_blocks) if ch_blocks else np.zeros((0, 12), np.float32)
    timbre = np.concatenate(tb_blocks) if tb_blocks else np.zeros((0, 12), np.float32)
    return chroma, timbre


# ---------------------------------------------------------------------------
# derived measures
# ---------------------------------------------------------------------------
def short_term_lufs(hop_sums: np.ndarray, hop_n: int) -> np.ndarray:
    """LUFS-S on a 0.1 s grid: 3 s window, channel-summed mean square."""
    w = int(round(LUFS_WIN / LUFS_HOP))
    if len(hop_sums) < w:
        return np.full(max(len(hop_sums), 1), -70.0)
    c = np.cumsum(np.vstack([np.zeros((1, hop_sums.shape[1])), hop_sums]), axis=0)
    blk = (c[w:] - c[:-w]) / (w * hop_n)
    p = blk.sum(axis=1)
    out = -0.691 + 10 * np.log10(p + 1e-12)
    # align to window START, pad the tail so index == start time / 0.1 s
    return np.concatenate([out, np.full(w - 1, out[-1])])


def onset_envelope(flux_bands: np.ndarray) -> np.ndarray:
    """Positive spectral flux over 40 log bands (Böck-style half-wave rectified)."""
    L = np.log1p(flux_bands * 1e4)
    d = np.diff(L, axis=0, prepend=L[:1])
    d[d < 0] = 0.0
    o = d.sum(axis=1)
    # subtract a 0.5 s moving median-ish baseline so quiet passages still register
    k = 47
    base = np.convolve(o, np.ones(k) / k, mode="same")
    o = np.maximum(o - base, 0.0)
    return o.astype(np.float32)


def estimate_tempo(o: np.ndarray, fps: float, bpm_lo=55.0, bpm_hi=190.0):
    z = o - o.mean()
    if z.std() < 1e-9:
        return None, 0.0
    z = z / z.std()
    n = 1 << int(np.ceil(np.log2(2 * len(z))))
    F = np.fft.rfft(z, n)
    ac = np.fft.irfft(F * np.conj(F), n)[: len(z)]
    ac /= (ac[0] + 1e-12)
    lo = max(2, int(fps * 60 / bpm_hi))
    hi = min(len(ac) - 1, int(fps * 60 / bpm_lo))
    if hi <= lo:
        return None, 0.0
    per = np.arange(lo, hi + 1)
    bpms = 60 * fps / per
    prior = np.exp(-0.5 * (np.log2(bpms / 120.0) / 0.9) ** 2)
    sc = ac[per] * prior
    i = int(np.argmax(sc))
    period = float(per[i])
    salience = float(sc[i] / (np.median(np.abs(sc)) + 1e-12))
    return period, salience


def beat_track(o: np.ndarray, period: float, alpha=0.55, tightness=100.0):
    """Ellis-style dynamic-programming beat tracker on the onset envelope."""
    loc = o - o.mean()
    s = loc.std()
    loc = loc / (s + 1e-9)
    N = len(loc)
    w = np.arange(-int(round(2 * period)), -int(round(period / 2)) + 1)
    w = w[w < 0]
    if len(w) == 0:
        return np.zeros(0, dtype=np.int64)
    txcost = -tightness * (np.log(-w / period)) ** 2
    cum = np.zeros(N, dtype=np.float64)
    back = np.full(N, -1, dtype=np.int64)
    for t in range(N):
        idx = t + w
        m = idx >= 0
        if m.any():
            sc = cum[idx[m]] + txcost[m]
            k = int(np.argmax(sc))
            cum[t] = alpha * sc[k] + (1 - alpha) * loc[t]
            back[t] = idx[m][k]
        else:
            cum[t] = (1 - alpha) * loc[t]
    tail = max(0, N - int(round(period)))
    t = int(tail + np.argmax(cum[tail:]))
    beats = []
    while t >= 0:
        beats.append(t)
        t = back[t]
    return np.array(beats[::-1], dtype=np.int64)


def novelty(S: np.ndarray, L: int) -> np.ndarray:
    """Foote checkerboard novelty with a Gaussian-tapered kernel.

    The untapered version (plain block means) was too noisy to peak-pick: on the
    first pass the nearest 'boundary' to a picked chorus entry came out 28 s away
    on a track whose loudness and arrangement measures both agreed to within
    half a second. A tapered kernel plus smoothing is what makes the boundary
    cross-check say anything at all.
    """
    N = S.shape[0]
    g = np.exp(-0.5 * (np.arange(-L, L) + 0.5) ** 2 / (L / 2.0) ** 2)
    K = np.outer(g, g)
    sign = np.ones((2 * L, 2 * L))
    sign[:L, L:] = -1.0
    sign[L:, :L] = -1.0
    K = K * sign
    K /= np.abs(K).sum()
    nov = np.zeros(N)
    P = np.pad(S, L, mode="edge")
    for t in range(N):
        nov[t] = float(np.sum(P[t: t + 2 * L, t: t + 2 * L] * K))
    # 2 s smoothing so peak-picking sees sections, not frame noise
    k = max(1, int(round(2.0 / STRUCT_HOP)))
    w = np.hanning(k + 2)[1:-1]
    w /= w.sum()
    return np.convolve(nov, w, mode="same")


def repetition_score(S: np.ndarray, min_lag_f: int, smooth_f: int) -> np.ndarray:
    """How strongly does the material at t recur elsewhere in the track?

    Time-lag smoothing: a CHORUS produces a long diagonal stripe in S, a single
    similar chord does not. Smoothing S along the diagonal direction before
    taking the max over lags is what separates 'this recurs' from 'this rhymes'.
    """
    N = S.shape[0]
    rep = np.zeros(N)
    if N < min_lag_f + 4:
        return rep
    ker = np.ones(smooth_f) / smooth_f
    for lag in range(min_lag_f, N - 3):
        d = np.diagonal(S, offset=lag)               # d[t] = S[t, t+lag]
        if len(d) < smooth_f:
            continue
        sm = np.convolve(d, ker, mode="same")
        rep[: len(sm)] = np.maximum(rep[: len(sm)], sm)
        rep[lag: lag + len(sm)] = np.maximum(rep[lag: lag + len(sm)], sm)
    return rep


def norm01(v: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(v)), float(np.max(v))
    if hi - lo < 1e-12:
        return np.zeros_like(v)
    return (v - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# the rule
# ---------------------------------------------------------------------------
# The composite score only chooses WHERE INSIDE a qualifying section to start.
# WHICH section is chosen is decided by the conjunction below, earliest-first,
# with no weights at all — the first attempt let a weighted plateau threshold
# choose the section, and that one number moved 20 of 44 tracks by more than
# four seconds. A parameter that swings half the catalogue is not a tie-break.
WEIGHTS = {"rep": 0.34, "loud": 0.22, "full": 0.18, "nov": 0.14, "onset": 0.12}
# Thresholds are ABSOLUTE, relative to the track's own maximum — not
# percentiles. A percentile conjunction (top 15% loudness AND top 15%
# repetition) intersects almost only at the final chorus, which silently
# defeated the "earliest" intent: on the first nine tracks it moved two picks
# more than two minutes later. "At the payoff level" means near the maximum,
# not in the top decile.
Q_LOUD_TOL = 1.75        # LU below the loudest 8 s window that still counts as payoff
Q_REP_FRAC = 0.85        # fraction of the track's peak repetition score
Q_FULL = 0.80            # fraction of the five bands within 3 dB of their best
SECTION_MIN_S = 4.0      # a qualifying run shorter than this is a spike
SECTION_GAP_S = 4.0      # runs separated by less than this are one section


def runs_of(mask: np.ndarray, gap_f: int, need_f: int):
    """Contiguous True runs, merging gaps shorter than gap_f, dropping short ones."""
    n = len(mask)
    out, i = [], 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        j = i
        while True:
            k = j
            while k < n and mask[k]:
                k += 1
            nxt = k
            while nxt < min(n, k + gap_f) and not mask[nxt]:
                nxt += 1
            if nxt < n and nxt - k < gap_f and mask[nxt]:
                j = nxt
            else:
                j = k
                break
        if j - i >= need_f:
            out.append((i, j))
        i = max(j, i + 1)
    return out


def analyse(path: Path, side: str, rng: np.random.Generator):
    sr, band, flux_bands, hop_sums, hop_n, nch, dur = pass_one(path)
    chroma, timbre = pass_two(path, sr)

    fps_a = sr / HOP_A
    fps_b = sr / HOP_B

    lufs_s = short_term_lufs(hop_sums, hop_n)
    onset = onset_envelope(flux_bands)

    # ---- beat grid -------------------------------------------------------
    period, salience = estimate_tempo(onset, fps_a)
    if period is None:
        beats_t = np.zeros(0)
        bpm = None
        beat_conf = 0.0
        ibi_cv = None
    else:
        bidx = beat_track(onset, period)
        beats_t = bidx / fps_a
        bpm = float(60 * fps_a / period)
        if len(beats_t) > 8:
            ibi = np.diff(beats_t)
            ibi_cv = float(np.std(ibi) / (np.mean(ibi) + 1e-9))
            osal = onset[np.clip(bidx, 0, len(onset) - 1)]
            hit = float(np.mean(osal > np.median(onset[onset > 0]) if (onset > 0).any() else 0))
            beat_conf = float(np.clip(
                0.45 * np.clip((salience - 1.0) / 3.0, 0, 1)
                + 0.35 * np.clip(1 - ibi_cv * 6, 0, 1)
                + 0.20 * np.clip(hit, 0, 1), 0, 1))
        else:
            ibi_cv, beat_conf = None, 0.0
    B = 60.0 / bpm if bpm else None          # one beat, seconds

    # ---- structure grid --------------------------------------------------
    n_g = int(dur / STRUCT_HOP)
    if n_g < 20:
        raise RuntimeError("track too short to analyse")
    g_t = np.arange(n_g) * STRUCT_HOP

    def to_grid(v: np.ndarray, fps: float, agg="mean") -> np.ndarray:
        idx = np.clip((g_t * fps).astype(int), 0, max(len(v) - 1, 0))
        idx2 = np.clip(((g_t + STRUCT_HOP) * fps).astype(int), 1, len(v))
        out = np.empty(n_g)
        for i in range(n_g):
            seg = v[idx[i]:max(idx2[i], idx[i] + 1)]
            out[i] = seg.mean() if agg == "mean" else seg.max()
        return out

    # structure features on the 0.5 s grid
    kb = max(1, int(round(STRUCT_HOP * fps_b)))
    nb = min(len(chroma), len(timbre)) // kb
    nb = min(nb, n_g)
    C = chroma[: nb * kb].reshape(nb, kb, 12).mean(1)
    T = np.log1p(timbre[: nb * kb].reshape(nb, kb, -1).mean(1) * 1e4)
    # Mean-remove BOTH before L2 normalising. Raw chroma vectors are all
    # non-negative, so their cosine similarity sits in a narrow high band and
    # the self-similarity matrix comes out almost flat — which flattens the
    # novelty curve and made the boundary cross-check useless on the first run.
    C = C - C.mean(axis=1, keepdims=True)
    C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
    T = T - T.mean(axis=1, keepdims=True)
    T = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
    S = 0.60 * (C @ C.T) + 0.40 * (T @ T.T)
    # contrast enhancement: median -> 0, 98th percentile -> 1
    p50, p98 = np.percentile(S, 50), np.percentile(S, 98)
    S = np.clip((S - p50) / max(p98 - p50, 1e-6), 0.0, 1.0)

    min_lag_f = int(MIN_REPEAT_LAG / STRUCT_HOP)
    rep_b = repetition_score(S, min_lag_f, smooth_f=int(8.0 / STRUCT_HOP))
    nov_b = novelty(S, L=int(8.0 / STRUCT_HOP))
    rep = np.interp(g_t, np.arange(nb) * STRUCT_HOP, rep_b)
    nov = np.interp(g_t, np.arange(nb) * STRUCT_HOP, nov_b)

    # loudness / fullness / onset density, all as "what the next 8 s look like"
    clip_f = int(CLIP_S / STRUCT_HOP)
    lu_g = to_grid(lufs_s, 1 / LUFS_HOP)
    on_g = to_grid(onset, fps_a)
    bnd_g = np.stack([to_grid(band[:, i], fps_a) for i in range(band.shape[1])], axis=1)
    bnd_db = 10 * np.log10(bnd_g + 1e-12)

    def fwd_mean(v):
        c = np.concatenate([[0.0], np.cumsum(v)])
        hi = np.minimum(np.arange(n_g) + clip_f, n_g)
        lo = np.arange(n_g)
        return (c[hi] - c[lo]) / np.maximum(hi - lo, 1)

    loud_f = fwd_mean(lu_g)
    onset_f = fwd_mean(on_g)
    rep_f = fwd_mean(rep)
    bandf = np.stack([fwd_mean(bnd_db[:, i]) for i in range(bnd_db.shape[1])], axis=1)
    # fullness: how many of the five bands sit within 3 dB of their best 8 s window
    bmax = bandf.max(axis=0)
    full_f = np.mean(bandf > (bmax - 3.0), axis=1)
    # arrangement-entry step: fullness after minus fullness before
    back = np.concatenate([np.full(clip_f, full_f[0]), full_f[:-clip_f]]) if n_g > clip_f else full_f
    step_f = full_f - back

    # ---- valid candidate window -----------------------------------------
    hi_t = dur - TAIL_GUARD
    valid = (g_t >= 0.0) & (g_t <= hi_t)
    if valid.sum() < 10:
        valid = g_t <= max(dur - 20.0, dur * 0.5)

    comps = {
        "rep": norm01(rep_f[valid]),
        "loud": norm01(loud_f[valid]),
        "full": norm01(full_f[valid]),
        "nov": norm01(nov[valid]),
        "onset": norm01(onset_f[valid]),
    }
    vt = g_t[valid]

    def score_with(w):
        return sum(w[k] * comps[k] for k in comps)

    loud_v, rep_v, full_v = loud_f[valid], rep_f[valid], full_f[valid]
    gap_f = int(round(SECTION_GAP_S / STRUCT_HOP))
    need_f = int(round(SECTION_MIN_S / STRUCT_HOP))

    loud_max, rep_max = float(loud_v.max()), float(rep_v.max())

    # ---- section boundaries: three independent detectors ------------------
    # (a) rising edges into the top-loudness band, sustained >= 4 s
    thr = (np.percentile(lu_g[lu_g > -60], 95) - 1.5) if (lu_g > -60).any() else -70.0
    above = lu_g >= thr
    hold = int(round(4.0 / STRUCT_HOP))
    loud_edges = [float(g_t[i]) for i in range(n_g)
                  if above[i] and (i == 0 or not above[i - 1])
                  and above[i:i + hold].mean() > 0.85]
    # (b) local maxima of the arrangement-entry step
    step_edges = [float(g_t[i]) for i in range(1, n_g - 1)
                  if step_f[i] > 0.25 and step_f[i] >= step_f[i - 1] and step_f[i] > step_f[i + 1]]
    # (c) checkerboard-novelty peaks, prominence-picked
    pk, _ = find_peaks(nov, prominence=0.35 * float(np.std(nov)),
                       distance=max(1, int(round(4.0 / STRUCT_HOP))))
    nov_edges = [float(g_t[i]) for i in pk]
    all_edges = sorted(set(loud_edges) | set(step_edges) | set(nov_edges))

    def sections(tol, frac, f_full):
        q = ((loud_v >= loud_max - tol)
             & (rep_v >= frac * rep_max)
             & (full_v >= f_full))
        return runs_of(q, gap_f, need_f)

    BACK_S, FWD_S = 12.0, 6.0

    def choose(w, tol=Q_LOUD_TOL, frac=Q_REP_FRAC, f_full=Q_FULL):
        """Earliest section that IS the payoff and IS the repeated material,
        then moved onto the boundary that OPENS it.

        Taking the best-scoring point anywhere inside the qualifying run put one
        track 22 s past its own section boundary: the clip resumed instead of
        beginning, which is the one thing the operational definition forbids.
        So the candidates are the run's leading edge plus every detected section
        boundary from 12 s before it to 6 s after, and the weights arbitrate.
        """
        sc = score_with(w)
        for relax in (0.0, 1.0, 2.0, 3.0):
            runs = sections(tol + relax * 1.25, max(frac - relax * 0.10, 0.45),
                            max(f_full - relax * 0.10, 0.45))
            if not runs:
                continue
            a, _ = runs[0]
            ta = float(vt[a])
            cand_i = {a}
            for e in all_edges:
                if ta - BACK_S <= e <= ta + FWD_S:
                    k = int(np.argmin(np.abs(vt - e)))
                    if abs(vt[k] - e) <= STRUCT_HOP:
                        cand_i.add(k)
            best = max(cand_i, key=lambda k: sc[k])
            return best, relax, len(runs)
        return int(np.argmax(sc)), 99.0, 0

    base = score_with(WEIGHTS)
    i_star, relax_used, n_sections = choose(WEIGHTS)
    t_star = float(vt[i_star])
    t_argmax = float(vt[int(np.argmax(base))])

    # ---- stability: perturb EVERY free parameter at once ------------------
    # 300 draws. Weights x exp(N(0,0.30)) renormalised; the two qualifying
    # percentiles +/- N(0,5) points; the fullness floor +/- N(0,0.05). The
    # fraction landing within 4 s of the pick is how much of the answer is the
    # music and how much is my choice of constants.
    N_PERT = 300
    pert = np.empty(N_PERT)
    for n in range(N_PERT):
        w = {k: WEIGHTS[k] * float(np.exp(rng.normal(0, 0.30))) for k in WEIGHTS}
        tot = sum(w.values())
        w = {k: v / tot for k, v in w.items()}
        j, _, _ = choose(w,
                         float(np.clip(Q_LOUD_TOL + rng.normal(0, 0.5), 0.5, 3.5)),
                         float(np.clip(Q_REP_FRAC + rng.normal(0, 0.05), 0.65, 0.97)),
                         float(np.clip(Q_FULL + rng.normal(0, 0.05), 0.55, 0.95)))
        pert[n] = float(vt[j])
    stability = float(np.mean(np.abs(pert - t_star) <= 4.0))
    # A perturbation landing 8 s away is a different bar of the same chorus; one
    # landing 90 s away is a different section. Those are not the same error, and
    # a single pass/fail at 4 s cannot tell them apart — so report the interval.
    stability_sec = float(np.mean(np.abs(pert - t_star) <= 12.0))
    p10, p90 = float(np.percentile(pert, 10)), float(np.percentile(pert, 90))
    p50 = float(np.median(pert))
    # If the UNPERTURBED constants land outside the band their own neighbourhood
    # produces, the baseline is the outlier, not the answer. B15 exposed this:
    # 300 draws clustered inside three seconds and the shipped pick was 51 s away.
    baseline_outlier = bool(t_star < p10 - 2.0 or t_star > p90 + 2.0)

    # ---- how far is the pick from each independent boundary detector? -----
    #      Distance to the NEAREST such event, not to one globally-chosen event
    #      — a song has several chorus entries and picking the second one is not
    #      a disagreement. NOTE: these three detectors also supply the candidate
    #      set above, so "agreement" here means the three detectors agree with
    #      EACH OTHER at the chosen point, not that an untouched third party
    #      ratified it. See the limits section of PREVIEW_STARTS.md.
    if t_star < 3.0:
        nov_edges = nov_edges + [0.0]      # the head of the file IS a boundary

    def nearest(es):
        return float(min((abs(e - t_star) for e in es), default=99.0))

    d_loud, d_step, d_nov = nearest(loud_edges), nearest(step_edges), nearest(nov_edges)
    t_loud = min(loud_edges, key=lambda e: abs(e - t_star)) if loud_edges else None
    t_step = min(step_edges, key=lambda e: abs(e - t_star)) if step_edges else None

    # ---- how many times does THIS eight seconds actually come back? ------
    # The single most direct test of the operational definition: a stranger can
    # only reuse material the song itself reuses. Diagonal-matched similarity of
    # the picked window against every other position in the track.
    rec_times, rec_peak = [], 0.0
    if nb > 4:
        i0 = int(np.clip(round(t_star / STRUCT_HOP), 0, nb - 1))
        wlen = min(int(CLIP_S / STRUCT_HOP), nb - i0)
        if wlen >= 4:
            sim = np.full(nb, -1.0)
            for tp in range(0, nb - wlen):
                sim[tp] = float(np.mean(S[i0: i0 + wlen, tp: tp + wlen].diagonal()))
            spk, _ = find_peaks(sim, distance=int(MIN_REPEAT_LAG / STRUCT_HOP))
            for i in spk:
                tt = i * STRUCT_HOP
                if abs(tt - t_star) >= MIN_REPEAT_LAG and sim[i] >= 0.45:
                    rec_times.append(round(float(tt), 1))
            rec_peak = float(max((sim[i] for i in spk
                                  if abs(i * STRUCT_HOP - t_star) >= MIN_REPEAT_LAG), default=0.0))

    # ---- LEVEL evidence: is this actually the payoff? --------------------
    def pct_of(v):
        return float(np.mean(v <= v[i_star]))

    lvl_loud = pct_of(loud_f[valid])
    lvl_rep = pct_of(rep_f[valid])
    lvl_full = float(full_f[valid][i_star])
    level_pass = int(lvl_loud >= 0.85) + int(lvl_rep >= 0.85) + int(lvl_full >= 0.80)
    # agreement tolerance = one bar, floored at 3 s
    tol = max(3.0, 4 * B) if B else 3.0
    edge_pass = int(d_loud <= tol) + int(d_step <= tol) + int(d_nov <= tol)

    # ---- snap to a 4/4 downbeat -----------------------------------------
    snap = {"downbeat_s": None, "phase": None, "snap_dist_s": None, "bar_s": None}
    D = t_star
    if len(beats_t) > 8 and B:
        near = (beats_t > t_star - 30) & (beats_t < t_star + 30)
        idx = np.where(near)[0]
        if len(idx) >= 8:
            lowe = np.interp(beats_t, g_t, bnd_db[:, 0] + bnd_db[:, 1])
            ons = np.interp(beats_t, np.arange(len(onset)) / fps_a, onset)
            lowe = norm01(lowe)
            ons = norm01(ons)
            w = lowe + 0.6 * ons
            best_p, best_v = 0, -9
            for p in range(4):
                sel = idx[(idx % 4) == p]
                if len(sel) == 0:
                    continue
                v = float(np.mean(w[sel]))
                if v > best_v:
                    best_p, best_v = p, v
            db_times = beats_t[(np.arange(len(beats_t)) % 4) == best_p]
            if len(db_times):
                j = int(np.argmin(np.abs(db_times - t_star)))
                D = float(db_times[j])
                snap = {"downbeat_s": round(D, 3), "phase": best_p,
                        "snap_dist_s": round(abs(D - t_star), 3),
                        "bar_s": round(4 * B, 3)}

    # ---- the integer second ---------------------------------------------
    if B:
        cand = int(np.floor((D - B) + 0.5))
        if cand > D or (D - cand) > 1.5 or cand < 0:
            cand = int(np.floor(D))
    else:
        cand = int(np.floor(D))
    cand = max(0, min(cand, int(dur - 31)))
    runway = D - cand
    runway_beats = runway / B if B else None

    beat_ok = bool(beat_conf >= 0.45 and (snap["snap_dist_s"] is not None)
                   and snap["snap_dist_s"] <= (4 * B if B else 2.0))
    if (stability >= 0.80 and level_pass >= 2 and edge_pass >= 2
            and beat_ok and len(rec_times) >= 1 and not baseline_outlier):
        conf = "HIGH"
    elif ((stability >= 0.55 or stability_sec >= 0.85)
          and level_pass >= 2 and edge_pass >= 1 and not baseline_outlier):
        conf = "MEDIUM"
    else:
        conf = "LOW"

    reasons = []
    if baseline_outlier:
        reasons.append(f"BASELINE IS THE OUTLIER: 80% of perturbations land in "
                       f"{fmt(p10)}–{fmt(p90)}, median {fmt(p50)}, but the unperturbed "
                       f"constants chose {fmt(t_star)} — treat {fmt(p50)} as the rival")
    if stability_sec < 0.55:
        reasons.append(f"parameter-unstable: 80% of perturbations land in "
                       f"{fmt(p10)}–{fmt(p90)} — the constants, not the music, are "
                       f"choosing the section")
    elif stability < 0.55:
        reasons.append(f"same section but loose within it: 80% of perturbations land in "
                       f"{fmt(p10)}–{fmt(p90)}")
    elif stability < 0.80:
        reasons.append(f"parameter-sensitive ({stability:.2f} within 4 s; "
                       f"80% band {fmt(p10)}–{fmt(p90)})")
    if level_pass < 2:
        reasons.append(f"not clearly the payoff (loud pctl {lvl_loud:.2f}, "
                       f"repeat pctl {lvl_rep:.2f}, fullness {lvl_full:.2f})")
    if edge_pass == 0:
        reasons.append(f"no independent section edge within 3 s "
                       f"(loud {d_loud:.1f}s, step {d_step:.1f}s, novelty {d_nov:.1f}s)")
    elif edge_pass == 1:
        reasons.append(f"only one edge measure agrees "
                       f"(loud {d_loud:.1f}s, step {d_step:.1f}s, novelty {d_nov:.1f}s)")
    if len(rec_times) == 0:
        reasons.append(f"picked 8 s never recurs (best match elsewhere {rec_peak:.2f}) "
                       f"— not a loop point")
    if not beat_ok:
        reasons.append(f"beat grid weak (conf {beat_conf:.2f}, bpm {bpm and round(bpm)}, "
                       f"snap {snap['snap_dist_s']}s)")
    if runway_beats is not None and (runway_beats > 2.2 or runway_beats < 0.15):
        reasons.append(f"runway {runway:.2f}s = {runway_beats:.2f} beats")
    if abs(t_argmax - t_star) > 4.0:
        reasons.append(f"earliest-qualifying rule moved the pick from the score's best "
                       f"moment {fmt(t_argmax)} to {fmt(t_star)}")
    if relax_used > 0:
        reasons.append(f"no moment met the payoff conjunction — thresholds relaxed "
                       f"{relax_used:.0f} step(s)")
    if t_star < 4.0:
        reasons.append("pick is at the head of the file — equivalent to no offset")
    if t_star >= float(vt[-1]) - 2.0:
        reasons.append(f"pick sits against the {TAIL_GUARD:.0f}s tail guard — the score was "
                       f"still rising when the window closed")
    cp = {k: float(comps[k][i_star]) for k in comps}
    if cp["onset"] < 0.10 and cp["loud"] > 0.90:
        reasons.append("loud but rhythmically sparse here — a strong payoff with little for "
                       "an editor to cut to")
    if cp["rep"] < 0.60 and cp["loud"] > 0.90:
        reasons.append("loudest moment is not the most-repeated one — measures disagree")

    return {
        "file": path.name,
        "side": side,
        "duration_s": round(dur, 2),
        "bpm": round(bpm, 1) if bpm else None,
        "beat_conf": round(beat_conf, 3),
        "ibi_cv": round(ibi_cv, 4) if ibi_cv is not None else None,
        "tempo_salience": round(salience, 2),
        "score_peak_s": round(t_star, 2),
        "score_argmax_s": round(t_argmax, 2),
        "downbeat_s": snap["downbeat_s"],
        "bar_s": snap["bar_s"],
        "downbeat_phase": snap["phase"],
        "snap_dist_s": snap["snap_dist_s"],
        "start_s": int(cand),
        "start_mmss": fmt(cand),
        "runway_s": round(runway, 3),
        "runway_beats": round(runway_beats, 2) if runway_beats else None,
        "stability": round(stability, 3),
        "stability_within_12s": round(stability_sec, 3),
        "perturbation_p10_s": round(p10, 1),
        "perturbation_p50_s": round(p50, 1),
        "perturbation_p90_s": round(p90, 1),
        "perturbation_band_mmss": f"{fmt(p10)}-{fmt(p90)}",
        "baseline_is_outlier": baseline_outlier,
        "level": {"loud_pctl": round(lvl_loud, 3), "repeat_pctl": round(lvl_rep, 3),
                  "fullness": round(lvl_full, 3), "passed": level_pass},
        "edge": {"d_loudness_s": round(d_loud, 2), "d_step_s": round(d_step, 2),
                 "d_novelty_s": round(d_nov, 2), "tolerance_s": round(tol, 2),
                 "passed": edge_pass},
        "recurrences": len(rec_times),
        "recurrence_times_s": rec_times[:8],
        "recurrence_peak_sim": round(rec_peak, 3),
        "nearest_loud_edge_s": round(t_loud, 2) if t_loud is not None else None,
        "nearest_step_edge_s": round(t_step, 2) if t_step is not None else None,
        "qualifying_sections": n_sections,
        "threshold_relax": relax_used,
        "confidence": conf,
        "flags": reasons,
        "components_at_pick": {k: round(float(comps[k][i_star]), 3) for k in comps},
    }


def fmt(s) -> str:
    if s is None:
        return "-"
    s = int(round(float(s)))
    return f"{s // 60:02d}:{s % 60:02d}"


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="output/distribution/DELIVER")
    ap.add_argument("--out", default="output/distribution/preview_starts.json")
    ap.add_argument("--only", default=None, help="substring filter for a quick test")
    a = ap.parse_args()

    root = Path(a.root)
    canon = {}
    plan = root.parent / "ALBUM_PLAN_LOCKED.json"      # READ ONLY
    if plan.exists():
        pj = json.loads(plan.read_text(encoding="utf-8"))
        for s, key in (("A", "side_a"), ("B", "side_b")):
            for e in pj.get(key, []):
                canon[(s, e["position"])] = e["title"]
    rows = []
    for side in ("A", "B"):
        d = root / f"SIDE_{side}"
        for p in sorted(d.glob("*.wav")):
            if a.only and a.only.lower() not in p.name.lower():
                continue
            rng = np.random.default_rng(20260808)
            try:
                r = analyse(p, side, rng)
            except Exception as e:  # a failure must be visible, not silently absent
                r = {"file": p.name, "side": side, "error": f"{type(e).__name__}: {e}",
                     "confidence": "LOW", "flags": ["analysis failed"]}
            m = re.match(r"^([AB])(\d\d) - (.+)\.wav$", p.name)
            r["position"] = int(m.group(2)) if m else None
            # Titles come from the locked album plan where available — a filename
            # cannot carry a slash (B27 is "Press / Compress / Express" on disc).
            r["title"] = canon.get((side, r["position"]), m.group(3) if m else p.stem)
            rows.append(r)
            lv = r.get("level", {}).get("passed", "-")
            eg = r.get("edge", {})
            print(f"{p.name[:44]:44s} {r.get('start_mmss','--:--')}  "
                  f"{r.get('confidence','?'):6s} bpm={r.get('bpm')} "
                  f"stab={r.get('stability')}/{r.get('stability_within_12s')} "
                  f"band={r.get('perturbation_band_mmss')} "
                  f"lvl={lv}/3 edge={eg.get('passed','-')}/3", flush=True)

    out = {
        "generated_at": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc).isoformat(),
        "field": "DistroKid — Preview clip start time (TikTok, Apple Music, iTunes)",
        "note": "one integer-second value per track; Spotify is NOT controlled by this field",
        "weights": WEIGHTS,
        "qualifying": {"loud_tol_lu": Q_LOUD_TOL, "repeat_frac_of_max": Q_REP_FRAC,
                       "fullness": Q_FULL, "min_section_s": SECTION_MIN_S},
        "rule": "earliest section that is simultaneously top-loudness, top-repetition and "
                "full-arrangement; weights only choose the spot inside it; snapped to the "
                "nearest 4/4 downbeat; rounded to the integer second one beat earlier, "
                "never later than the downbeat",
        "clip_window_s": CLIP_S,
        "tail_guard_s": TAIL_GUARD,
        "tracks": rows,
    }
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwrote {a.out}  ({len(rows)} tracks)")
    return 0


if __name__ == "__main__":
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
