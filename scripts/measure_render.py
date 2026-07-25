#!/usr/bin/env python3
"""measure_render.py — what the RENDER actually did, versus what the prompt asked for.

Every gate in the Lofn pipeline reads TEXT. Until 2026-07-24 nothing in the
harness had ever measured a finished audio render, so an entire class of
failure was invisible by construction: a song can pass all 16 points of the
Suno gate and still arrive with its defining gesture smoothed away.

This is the numeric half of a render audit. It needs no network and no API key.
The perceptual half — a listening model — is described in
`.claude/skills/lofn-render-audit/SKILL.md`; the two check each other.

    pip install numpy soundfile        # libsndfile >= 1.1 decodes MP3
    python3 scripts/measure_render.py track.mp3 [more.mp3 ...]

Findings that motivated each measure (2026-07-24, three finished tracks):
  * a specified ~4-second full stop rendered as a 0.40 s / 5.7 dB dip
  * a specified quarter-tone drone pair produced zero isolated tonal components
  * two tracks rendered essentially mono despite hard-pan direction — but the
    third (the Staff Pick) did not, which is why `stereo` reports a spectrum
    rather than a boolean. See THE GRAIN LAW in the skill.
"""
import sys, os
import numpy as np
import soundfile as sf


def load(path):
    x, sr = sf.read(path, always_2d=True)
    return x, sr, x.mean(1), len(x) / sr


def envelope(mono, sr, hop=0.05):
    h = int(sr * hop)
    n = len(mono) // h
    env = np.array([np.sqrt(np.mean(mono[i*h:(i+1)*h] ** 2) + 1e-12) for i in range(n)])
    return env, 20 * np.log10(env / (env.max() + 1e-12) + 1e-12), hop


def gaps(db, hop, floor_db=-28, min_len=0.8):
    """Sustained quiet stretches — the hollow-centre test."""
    out, i, quiet = [], 0, db < floor_db
    while i < len(quiet):
        if quiet[i]:
            j = i
            while j < len(quiet) and quiet[j]:
                j += 1
            if (j - i) * hop >= min_len:
                out.append((i * hop, (j - i) * hop))
            i = j
        else:
            i += 1
    return out


def deepest_dip(db, hop, lo_frac=0.35, hi_frac=0.65, smooth_s=0.5):
    """A specified STOP that got smoothed shows up as a shallow, narrow dip.
    Reported as depth AND width — a 4 s stop cannot hide as 0.4 s."""
    k = max(1, int(smooth_s / hop))
    sm = np.convolve(db, np.ones(k) / k, mode="same")
    a, b = int(len(sm) * lo_frac), int(len(sm) * hi_frac)
    j = int(np.argmin(sm[a:b])) + a
    ref = np.median(sm[max(0, j - k*4): j + k*4])
    lvl = sm[j] + 3
    p = j
    while p > 0 and sm[p] <= lvl:
        p -= 1
    q = j
    while q < len(sm) - 1 and sm[q] <= lvl:
        q += 1
    return dict(at_s=j * hop, depth_db=ref - sm[j], width_s=(q - p) * hop)


def tonal_components(sig, sr, t0=0.0, t1=20.0, fmin=70, fmax=6000,
                     prominence_db=12, min_sep_cents=50):
    """Sustained, ISOLATED narrowband tones — drones, hums, whines.

    Median-across-frames so notes and transients wash out and only sustained
    material survives. Two guards, both learned the hard way: a local-prominence
    floor, and a minimum cents separation so adjacent FFT bins cannot be
    reported as a pair. Without the second guard this function 'found' six
    quarter-tone drone pairs that were one low-frequency cluster — at 60 Hz the
    neighbouring bin IS ~3% away, so the result was arithmetic, not evidence.
    """
    seg = sig[int(t0 * sr): int(t1 * sr)]
    N = sr
    frames = [np.abs(np.fft.rfft(seg[i:i+N] * np.hanning(N)))
              for i in range(0, max(1, len(seg) - N), N // 2)]
    if not frames:
        return []
    S = np.median(np.array(frames), axis=0)
    f = np.fft.rfftfreq(N, 1 / sr)
    lo, hi = np.searchsorted(f, fmin), np.searchsorted(f, fmax)
    S, f = S[lo:hi], f[lo:hi]
    thr = 10 ** (prominence_db / 20)
    found = []
    for i in range(30, len(S) - 30):
        nb = np.median(np.concatenate([S[i-30:i-5], S[i+5:i+30]]))
        if S[i] > nb * thr and S[i] == S[max(0, i-3):i+4].max():
            found.append((S[i] / nb, f[i]))
    found.sort(reverse=True)
    keep = []
    for pr, fr in found:
        if all(abs(1200 * np.log2(fr / k[1])) > min_sep_cents for k in keep):
            keep.append((pr, fr))
    return keep[:8]


def interval_pairs(freqs, ratio, tol=0.004):
    """Do any two tones sit at a target interval? ratio 2**(1/24) = quarter tone."""
    out = []
    for i in range(len(freqs)):
        for j in range(i + 1, len(freqs)):
            r = freqs[j] / freqs[i]
            if abs(r - ratio) < tol:
                out.append((freqs[i], freqs[j], r))
    return out


def stereo(L, R):
    M, S = (L + R) / 2, (L - R) / 2
    return dict(
        correlation=float(np.corrcoef(L, R)[0, 1]),
        side_mid=float(np.sqrt(np.mean(S ** 2)) / (np.sqrt(np.mean(M ** 2)) + 1e-12)),
    )


def width_vs_level(L, R, sr, win=1.0):
    """Does it widen where it gets loud? The spec usually says the chorus opens
    up. A NEGATIVE correlation means it narrows into the climax instead —
    which on the house benchmark turned out to be better than what was asked
    for. Report it; never auto-fail it."""
    h = int(sr * win)
    n = min(len(L), len(R)) // h
    w, lv = [], []
    for i in range(n):
        l, r = L[i*h:(i+1)*h], R[i*h:(i+1)*h]
        m, s = (l + r) / 2, (l - r) / 2
        mm = np.sqrt(np.mean(m ** 2))
        w.append(np.sqrt(np.mean(s ** 2)) / (mm + 1e-9))
        lv.append(20 * np.log10(mm + 1e-12))
    w, lv = np.array(w), np.array(lv)
    live = lv > (lv.max() - 30)
    return dict(width_min=float(w.min()), width_max=float(w.max()),
                corr_width_level=float(np.corrcoef(w[live], lv[live])[0, 1]))


def band_profile(mono, sr, bands=((20, 80), (80, 300), (300, 2000), (8000, 16000))):
    """Energy per band per quarter — tests 'more sub and more sky at the end'."""
    N = 1 << 15
    out = {}
    for f1, f2 in bands:
        vals = []
        for i in range(0, len(mono) - N, N):
            F = np.abs(np.fft.rfft(mono[i:i+N] * np.hanning(N)))
            f = np.fft.rfftfreq(N, 1 / sr)
            vals.append(np.sqrt(np.mean(F[(f >= f1) & (f < f2)] ** 2)))
        v = np.array(vals)
        q = max(1, len(v) // 4)
        base = v[:q].mean() + 1e-12
        out[f"{f1}-{f2}Hz"] = [round(float(20 * np.log10(v[i*q:(i+1)*q].mean() / base + 1e-12)), 1)
                               for i in range(4)]
    return out


def tempo_candidates(env, hop):
    """NOTE: needs a FINE hop. The first version ran on the 50 ms envelope and
    reported ~120 BPM for a track measured at 109 by a 10 ms hop — at 110 BPM a
    beat is 545 ms, i.e. 11 frames, so beat-level resolution was far too coarse
    and the peak landed on a neighbouring lag. That wrong number was reported as
    a finding about the renderer ("exact BPM drifts") when it was an artifact of
    my own analysis. Call with hop <= 0.01."""
    d = np.diff(env)
    d[d < 0] = 0
    if d.std() == 0:
        return []
    ac = np.correlate(d - d.mean(), d - d.mean(), "full")[len(d) - 1:]
    lags = np.arange(1, len(ac)) * hop
    cand = [(ac[i], 60 / lags[i - 1]) for i in range(1, len(ac)) if 0.3 < lags[i - 1] < 1.2]
    cand.sort(reverse=True)
    return [round(b) for _, b in cand[:4]]


def audit(path):
    x, sr, mono, dur = load(path)
    env, db, hop = envelope(mono, sr)
    fine_env, _, fine_hop = envelope(mono, sr, hop=0.01)   # tempo needs 10 ms
    r = {
        "file": os.path.basename(path),
        "duration_s": round(dur, 1),
        "sample_rate": sr,
        "peak": round(float(np.max(np.abs(mono))), 3),
        "crest_db": round(float(20 * np.log10(np.max(np.abs(mono)) / (np.sqrt(np.mean(mono**2)) + 1e-12))), 1),
        "loudness_spread_db": round(float(np.percentile(db[db > -40], 90) - np.percentile(db[db > -40], 10)), 1),
        "opening_2s_db": round(float(db[:int(2/hop)].mean()), 1),
        "quiet_gaps": [(round(t, 2), round(l, 2)) for t, l in gaps(db, hop)],
        "deepest_mid_dip": {k: round(v, 2) for k, v in deepest_dip(db, hop).items()},
        "tempo_candidates_bpm": tempo_candidates(fine_env, fine_hop),
        "bands_by_quarter_db": band_profile(mono, sr),
    }
    if x.shape[1] == 2:
        L, R = x[:, 0], x[:, 1]
        r["stereo"] = {k: round(v, 3) for k, v in stereo(L, R).items()}
        r["width_dynamics"] = {k: round(v, 3) for k, v in width_vs_level(L, R, sr).items()}
        tc = tonal_components(R, sr)
    else:
        tc = tonal_components(mono, sr)
    r["sustained_tones_hz"] = [round(f, 1) for _, f in tc]
    r["quarter_tone_pairs"] = interval_pairs([f for _, f in tc], 2 ** (1 / 24))
    return r


if __name__ == "__main__":
    import json
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    print(json.dumps([audit(p) for p in sys.argv[1:]], indent=2))
