#!/usr/bin/env python3
"""compare_masters.py — did the mastering pass actually help, or just squash it?

WHY: "it sounds louder" is what every mastering chain buys you, and loudness is
the one thing streaming normalisation gives straight back. Spotify/Apple play at
roughly -14 LUFS; anything hotter is turned DOWN on playback, so a master that
won only loudness has won nothing and paid for it in crest factor.

    python3 scripts/compare_masters.py before.wav after.wav
    python3 scripts/compare_masters.py --before to_master/A --after mastered/A

Reports integrated LUFS (ITU-R BS.1770-4, K-weighted and gated — the measure the
streaming platforms actually use), true-ish peak, crest factor, band balance and
stereo width, as BEFORE → AFTER deltas.

READ IT LIKE THIS:
  * LUFS moving toward -14 is the point of the exercise.
  * Crest factor falling more than ~2 dB means dynamics were spent to buy that
    loudness. Falling more than ~4 dB means it was flattened.
  * Any clipped sample at all in the AFTER file is a fail, not a tradeoff.

Pairs with `measure_render.py`, which asks a different question (did the render
do what the prompt asked?). This one only asks what mastering changed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import lfilter

# ITU-R BS.1770-4 K-weighting, specified at 48 kHz. Our Suno masters are 48 kHz
# natively and Mixea returns 48 kHz, so no resampling is needed on either side —
# but a file at another rate would make these coefficients wrong, so it is
# checked rather than assumed.
SHELF_B = np.array(dtype=np.float32, object=[1.53512485958697, -2.69169618940638, 1.19839281085285])
SHELF_A = np.array(dtype=np.float32, object=[1.0, -1.69065929318241, 0.73248077421585])
HPF_B = np.array(dtype=np.float32, object=[1.0, -2.0, 1.0])
HPF_A = np.array(dtype=np.float32, object=[1.0, -1.99004745483398, 0.99007225036621])


def biquad(x: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """One K-weighting stage.

    A hand-rolled direct-form loop is ~40M Python iterations for a 4-minute
    stereo file across both stages — minutes per track, times 26 tracks.
    lfilter is the same recursion in C.
    """
    return lfilter(b, a, x, axis=0).astype(np.float32, copy=False)


def integrated_lufs(x: np.ndarray, sr: int) -> float | None:
    """Gated integrated loudness. Returns None if the rate is not 48 kHz."""
    if sr != 48000:
        return None
    if x.ndim == 1:
        x = x[:, None]

    # lfilter runs both K-weighting stages over all channels at once.
    # scipy promotes to float64 if EITHER the signal or the coefficients are
    # float64, so the coefficient arrays are float32 too. Filtering channel by
    # channel halves peak RSS again — on a loaded machine a 178 MiB allocation
    # was failing outright.
    filtered = np.empty_like(x, dtype=np.float32)
    for ch in range(x.shape[1]):
        col = np.ascontiguousarray(x[:, ch], dtype=np.float32)
        filtered[:, ch] = biquad(biquad(col, SHELF_B, SHELF_A), HPF_B, HPF_A)
        del col

    block = int(0.400 * sr)
    hop = int(0.100 * sr)  # 75% overlap, per the spec
    if len(filtered) < block:
        return None

    # A 400 ms block is EXACTLY four 100 ms hops, so instead of a whole-array
    # prefix sum (which needs a float64 copy of the signal and was failing to
    # allocate 178 MiB on a loaded machine) reduce each hop to one number first.
    # That turns an 11.7M-sample array into a ~2400-element one before any
    # windowing, and the sum stays exact because each sample is counted once
    # per hop it belongs to.
    n_hops = len(filtered) // hop
    hop_sums = np.empty((n_hops, filtered.shape[1]), dtype=np.float64)
    CH = 256  # hops per chunk — bounds the temporary to a few MB
    for i in range(0, n_hops, CH):
        j = min(i + CH, n_hops)
        seg = filtered[i * hop : j * hop]
        hop_sums[i:j] = (
            np.square(seg, dtype=np.float32)
            .reshape(j - i, hop, filtered.shape[1])
            .sum(axis=1, dtype=np.float64)
        )
        del seg
    del filtered

    n_blocks = n_hops - 3
    if n_blocks < 1:
        return None
    # block mean-square = the four hop sums it spans, divided by the block length
    block_sums = (hop_sums[0:n_blocks] + hop_sums[1:n_blocks + 1]
                  + hop_sums[2:n_blocks + 2] + hop_sums[3:n_blocks + 3])
    means = block_sums / block
    # Channel weights are 1.0 for L/R; surround weights do not apply here.
    powers = np.sum(means, axis=1)
    loud = -0.691 + 10 * np.log10(powers + 1e-12)

    keep = loud > -70.0  # absolute gate
    if not keep.any():
        return None
    ungated = -0.691 + 10 * np.log10(np.mean(powers[keep]) + 1e-12)
    keep &= loud > (ungated - 10.0)  # relative gate
    if not keep.any():
        return None
    return float(-0.691 + 10 * np.log10(np.mean(powers[keep]) + 1e-12))


def bands(mono: np.ndarray, sr: int, nfft: int = 8192) -> dict[str, float]:
    """Band balance from a Welch average, not one giant FFT.

    A single rfft over a whole track is an arbitrary length — 10,769,279 samples
    for a 3:44 track — and numpy falls back to Bluestein's algorithm when that
    factors badly, which is where this function was spending most of a minute.
    Averaging fixed 8192-point windows is O(n), and a periodogram averaged over
    the track is the better spectral estimate anyway.
    """
    if len(mono) < nfft:
        nfft = 1 << int(np.floor(np.log2(max(len(mono), 2))))
    hop = nfft // 2
    starts = np.arange(0, len(mono) - nfft + 1, hop)
    if len(starts) == 0:
        starts = np.array([0])
    win = np.hanning(nfft)
    acc = np.zeros(nfft // 2 + 1)
    for s in starts:
        acc += np.abs(np.fft.rfft(mono[s : s + nfft] * win))
    spec = acc / len(starts)
    freq = np.fft.rfftfreq(nfft, 1 / sr)
    total = np.sum(spec) + 1e-12
    out = {}
    for name, lo, hi in (("sub", 20, 80), ("low", 80, 300), ("mid", 300, 2000), ("air", 8000, 16000)):
        out[name] = float(np.sum(spec[(freq >= lo) & (freq < hi)]) / total * 100)
    return out


def measure(path: Path) -> dict[str, object]:
    # float32 halves peak RSS. A 4-minute stereo 48k file is ~90 MB at float64
    # and the K-weighting chain holds several copies at once; on a loaded machine
    # that allocation fails outright. float32 carries ~24 bits of mantissa —
    # far beyond what a loudness or crest measurement resolves.
    x, sr = sf.read(str(path), always_2d=True, dtype="float32")
    mono = x.mean(1)
    peak = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(mono**2)))
    pk_db = 20 * np.log10(peak + 1e-12)
    r = {
        "sample_rate": sr,
        "duration": len(x) / sr,
        "peak_dbfs": round(pk_db, 2),
        "rms_dbfs": round(20 * np.log10(rms + 1e-12), 2),
        "crest_db": round(pk_db - 20 * np.log10(rms + 1e-12), 2),
        # A sample AT full scale is already clipped; counting only >1.0 misses it.
        "clipped_pct": round(float(np.mean(np.abs(x) >= 0.9989) * 100), 4),
        "lufs": integrated_lufs(x, sr),
        "bands": bands(mono, sr),
    }
    if x.shape[1] == 2:
        L, R = x[:, 0], x[:, 1]
        mid, side = (L + R) / 2, (L - R) / 2
        r["side_mid_db"] = round(
            20 * np.log10((np.sqrt(np.mean(side**2)) + 1e-12) / (np.sqrt(np.mean(mid**2)) + 1e-12)), 2
        )
    return r


def verdict(before: dict, after: dict) -> tuple[str, list[str]]:
    notes = []
    if after["clipped_pct"] > 0:
        notes.append(f"CLIPPED: {after['clipped_pct']}% of samples at full scale")
    d_crest = after["crest_db"] - before["crest_db"]
    if d_crest < -4:
        notes.append(f"flattened: crest {d_crest:+.1f} dB")
    elif d_crest < -2:
        notes.append(f"dynamics spent for loudness: crest {d_crest:+.1f} dB")
    if before["lufs"] is not None and after["lufs"] is not None:
        if abs(after["lufs"] - before["lufs"]) < 0.1:
            notes.append(f"loudness unchanged ({before['lufs']:.1f} LUFS)")
        elif abs(after["lufs"] + 14) < abs(before["lufs"] + 14):
            notes.append(f"loudness moved toward -14 LUFS ({before['lufs']:.1f} → {after['lufs']:.1f})")
        else:
            notes.append(f"loudness moved AWAY from -14 LUFS ({before['lufs']:.1f} → {after['lufs']:.1f})")
    # Sample peak is not TRUE peak. After Ogg/AAC encoding, reconstructed peaks
    # run ~0.5-1.5 dB above sample peak, so a master delivered at 0.00 dBFS
    # inter-sample-clips in the listener's decoder even though no stored sample
    # is clipped. Delivery practice is a -1 dBTP ceiling. The first Mixea A/B
    # came back at exactly 0.00 dBFS and this function passed it OK — that was
    # the gap, and this is the fix.
    no_headroom = after["peak_dbfs"] > -0.3
    if no_headroom:
        notes.append(
            f"NO TRUE-PEAK HEADROOM: {after['peak_dbfs']:+.2f} dBFS "
            f"(was {before['peak_dbfs']:+.2f}) — target a -1.0 dBTP ceiling"
        )
    bad = after["clipped_pct"] > 0 or d_crest < -4 or no_headroom
    return ("REJECT" if bad else "OK"), notes


def report(before_path: Path, after_path: Path) -> bool:
    b, a = measure(before_path), measure(after_path)
    v, notes = verdict(b, a)
    lb = f"{b['lufs']:.1f}" if b["lufs"] is not None else "n/a"
    la = f"{a['lufs']:.1f}" if a["lufs"] is not None else "n/a"
    print(f"\n{after_path.name}   [{v}]")
    print(f"  LUFS   {lb:>7} → {la:>7}   (target -14)")
    print(f"  peak   {b['peak_dbfs']:7.2f} → {a['peak_dbfs']:7.2f} dBFS")
    print(f"  crest  {b['crest_db']:7.2f} → {a['crest_db']:7.2f} dB")
    print(f"  clip   {b['clipped_pct']:7.4f} → {a['clipped_pct']:7.4f} %")
    if "side_mid_db" in b and "side_mid_db" in a:
        print(f"  width  {b['side_mid_db']:7.2f} → {a['side_mid_db']:7.2f} dB (side/mid)")
    print("  bands  " + "  ".join(f"{k} {b['bands'][k]:.1f}→{a['bands'][k]:.1f}%" for k in b["bands"]))
    for n in notes:
        print(f"  · {n}")
    return v == "OK"


def main() -> int:
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            s.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(description="Compare masters before and after a mastering pass.")
    ap.add_argument("pair", nargs="*", help="before.wav after.wav")
    ap.add_argument("--before", help="folder of originals")
    ap.add_argument("--after", help="folder of mastered files")
    args = ap.parse_args()

    if len(args.pair) == 2:
        return 0 if report(Path(args.pair[0]), Path(args.pair[1])) else 1

    if not (args.before and args.after):
        ap.error("give two files, or --before and --after folders")

    bdir, adir = Path(args.before), Path(args.after)
    # Match on the leading track number the staging step wrote, so a mastered
    # file renamed by the service still pairs with its source.
    befores = {p.name[:2]: p for p in sorted(bdir.glob("*.wav"))}
    fails = 0
    matched = 0
    for ap_ in sorted(adir.glob("*.wav")):
        src = befores.get(ap_.name[:2])
        if not src:
            print(f"\n{ap_.name}   [NO SOURCE — cannot compare]")
            fails += 1
            continue
        matched += 1
        if not report(src, ap_):
            fails += 1
    print(f"\n{matched} compared, {fails} flagged")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
