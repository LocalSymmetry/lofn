#!/usr/bin/env python3
"""Rebuild RUN_STATE.md/.json for a Lofn daily music run BY STAT-ING DISK.

Disk is the only authority. This manifest is a CACHE the coordinator rebuilds
after every wave -- it must never assert an artifact that is not on disk.
(`lofn-daily` SKILL, "Run-state manifest & resume"; EXECUTION.md section 6.)

Usage: python3 scripts/build_run_state.py <run-dir> [--icb-sha SHA]
"""
import sys
import os
import hashlib
import json
import datetime

THIN_BYTES = 500  # an artifact smaller than this is a stub, not a deliverable

COORD_STEPS = [
    ('00', 'step00_aesthetics_and_genres.md'),
    ('01', 'step01_essence_and_facets.md'),
    ('02', 'step02_concepts.md'),
    ('03', 'step03_artist_and_critique.md'),
    ('04', 'step04_medium.md'),
    ('05', 'step05_refine_medium.md'),
]
PAIR_STEPS = [
    ('06', 'facets'),
    ('07', 'song_guides'),
    ('08', 'generation'),
    ('09', 'artist_refined'),
    ('10', 'revision_synthesis'),
]


def stat_row(step, pair, path):
    exists = os.path.exists(path)
    size = os.path.getsize(path) if exists else 0
    sha = ''
    if exists and size:
        sha = hashlib.sha256(open(path, 'rb').read()).hexdigest()[:10]
    if not exists:
        status = 'pending'
    elif size < THIN_BYTES:
        status = 'thin'
    else:
        status = 'done'
    return {
        'step': step, 'pair': pair, 'modality': 'music',
        'canonical_path': path.replace(os.sep, '/'),
        'exists': exists, 'byte_size': size, 'sha': sha,
        'gate_verdict': '', 'attempt_count': 1, 'status': status,
    }


def main():
    run = sys.argv[1].rstrip('/').rstrip('\\')
    icb_sha = None
    if '--icb-sha' in sys.argv:
        icb_sha = sys.argv[sys.argv.index('--icb-sha') + 1]

    music = os.path.join(run, 'music')
    icb = os.path.join(music, 'CREATIVE_CONTEXT.md')
    if os.path.exists(icb) and not icb_sha:
        icb_sha = hashlib.sha256(open(icb, 'rb').read()).hexdigest()[:12]

    rows = []
    for step, fname in COORD_STEPS:
        rows.append(stat_row(step, '-', os.path.join(music, fname)))
    for n in range(1, 7):
        pid = 'P%02d' % n
        rows.append(stat_row('05b', pid, os.path.join(
            music, 'p%02d_step05_isolated_brief.md' % n)))
        for step, name in PAIR_STEPS:
            rows.append(stat_row(step, pid, os.path.join(
                music, 'pair_%02d_step%s_%s.md' % (n, step, name))))

    done = sum(1 for r in rows if r['status'] == 'done')
    pending = sum(1 for r in rows if r['status'] == 'pending')
    thin = sum(1 for r in rows if r['status'] == 'thin')
    stamp = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')

    out = [
        '# RUN_STATE - %s' % os.path.basename(run),
        '**Rebuilt from disk:** %s' % stamp,
        '**ICB sha:** `%s` - **Modality:** music - **Intended cardinality:** 6 pairs x 4 = 24 songs' % icb_sha,
        '**Artifacts:** %d done - %d pending - %d thin (<%dB)' % (done, pending, thin, THIN_BYTES),
        '',
        'This manifest is a CACHE rebuilt by stat-ing files. If it disagrees with disk, DISK WINS.',
        '',
        '| step | pair | status | bytes | sha | file |',
        '|---|---|---|---|---|---|',
    ]
    for r in rows:
        out.append('| %s | %s | %s | %d | %s | %s |' % (
            r['step'], r['pair'], r['status'], r['byte_size'], r['sha'],
            os.path.basename(r['canonical_path'])))

    with open(os.path.join(run, 'RUN_STATE.md'), 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(out) + '\n')
    with open(os.path.join(run, 'RUN_STATE.json'), 'w', encoding='utf-8') as fh:
        json.dump({'icb_sha': icb_sha, 'rebuilt_at': stamp, 'rows': rows},
                  fh, indent=1)

    print('done=%d pending=%d thin=%d' % (done, pending, thin))
    for r in rows:
        if r['status'] != 'done':
            print('  %-8s %3s %s %s' % (r['status'], r['step'], r['pair'],
                                        os.path.basename(r['canonical_path'])))


if __name__ == '__main__':
    main()
