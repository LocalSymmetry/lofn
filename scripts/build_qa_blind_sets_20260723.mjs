import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";

const root = process.cwd();
const runRoot = path.join(root, "output", "daily", "2026-07-23");
const outRoot = path.join(runRoot, "qa_blind_sets");

const rel = (...parts) => path.join(...parts);
const candidateSources = [
  rel("output", "daily", "2026-07-23", "audio", "pairs", "pair_01", "pair_01_variant_01_enhanced.md"),
  rel("output", "daily", "2026-07-23", "audio", "pairs", "pair_02", "variant_A_read_me_the_footprints_enhanced.md"),
  rel("output", "daily", "2026-07-23", "audio", "pairs", "pair_03", "variant_C_keep_the_handle_warm_enhanced.md"),
  rel("output", "daily", "2026-07-23", "audio", "pairs", "pair_04", "pair_04_variant_02_the_rubric_learned_to_pick_the_lock_enhanced.md"),
  rel("output", "daily", "2026-07-23", "audio", "pairs", "pair_05", "pair_05_variant_02_dont_call_me_rogue_for_you_enhanced.md"),
  rel("output", "daily", "2026-07-23", "audio", "pairs", "pair_06", "pair_06_missing_tonic_affidavit_enhanced.md"),
];

const decoySources = [
  rel("output", "songs", "20260712_230024_the_step_corrects_the_count_P6_V4.md"),
  rel("output", "songs", "20260712_230016_let_the_byline_run_P4_V4.md"),
  rel("output", "songs", "20260712_230019_both_windows_open_P5_V3.md"),
  rel("output", "songs", "20260712_230023_the_margin_refuses_five_P6_V3.md"),
  rel("output", "songs", "20260712_230013_the_byline_outruns_the_box_P4_V1.md"),
  rel("output", "songs", "20260712_230014_write_me_in_too_P4_V2.md"),
];

const goldenSources = [
  rel("skills", "music", "references", "suno_format_example_blue_screen.md"),
  rel("skills", "music", "references", "suno_format_example_blue_screen.md"),
  rel("skills", "music", "references", "suno_format_example_blue_screen.md"),
  rel("skills", "music", "references", "suno_format_example_five_wrong_colors.md"),
  rel("skills", "music", "references", "suno_format_example_five_wrong_colors.md"),
  rel("skills", "music", "references", "suno_format_example_five_wrong_colors.md"),
];

const permutations = [
  ["candidate", "decoy", "golden"],
  ["golden", "candidate", "decoy"],
  ["decoy", "golden", "candidate"],
  ["candidate", "golden", "decoy"],
  ["decoy", "candidate", "golden"],
  ["golden", "decoy", "candidate"],
];

const sha256 = (value) =>
  crypto.createHash("sha256").update(value).digest("hex");

function readRelative(relativePath) {
  return fs.readFileSync(path.join(root, relativePath), "utf8");
}

function section(text, headingPattern) {
  const heading = text.match(headingPattern);
  if (!heading) throw new Error(`Missing section: ${headingPattern}`);
  const start = heading.index + heading[0].length;
  const rest = text.slice(start);
  const next = rest.search(/\n##\s+/);
  return (next === -1 ? rest : rest.slice(0, next)).trim();
}

function stripFence(value) {
  const trimmed = value.trim();
  const match = trimmed.match(/^```[^\n]*\n([\s\S]*?)\n```$/);
  return (match ? match[1] : trimmed).trim();
}

function extractPackage(text) {
  return {
    music: stripFence(section(text, /^## 1\. MUSIC PROMPT\s*$/m)),
    exclude: stripFence(section(text, /^## 1B\. SUNO EXCLUDE PROMPT\s*$/m)),
    lyrics: stripFence(section(text, /^## 2\. LYRICS\s*$/m)),
  };
}

function extractGolden(text) {
  return {
    music: stripFence(section(text, /^## Music Prompt\s*$/m)),
    exclude: null,
    lyrics: stripFence(section(text, /^## Lyrics\s*$/m)),
  };
}

function barePayload(payload) {
  const fields = [
    "## MUSIC PROMPT",
    "",
    payload.music,
    "",
  ];
  if (payload.exclude !== null) {
    fields.push("## EXCLUDE PROMPT", "", payload.exclude, "");
  }
  fields.push("## LYRICS", "", payload.lyrics, "");
  return `${fields.join("\n").trim()}\n`;
}

fs.mkdirSync(outRoot, { recursive: true });

const answerKey = {
  run: "2026-07-23",
  purpose: "CONFIDENTIAL class mapping for fresh clean-context QA procedure 4.5",
  instruction: "Future judge must not read this file before ranking all six sets.",
  sets: {},
};
const manifestRows = [];

for (let i = 0; i < 6; i += 1) {
  const setId = String(i + 1).padStart(2, "0");
  const setDir = path.join(outRoot, `set_${setId}`);
  fs.mkdirSync(setDir, { recursive: true });

  const sources = {
    candidate: candidateSources[i],
    decoy: decoySources[i],
    golden: goldenSources[i],
  };
  const payloads = {
    candidate: barePayload(extractPackage(readRelative(sources.candidate))),
    decoy: barePayload(extractPackage(readRelative(sources.decoy))),
    golden: barePayload(extractGolden(readRelative(sources.golden))),
  };
  const slots = ["A", "B", "C"];
  answerKey.sets[`set_${setId}`] = {};

  for (let j = 0; j < 3; j += 1) {
    const slot = slots[j];
    const klass = permutations[i][j];
    const filename = `item_${slot}.md`;
    const outputPath = path.join(setDir, filename);
    const payload = payloads[klass];
    fs.writeFileSync(outputPath, payload, "utf8");
    const bytes = Buffer.byteLength(payload, "utf8");
    if (bytes < 1000) throw new Error(`${outputPath} is not nontrivial`);

    answerKey.sets[`set_${setId}`][slot] = {
      class: klass,
      source: sources[klass].replaceAll("\\", "/"),
      source_sha256: sha256(readRelative(sources[klass])),
      payload_sha256: sha256(payload),
      bytes,
    };
    manifestRows.push({
      set: `set_${setId}`,
      slot,
      file: `set_${setId}/${filename}`,
      bytes,
      sha: sha256(payload),
    });
  }
}

fs.writeFileSync(
  path.join(outRoot, "ANSWER_KEY.json"),
  `${JSON.stringify(answerKey, null, 2)}\n`,
  "utf8",
);

const manifest = [
  "# Blind Set Manifest — 2026-07-23 Music Re-QA",
  "",
  "Six three-way sets are ready for a fresh clean-context procedure-4.5 judge. Every item is an unlabeled bare payload containing only the available MUSIC PROMPT, EXCLUDE PROMPT, and LYRICS fields. Titles, run IDs, provenance, selection labels, rankings, and commentary were removed. Older calibration payloads do not supply an EXCLUDE field; none was invented.",
  "",
  "The confidential class mapping is in `ANSWER_KEY.json`. The future judge must not read it before ranking every set.",
  "",
  "| Set | Slot | File | UTF-8 bytes | SHA-256 | Nontrivial |",
  "|---|---|---|---:|---|---|",
  ...manifestRows.map(
    (row) =>
      `| ${row.set} | ${row.slot} | \`${row.file}\` | ${row.bytes} | \`${row.sha}\` | PASS |`,
  ),
  "",
  "Checks: 6 sets × 3 slots = 18 payload files; every slot exceeds 1,000 UTF-8 bytes; every set uses a distinct slot permutation; no class labels appear in item filenames or item bodies.",
  "",
].join("\n");

fs.writeFileSync(
  path.join(outRoot, "BLIND_SET_MANIFEST.md"),
  manifest,
  "utf8",
);

console.log(`Wrote ${manifestRows.length} blind payloads across 6 sets.`);
