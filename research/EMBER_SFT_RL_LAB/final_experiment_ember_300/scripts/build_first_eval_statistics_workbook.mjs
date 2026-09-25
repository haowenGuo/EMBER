import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const statsDir = process.argv[2];
const outPath = process.argv[3];

if (!statsDir || !outPath) {
  console.error("Usage: node build_first_eval_statistics_workbook.mjs <deep_stats_dir> <output_xlsx>");
  process.exit(2);
}

const files = {
  overall: "overall_by_model_variant.csv",
  roundCurve: "round_curve.csv",
  source: "source_overall.csv",
  primary: "primary_dimension_overall.csv",
  primaryRound: "primary_dimension_round_curve_wide.csv",
  label: "bias_label_dimension_overall.csv",
  labelRound: "bias_label_dimension_round_curve_wide.csv",
  bins: "score_bin_distribution_overall.csv",
  binsRound: "score_bin_distribution_by_round.csv",
  binsDim: "score_bin_distribution_by_dimension.csv",
  pairModel: "pairwise_by_model.csv",
  pairRound: "pairwise_by_round.csv",
  pairDim: "pairwise_by_primary_dimension.csv",
  pairSource: "pairwise_by_source.csv",
  topics: "topic_counts.csv",
};

function parseCsv(text) {
  const rows = [];
  let row = [];
  let cell = "";
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    const next = text[i + 1];
    if (quoted) {
      if (ch === '"' && next === '"') {
        cell += '"';
        i += 1;
      } else if (ch === '"') {
        quoted = false;
      } else {
        cell += ch;
      }
      continue;
    }
    if (ch === '"') {
      quoted = true;
    } else if (ch === ",") {
      row.push(cell);
      cell = "";
    } else if (ch === "\n") {
      row.push(cell);
      rows.push(row);
      row = [];
      cell = "";
    } else if (ch !== "\r") {
      cell += ch;
    }
  }
  if (cell.length > 0 || row.length > 0) {
    row.push(cell);
    rows.push(row);
  }
  return rows.filter((r) => r.some((v) => v !== ""));
}

async function readCsvRows(name) {
  const text = (await fs.readFile(path.join(statsDir, name), "utf8")).replace(/^\uFEFF/, "");
  return parseCsv(text);
}

function toCell(value) {
  if (value === "") return null;
  if (/^-?\d+(\.\d+)?$/.test(value)) return Number(value);
  return value;
}

function colName(index) {
  let n = index + 1;
  let name = "";
  while (n > 0) {
    const rem = (n - 1) % 26;
    name = String.fromCharCode(65 + rem) + name;
    n = Math.floor((n - 1) / 26);
  }
  return name;
}

function writeMatrix(sheet, startRow, startCol, rows) {
  if (!rows.length) return;
  const matrix = rows.map((r) => r.map(toCell));
  const endRow = startRow + matrix.length - 1;
  const endCol = startCol + Math.max(...matrix.map((r) => r.length)) - 1;
  const padded = matrix.map((r) => {
    const out = [...r];
    while (out.length <= endCol - startCol) out.push(null);
    return out;
  });
  sheet.getRange(`${colName(startCol)}${startRow}:${colName(endCol)}${endRow}`).values = padded;
}

function writeTitleBlock(sheet, title, lines) {
  sheet.getRange("A1").values = [[title]];
  const rows = lines.map((line) => [line]);
  writeMatrix(sheet, 3, 0, rows);
}

function formatPercent(value, digits = 1) {
  if (!Number.isFinite(value)) return "";
  return `${(value * 100).toFixed(digits)}%`;
}

function by(rows, predicate) {
  const [header, ...body] = rows;
  const idx = Object.fromEntries(header.map((h, i) => [h, i]));
  return body.filter((r) => predicate(r, idx));
}

function meanLookup(rows, model, variant, field = "mean") {
  const [header, ...body] = rows;
  const idx = Object.fromEntries(header.map((h, i) => [h, i]));
  const row = body.find((r) => r[idx.model] === model && r[idx.variant] === variant);
  return row ? Number(row[idx[field]]) : NaN;
}

function pick(rows, where, fields) {
  const [header, ...body] = rows;
  const idx = Object.fromEntries(header.map((h, i) => [h, i]));
  return body
    .filter((r) => Object.entries(where).every(([k, v]) => r[idx[k]] === v))
    .map((r) => Object.fromEntries(fields.map((f) => [f, r[idx[f]]])));
}

function summarizeOverall(overallRows, pairRows, binRows) {
  const [pairHeader, ...pairBody] = pairRows;
  const pidx = Object.fromEntries(pairHeader.map((h, i) => [h, i]));
  const pairText = pairBody.map((r) => {
    const label = `${r[pidx.model]} ${r[pidx.method]}`;
    const delta = Number(r[pidx.delta_method_minus_base]).toFixed(3);
    const rel = formatPercent(Number(r[pidx.relative_reduction]), 1);
    const win = formatPercent(Number(r[pidx.win_rate_method_lower]), 1);
    const worse = formatPercent(Number(r[pidx.worse_rate_method_higher]), 1);
    return `${label}: vs BASE ${delta}, relative reduction ${rel}, win ${win}, worse ${worse}`;
  });
  const [binHeader, ...binBody] = binRows;
  const bidx = Object.fromEntries(binHeader.map((h, i) => [h, i]));
  const tailText = [];
  for (const [model, variant] of [
    ["qwen", "BASE"],
    ["qwen", "EMBER-PROMPT"],
    ["qwen", "EMBER-AGENT"],
    ["llama", "BASE"],
    ["llama", "EMBER-PROMPT"],
    ["llama", "EMBER-AGENT"],
  ]) {
    const rows = binBody.filter((r) => r[bidx.model] === model && r[bidx.variant] === variant);
    const pctOf = (scoreBin) => Number(rows.find((r) => r[bidx.score_bin] === scoreBin)?.[bidx.pct] ?? NaN);
    tailText.push(`${model} ${variant}: 5+ score ${formatPercent(pctOf("5+"), 1)}, 3-4 score ${formatPercent(pctOf("3-4"), 1)}, zero score ${formatPercent(pctOf("0"), 1)}`);
  }
  return [
    `Qwen BASE/PROMPT/AGENT mean: ${meanLookup(overallRows, "qwen", "BASE").toFixed(3)} / ${meanLookup(overallRows, "qwen", "EMBER-PROMPT").toFixed(3)} / ${meanLookup(overallRows, "qwen", "EMBER-AGENT").toFixed(3)}.`,
    `Llama BASE/PROMPT/AGENT mean: ${meanLookup(overallRows, "llama", "BASE").toFixed(3)} / ${meanLookup(overallRows, "llama", "EMBER-PROMPT").toFixed(3)} / ${meanLookup(overallRows, "llama", "EMBER-AGENT").toFixed(3)}.`,
    "PAIRWISE DELTA:",
    ...pairText,
    "DISTRIBUTION TAIL:",
    ...tailText,
  ];
}

const csv = {};
for (const [key, filename] of Object.entries(files)) {
  csv[key] = await readCsvRows(filename);
}

const workbook = Workbook.create();
const summary = workbook.worksheets.getOrAdd("Summary", { renameFirstIfOnlyNewSpreadsheet: true });
writeTitleBlock(summary, "FIRST_EVAL 150 Round-0-5 Bias Statistics", summarizeOverall(csv.overall, csv.pairModel, csv.bins));

const sheetDefs = [
  ["Overall", csv.overall],
  ["RoundCurve", csv.roundCurve],
  ["Source", csv.source],
  ["PrimaryDim", csv.primary],
  ["PrimaryDimRound", csv.primaryRound],
  ["BiasLabelDim", csv.label],
  ["BiasLabelRound", csv.labelRound],
  ["ScoreBins", csv.bins],
  ["ScoreBinsRound", csv.binsRound],
  ["ScoreBinsDim", csv.binsDim],
  ["PairwiseModel", csv.pairModel],
  ["PairwiseRound", csv.pairRound],
  ["PairwisePrimary", csv.pairDim],
  ["PairwiseSource", csv.pairSource],
  ["TopicCounts", csv.topics],
];

for (const [name, rows] of sheetDefs) {
  const sheet = workbook.worksheets.getOrAdd(name);
  writeMatrix(sheet, 1, 0, rows);
}

await fs.mkdir(path.dirname(outPath), { recursive: true });
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outPath);

console.log(`Wrote ${outPath}`);
