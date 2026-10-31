// Starts a new lab folder from the newest one.
//   node tools/new-lab.js <id> "<Lab title>" "<Deck file.pdf>" [more decks...]
//   node tools/new-lab.js cnn "CNN Lab" "CNN.pdf"
// Copies common.jsx (shared UI kit + math helpers) from the most recently built lab and writes a
// lab.json skeleton. Lessons are added to lab.json as their .jsx files are written.
const fs = require("fs");
const path = require("path");

const ROOT = path.join(__dirname, "..");
const [id, title, ...decks] = process.argv.slice(2);
if (!id || !title || !decks.length || !/^[a-z0-9-]+$/.test(id)) {
  console.error('usage: node tools/new-lab.js <id: lowercase-with-dashes> "<Lab title>" "<Deck.pdf>" [...]');
  process.exit(1);
}
const dir = path.join(ROOT, "labs", id);
if (fs.existsSync(dir)) {
  console.error("labs/" + id + " already exists");
  process.exit(1);
}
const course = JSON.parse(fs.readFileSync(path.join(ROOT, "course.json"), "utf8"));
for (const d of decks) if (!course.decks.some((x) => x.file === d)) console.warn("warning: " + d + " is not in course.json; add it there so the course map shows it");

const labs = fs
  .readdirSync(path.join(ROOT, "labs"))
  .map((d) => path.join(ROOT, "labs", d, "lab.json"))
  .filter((f) => fs.existsSync(f))
  .map((f) => JSON.parse(fs.readFileSync(f, "utf8")))
  .sort((a, b) => (a.built < b.built ? 1 : -1));
const base = labs[0];

fs.mkdirSync(dir, { recursive: true });
fs.copyFileSync(path.join(ROOT, "labs", base.id, "common.jsx"), path.join(dir, "common.jsx"));
const today = new Date().toISOString().slice(0, 10);
const lab = {
  id,
  title,
  decks,
  sub: "Deep Learning · " + decks.map((d) => d.replace(/\.pdf$/, "")).join(" & ") + " · N hands-on lessons",
  stamp: "Sheet: <story><br>Legend: <what the maps show>",
  anchor: "<the running story, e.g. fitting delivery fees>",
  navWidth: 190,
  built: today,
  artifact: "",
  deckDividers: {},
  footer:
    "Each lesson runs Puzzle → Explore → Name → Challenge (interactive-educator method), for a learner who knows the chain rule and linear algebra but has no picture of networks yet. " +
    'Source deck: M. Soleymani, "' + decks.map((d) => d.replace(/\.pdf$/, "")).join('", "') + '", Sharif University of Technology. <Say which numbers are made-up teaching data.>',
  lessons: [],
};
fs.writeFileSync(path.join(dir, "lab.json"), JSON.stringify(lab, null, 2) + "\n");
console.log("created labs/" + id + " (common.jsx copied from labs/" + base.id + ")");
console.log("next: write l1-<name>.jsx ..., add each to lab.json lessons, then: node tools/build.js " + id);
