// Builds every lab (or the ones named) into docs/, plus the gallery docs/index.html.
//   node tools/build.js                 -> all labs + gallery
//   node tools/build.js optimization-2  -> one lab + gallery
// Each lesson is a claude.ai-style React file (imports from "react", default export).
// The lab's common.jsx (optional) is prepended to every lesson; JSX is compiled here. A lesson may also
// import lucide-react or recharts and use Tailwind classes, as the skill allows; those load from CDNs.
// Outputs per lab:
//   docs/<id>/index.html        GitHub Pages copy (full document, "all labs" link goes to ../)
//   build/<id>.artifact.html    copy to publish as a claude.ai Artifact (the host adds the skeleton)
const fs = require("fs");
const path = require("path");
const Babel = require(path.join(__dirname, "babel.min.js"));

const ROOT = path.join(__dirname, "..");
const PAGES_URL = "https://parsaivi.github.io/deep-learning-labs/";
const SKELETON = '<!doctype html>\n<html lang="en">\n<meta charset="utf-8">\n<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n';

function strip(src) {
  return src
    .replace(/^import\s*\{([^}]*)\}\s*from\s*["']react["'];?\s*$/m, (m, names) => "const {" + names + "} = React;")
    .replace(/^\/\/ @common-end.*$/m, "");
}
// Libraries the interactive-educator skill may import (build-spec "Available libraries"), mapped to UMD globals.
const LIBS = {
  // lucide's UMD build reads a lowercase `react` global
  "lucide-react": { global: "LucideReact", pre: "window.react = window.React;", scripts: ["https://unpkg.com/lucide-react@0.400.0/dist/umd/lucide-react.min.js"] },
  recharts: { global: "Recharts", scripts: ["https://unpkg.com/prop-types@15.8.1/prop-types.min.js", "https://cdnjs.cloudflare.com/ajax/libs/recharts/2.12.7/Recharts.min.js"] },
};
const TAILWIND = "https://cdn.tailwindcss.com/3.4.16";
// import X, { a, b as c } from "lib"  ->  const X = G.default || G; const { a, b: c } = G;
function rewriteImports(src, used) {
  return src.replace(/^import\s+([\s\S]*?)\s+from\s*["']([^"']+)["'];?[ \t]*$/gm, (m, what, lib) => {
    if (lib === "react" || lib === "./common") return "";
    const L = LIBS[lib];
    if (!L) throw new Error('unsupported import "' + lib + '"');
    used.add(lib);
    const G = "window." + L.global;
    let out = "";
    const ns = what.match(/\*\s+as\s+(\w+)/);
    if (ns) return "const " + ns[1] + " = " + G + ";";
    const named = what.match(/\{([^}]*)\}/);
    const def = what.replace(/\{[^}]*\}/, "").replace(/,/g, "").trim();
    if (def) out += "const " + def + " = " + G + ".default || " + G + "; ";
    if (named) out += "const {" + named[1].replace(/\s+as\s+/g, ": ") + "} = " + G + ";";
    return out;
  });
}
function esc(s) {
  return String(s).replace(/&(?!amp;|lt;|gt;|#)/g, "&amp;");
}
function readLab(id) {
  return JSON.parse(fs.readFileSync(path.join(ROOT, "labs", id, "lab.json"), "utf8"));
}

function buildLab(id) {
  const dir = path.join(ROOT, "labs", id);
  const lab = readLab(id);
  // common.jsx is optional: a single file straight out of the skill builds without one
  const cp = path.join(dir, "common.jsx");
  const common = fs.existsSync(cp) ? strip(fs.readFileSync(cp, "utf8")) : "";
  const used = new Set();
  let tailwind = false;
  const present = [];
  let blocks = "";
  for (const L of lab.lessons) {
    const fp = path.join(dir, L.file);
    if (!fs.existsSync(fp)) {
      console.warn("  skip " + L.file + " (missing)");
      continue;
    }
    let src = fs.readFileSync(fp, "utf8");
    if (!common) src = src.replace(/^import\s+(?:\w+\s*,\s*)?\{([^}]*)\}\s*from\s*["']react["'];?[ \t]*$/m, (m, names) => "const {" + names + "} = React;");
    src = rewriteImports(src, used);
    if (/className=/.test(src)) tailwind = true;
    let compName = null;
    src = src.replace(/export\s+default\s+function\s+(\w+)/, (m, name) => {
      compName = name;
      return "function " + name;
    });
    if (!compName) throw new Error("no default export in " + L.file);
    if (/^\s*import\s/m.test(src)) throw new Error("unsupported import left in " + L.file);
    const wrapped = "(function (React) {\n" + common + "\n" + src + "\nreturn " + compName + ";\n})(window.React)";
    let code;
    try {
      code = Babel.transform(wrapped, { presets: ["react"], sourceType: "script" }).code;
    } catch (e) {
      console.error("JSX error in " + id + "/" + L.file + ":\n" + e.message);
      process.exit(1);
    }
    blocks += '<script>\nwindow.__LESSON_SRC = window.__LESSON_SRC || {};\nwindow.__LESSON_SRC["' + L.id + '"] = function () { return ' +
      code.replace(/<\/script/gi, "<\\/script").replace(/;\s*$/, "") + "; };\n</script>\n";
    present.push({ id: L.id, n: L.n, title: L.title, sub: L.sub });
  }

  let libTags = tailwind ? '<script src="' + TAILWIND + '"></script>\n' : "";
  for (const lib of used) if (LIBS[lib].pre) libTags += "<script>" + LIBS[lib].pre + "</script>\n";
  for (const lib of used) for (const u of LIBS[lib].scripts) libTags += '<script src="' + u + '"></script>\n';
  const tpl = fs.readFileSync(path.join(__dirname, "template.html"), "utf8");
  const page = (home) =>
    tpl
      .split("__TITLE__").join(esc(lab.title))
      .replace("__SUB__", esc(lab.sub))
      .replace("__STAMP__", lab.stamp)
      .replace("__FOOTER__", lab.footer)
      .replace("__HOME__", home)
      .replace("__NAVW__", String(lab.navWidth || 190))
      .replace("/*__DECKS__*/", JSON.stringify(lab.deckDividers || {}))
      .replace("/*__LESSONS__*/", JSON.stringify(present))
      .replace("<!--__SOURCES__-->", () => libTags + blocks);
  fs.mkdirSync(path.join(ROOT, "docs", id), { recursive: true });
  fs.mkdirSync(path.join(ROOT, "build"), { recursive: true });
  fs.writeFileSync(path.join(ROOT, "docs", id, "index.html"), SKELETON + page("../"));
  const art = page(PAGES_URL);
  fs.writeFileSync(path.join(ROOT, "build", id + ".artifact.html"), art);
  console.log("built " + id + ": " + present.map((p) => p.id).join(", ") + " (" + Math.round(art.length / 1024) + " KB)");
}

function buildGallery() {
  const course = JSON.parse(fs.readFileSync(path.join(ROOT, "course.json"), "utf8"));
  const labs = fs
    .readdirSync(path.join(ROOT, "labs"))
    .filter((d) => fs.existsSync(path.join(ROOT, "labs", d, "lab.json")))
    .map(readLab);
  fs.mkdirSync(path.join(ROOT, "docs", "explainers"), { recursive: true });
  for (const x of course.explainers || []) fs.copyFileSync(path.join(ROOT, "explainers", x.file), path.join(ROOT, "docs", "explainers", x.file));
  const data = { course, labs: labs.map((l) => ({ id: l.id, title: l.title, sub: l.sub, decks: l.decks, anchor: l.anchor, built: l.built, artifact: l.artifact, lessons: l.lessons.map((x) => ({ id: x.id, n: x.n, title: x.title, sub: x.sub, slides: x.slides })) })) };
  const tpl = fs.readFileSync(path.join(__dirname, "gallery.template.html"), "utf8");
  fs.writeFileSync(path.join(ROOT, "docs", "index.html"), tpl.replace("/*__DATA__*/", () => JSON.stringify(data)));
  fs.writeFileSync(path.join(ROOT, "docs", ".nojekyll"), "");
  console.log("built gallery: " + labs.length + " labs, " + (course.explainers || []).length + " explainers");
}

const ids = process.argv.slice(2);
const all = fs.readdirSync(path.join(ROOT, "labs")).filter((d) => fs.existsSync(path.join(ROOT, "labs", d, "lab.json")));
for (const id of ids.length ? ids : all) buildLab(id);
buildGallery();
