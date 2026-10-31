# Guide: turning the next deck into a lab

You generate a lab with the **interactive-educator** skill (`/aha`), exactly as published.
This repo doesn't change the skill. It only gives the skill's output a place to live, a build, a check and a public page.

```
deck PDF ──/aha──▶ lesson .jsx files ──tools/build.js──▶ docs/<lab>/index.html ──git push──▶ GitHub Pages
```

## 0. One-time setup

- Start Claude Code **inside this folder** (`deep-learning-labs/`). Claude Code picks up the skill from
  `.claude/skills/interactive-educator/` (a byte-identical copy of the upstream skill, CC BY 4.0), so `/aha` works
  without installing anything.
- The slide PDFs stay one folder up (`../CNN.pdf` etc.) and are never committed.
- Needs `node` (for the build) and `google-chrome` (for screenshots/crash checks).

## 1. Make the lab folder

```bash
node tools/new-lab.js cnn "CNN Lab" "CNN.pdf"
```

This creates `labs/cnn/lab.json` and copies `common.jsx` (the Blueprint UI kit and math helpers shared by the earlier labs)
from the newest lab. Using `common.jsx` is optional: delete it if you want plain skill output.

## 2. Generate the lessons with the skill

A prompt that worked for the three existing labs (replace the deck and folder):

> /aha the deck ../CNN.pdf — one 4-screen lesson per big idea in the deck.
> Learner profile: reuse the one in GUIDE.md (don't re-ask).
> Save each lesson as labs/cnn/l1-<name>.jsx, l2-…, and add it to labs/cnn/lab.json `lessons` with the slide range it covers.
> You may use the helpers in labs/cnn/common.jsx. Build with `node tools/build.js cnn` and check every screen with tools/shoot.sh.

Notes:
- The skill's default output path (`/mnt/user-data/outputs/…`) only exists on claude.ai. In Claude Code, telling it where to save
  (as in the prompt above) is enough. You never need to edit the skill.
- The skill's raw output works as-is. The build supports the same environment the skill targets: `react` hooks,
  `lucide-react`, `recharts` and Tailwind core classes. These load from CDNs only when a lab uses them.
- If you run `/aha` on claude.ai instead, download the `.jsx`, drop it into `labs/<lab>/`, and add it to `lab.json` by hand.

### Learner profile (answers to the skill's Phase 1 questions)

| Question | Answer used so far |
|---|---|
| Background | Knows the chain rule and linear algebra; no intuition for networks yet |
| Theme | Blueprint (dark navy drafting sheet, cyan + amber) |
| Running story (anchor) | Delivery app: MLP = city delivery zones; Optimization I & II = fitting delivery fees (per-km rate, base fee) |
| Depth | One lesson per big idea, about 6 per deck, each Puzzle → Explore → Name → Challenge |

Change any row here and the next lab follows it.

## 3. Build and look

```bash
node tools/build.js cnn                          # one lab (+ the gallery). No argument = all labs.
xdg-open docs/cnn/index.html                     # open it locally
tools/shoot.sh cnn conv-3 /tmp/conv-3.png        # screenshot lesson "conv", screen 3, and fail on any crash
```

Each lab page links lessons as `#<lessonId>` or `#<lessonId>-<screen 1..4>`, so you can send someone straight to a challenge.

## 4. lab.json fields

| Field | Meaning |
|---|---|
| `id` | folder name and URL path (`/cnn/`) |
| `title`, `sub`, `stamp`, `footer` | header, sub-header, the boxed "sheet" legend top-right, and the source/credits line (HTML allowed in `stamp`/`footer`) |
| `decks` | PDF file names this lab covers; marks them ✓ on the gallery's course map |
| `anchor` | the running story, shown on the gallery card |
| `deckDividers` | `{ "5": "Deck 2 · …" }` puts a heading before lesson 5 when a lab spans two decks |
| `lessons[]` | `id`, `n`, `title`, `sub` (one-line keywords), `file`, `slides` |
| `artifact` | optional claude.ai Artifact URL, shown as "claude.ai copy" |
| `built` | date, shown on the card |

Visual-explainer pages (plain scrolling pages) go in `explainers/` and are listed under `explainers` in `course.json`.
`course.json` also holds the course deck order for the map; edit it freely.

## 5. Publish

```bash
git add -A && git commit -m "Add CNN lab" && git push
```

GitHub Pages serves `docs/`, so the gallery and the new lab are live a minute later at
https://parsaivi.github.io/deep-learning-labs/.
For a private claude.ai copy as well, ask Claude to publish `build/cnn.artifact.html` as an Artifact and put the URL in `lab.json`.

## Quality checklist (lessons from the first three labs)

- **Tune every challenge numerically before building the UI.** Find the pass window (e.g. "η between 0.018 and 0.035 passes")
  with a quick node script, so the challenge is neither impossible nor passable by luck.
- **Single random runs can be too noisy to teach from.** When the noise swamps the effect, plot the exact expected curve and say so in the footer.
- **Say what's made up.** The footer states which data and models are teaching toys.
- **Check every screen**, not just the first: crashes often show up on screen 4 only. Also check a phone-width view.
- **When a toy model breaks the real lesson** (e.g. a huge η rescuing L2 in a 2-weight model), constrain the challenge and give the story a reason for the limit, instead of teaching the wrong thing.
