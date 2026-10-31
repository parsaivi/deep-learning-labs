# Deep Learning Labs

Interactive, hands-on lessons built from the Deep Learning course decks (M. Soleymani, Sharif University of Technology).
Each lab takes one lecture deck and turns every big idea into a 4-screen lesson: **Puzzle → Explore → Name → Challenge**.
You drag, tune and break things before the idea gets its name.

**Open the gallery: https://parsaivi.github.io/deep-learning-labs/**

| Lab | Deck | Lessons |
|---|---|---|
| [MLP & Backprop Lab](https://parsaivi.github.io/deep-learning-labs/mlp-backprop/) | MLP, MLP & Back-prop | XOR & hidden layers · universal classifier · universal approximator · depth vs width · smooth activations & loss · computational graph · backprop for a layer |
| [Optimization Lab](https://parsaivi.github.io/deep-learning-labs/optimization-1/) | Optimization I | step size vs curvature · condition number · Newton & saddles · decaying η · momentum & Nesterov · Adagrad / RMSprop / Adam |
| [Optimization Lab II](https://parsaivi.github.io/deep-learning-labs/optimization-2/) | Optimization II | SGD vs batch & sample order · why SGD never settles · mini-batch size · reading loss curves · warmup & schedules · L2 vs cross-entropy |

Visual explainer: [Lines that learn: MLPs and back-propagation](https://parsaivi.github.io/deep-learning-labs/explainers/mlp-backprop.html)

## Layout

```
labs/<lab>/          lab.json (titles, decks, lesson list) + one .jsx per lesson (+ optional common.jsx)
explainers/          standalone visual-explainer pages
course.json          every course deck, in order (drives the course map)
tools/build.js       labs/ → docs/ (GitHub Pages) and build/ (claude.ai Artifact copies)
tools/new-lab.js     start a new lab folder
tools/shoot.sh       headless-Chrome screenshot + crash check of one lesson screen
docs/                built site, served by GitHub Pages (don't edit by hand)
.claude/skills/interactive-educator/   the /aha skill, unchanged from upstream
```

**Adding the next deck: see [GUIDE.md](GUIDE.md).** In short: `node tools/new-lab.js`, run `/aha` on the deck,
`node tools/build.js <lab>`, then commit and push.

## Credits

- Lessons are generated with Claude using the [interactive-educator](https://github.com/Wamikmk/interactive-educator) skill by Wamik (CC BY 4.0), vendored unchanged in `.claude/skills/`.
- Slide content belongs to its authors (M. Soleymani; after CMU 11-785 and Stanford cs231n). The PDFs aren't included here.
- All deliveries, fees, zones and "networks" in the labs are made-up teaching data.
