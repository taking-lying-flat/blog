# DiffusionNFT Source

- User export: `/home/cherry-cloud/Desktop/无标题文档.lake`, archived unchanged as `source.lake`.
- The Lake export is rendered directly and appended after Flow-GRPO and its derivations in Generative Models VI.
- Formula and image assets referenced by the export are stored under `assets/` and indexed by `manifest.json`.

The introduction and background render directly from the Lake export. At the user's request, the selected passage beginning “在线强化学习” is reorganized in `../diffusion-nft-training.md`, retaining the draft's sequence of distributions, reinforcement guidance, policy optimization, advantages, and implementation choices. Related formulas are grouped and numbered, repeated transitions are shortened, and the training algorithm appears in LaTeX at the end.

The user subsequently requested “Proof Of Theorems”. This section follows Appendix A of the [ICLR 2026 conference paper](https://proceedings.iclr.cc/paper_files/paper/2026/file/d8e68ddfe22520b45b8fb8d5cbde5a21-Paper-Conference.pdf): Lemmas A.1–A.2 and the proofs of Theorems 3.1–3.2 (A.3–A.4). Proof equations use A.1–A.8 so the existing implementation references remain unchanged. The final square completion uses an algebraically equivalent expansion that does not divide by the posterior mixing coefficient. No experiment section is added.
