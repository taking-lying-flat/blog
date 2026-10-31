# OnPolicy Distill

The article combines two desktop Lake exports in reading order:

1. `opd/source.lake`: the OPD and generalized knowledge distillation notes, imported from `无标题文档.lake` with 59 formula cards and one locally archived image.
2. `source.lake`: the DOPD notes, imported from `𝑶𝒏𝑷𝒐𝒍𝒊𝒄𝒚𝑫𝒊𝒔𝒕𝒊𝒍𝒍-𝑰.lake` with 45 formula cards and six images.

Both unchanged sources are preserved with their manifests. The site prepends OPD before DOPD, renders the OPD formulas through MathJax, and publishes the combined article as **OnPolicy Distill** in the **策略优化** category. The GKD screenshot is replaced at render time by the selectable LaTeX transcription in `opd-algorithm.md`; OPD equation annotations are English, every display equation uses the site's responsive numbering, and unnecessary display breaks are removed. Prose edits are recorded in `opd/text-overrides.json`. DOPD formula overrides remove only the outer `\boxed` decoration to match the existing blog style; the underlying expressions remain from the exports.
