# Poster (symposium template, A1 portrait)

    .venv/bin/python poster/build_poster.py --template <template.pptx> --figs <figure dir> --out <draft.pptx>

`build_poster.py` keeps the template header (title box, names, logos, rule),
removes the example boxes and lays out the two-act poster at millimetre
coordinates (`LAYOUT`): Act I (critic calibration: GapReach clue, Can/Square
ladder, demo replay) on the left, Table 1 + Act II (entropy collapse,
prescription, Square trade-off and causal test) on the right. Text with
`[tonight]` marks values from runs still in progress; `[colleague]` marks
material from the GapReach2D study. It also writes `<out>_layout.png`, a
wireframe of every box, since a PPTX cannot be rendered here.

Figures come from `scripts/plot_results.py` (`success_<axis>_early.png`)
plus `replay_bars.png` (early-AUC bars of the replay axis, drawn inline in
the session that produced draft v0; to be moved into a script with the
final numbers).
