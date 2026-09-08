"""Restructure the two-column A1 poster: GapReach2D result, robomimic verification, temperature.

    python poster/restructure_poster.py --in <team Poster.pptx> --figs <dir> --out <draft.pptx>

Works on a copy of the team's poster and never touches the input. Shapes are
addressed by their names in that file (TextBox 137, Table 144, Picture 149, ...),
so it is specific to that layout. <figs> must hold critic_ladder_early.png and
dip_can.png from scripts/poster_figs.py.

Text markup: **bold**, ^{superscript}, _{subscript} (not nested). A paragraph
may also be given as (text, {"bold_all": True}).

Measured on the team's file: Arial 21 pt in a 10.9 in column wraps at about
68 characters and takes 0.42 in per line; 19 pt about 75 characters / 0.38 in;
17 pt about 85 characters / 0.34 in. The budgets below follow those numbers.
"""
import argparse
import copy
import re

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

INK = RGBColor(0x1A, 0x1A, 0x1A)
MUTED = RGBColor(0x5A, 0x60, 0x68)
NAVY = RGBColor(0x00, 0x22, 0x66)
TEAL = RGBColor(0x15, 0x60, 0x82)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
TOKEN = re.compile(r"(\*\*.+?\*\*|\^\{.+?\}|_\{.+?\})")

L_X, R_X, COL_W = 0.4, 11.8, 10.9


def style_run(run, size, bold=False, italic=False, color=INK, baseline=None):
    f = run.font
    f.name = "Arial"
    f.size = Pt(size)
    f.bold = bold
    f.italic = italic
    f.color.rgb = color
    if baseline:
        run._r.get_or_add_rPr().set("baseline", baseline)


def fill_paragraph(para, text, size, color=INK, bold_all=False, italic=False):
    for piece in TOKEN.split(text):
        if not piece:
            continue
        bold, baseline, body = bold_all, None, piece
        if piece.startswith("**"):
            bold, body = True, piece[2:-2]
        elif piece.startswith("^{"):
            baseline, body = "30000", piece[2:-1]
        elif piece.startswith("_{"):
            baseline, body = "-25000", piece[2:-1]
        run = para.add_run()
        run.text = body
        style_run(run, size, bold=bold, italic=italic, color=color, baseline=baseline)


def set_text(shape, paragraphs, size=21, color=INK, align=None, line=1.14, after=6, bold_all=False, italic=False):
    tf = shape.text_frame
    tf.word_wrap = True
    tf.clear()
    for i, item in enumerate(paragraphs):
        text, opts = (item, {}) if isinstance(item, str) else item
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.line_spacing = line
        para.space_after = Pt(after)
        if align is not None:
            para.alignment = align
        fill_paragraph(para, text, size, color=color, bold_all=opts.get("bold_all", bold_all), italic=italic)


def place(shape, x, y, w=None, h=None):
    shape.left, shape.top = Inches(x), Inches(y)
    if w is not None:
        shape.width = Inches(w)
    if h is not None:
        shape.height = Inches(h)


def remove(shape):
    el = shape._element
    el.getparent().remove(el)


def add_textbox(slide, x, y, w, h, paragraphs, **kw):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    set_text(box, paragraphs, **kw)
    return box


def style_cell(cell, text, size, header=False, align=PP_ALIGN.CENTER):
    cell.margin_left = cell.margin_right = Emu(79200)
    cell.margin_top = cell.margin_bottom = Emu(18000)
    cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf = cell.text_frame
    tf.word_wrap = True
    para = tf.paragraphs[0]
    para.alignment = align
    fill_paragraph(para, text, size, color=WHITE if header else INK, bold_all=header)
    if header:
        cell.fill.solid()
        cell.fill.fore_color.rgb = TEAL


def add_table(slide, x, y, w, rows, col_w, row_h, size=19, header_rows=1, merges=()):
    """A table in the poster's own look: teal header, white bold header text, banded rows."""
    n_rows, n_cols = len(rows), len(rows[0])
    frame = slide.shapes.add_table(n_rows, n_cols, Inches(x), Inches(y), Inches(w), Inches(sum(row_h)))
    table = frame.table
    tbl_pr = table._tbl.tblPr
    tbl_pr.set("firstRow", "1")
    tbl_pr.set("bandRow", "1")
    for ci, cw in enumerate(col_w):
        table.columns[ci].width = Inches(cw)
    for ri, row in enumerate(rows):
        table.rows[ri].height = Inches(row_h[ri])
        for ci, text in enumerate(row):
            style_cell(table.cell(ri, ci), text, size, header=ri < header_rows,
                       align=PP_ALIGN.LEFT if (ci == 0 and ri >= header_rows) else PP_ALIGN.CENTER)
    for (r0, c0, r1, c1) in merges:
        table.cell(r0, c0).merge(table.cell(r1, c1))
    return frame


def insert_row_after(table_shape, src_idx, texts):
    """Duplicate row src_idx (keeps its fonts and colours) below itself and set its cell texts."""
    tbl = table_shape.table._tbl
    src = tbl.tr_lst[src_idx]
    new_tr = copy.deepcopy(src)
    src.addnext(new_tr)
    for tc, text in zip(new_tr.tc_lst, texts):
        runs = tc.findall(".//" + qn("a:r"))
        if not runs:
            continue
        runs[0].find(qn("a:t")).text = text
        for extra in runs[1:]:
            extra.getparent().remove(extra)
    table_shape.height = table_shape.height + src.h


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--figs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prs = Presentation(args.src)
    slide = prs.slides[0]
    by_name = {}
    for sh in slide.shapes:
        by_name.setdefault(sh.name, []).append(sh)

    def one(name):
        (shape,) = by_name[name]
        return shape

    def rename_first_cell(table_shape, row, col, text):
        cell = table_shape.table.cell(row, col)
        runs = cell.text_frame.paragraphs[0].runs
        runs[0].text = text
        for extra in runs[1:]:
            extra._r.getparent().remove(extra._r)

    # ---- title ---------------------------------------------------------
    for sh in by_name["제목 1"]:
        if sh.text_frame.text.startswith("Improving"):
            set_text(sh, ["Offline RL Methods for Offline-to-Online Fine-Tuning of Diffusion Policies with DSRL"],
                     size=54, color=NAVY, align=PP_ALIGN.CENTER, line=1.0, after=0, bold_all=True)

    # ---- left column ---------------------------------------------------
    place(one("TextBox 137"), L_X, 5.55, COL_W, 3.3)
    set_text(one("TextBox 137"), [
        "**DSRL** fine-tunes a frozen diffusion policy by learning which initial noise to feed it, but its "
        "offline-to-online setting applies **no offline RL** to the critic: Q^{A} starts uncalibrated and its error "
        "propagates through the two-critic chain. We explore offline RL methods for pretraining that critic — "
        "**TD, IQL, CQL, Cal-QL** — with the online algorithm untouched. On a 2D surrogate Cal-QL cuts the online "
        "steps to 50% success by **2.7×**; on robomimic Can and Square every pretrained critic lifts the floor of "
        "the early dip, but no method beats in-sample IQL.",
    ])

    place(one("TextBox 138"), L_X, 8.95, COL_W, 0.8)
    place(one("TextBox 139"), L_X, 9.8, COL_W, 2.9)
    set_text(one("TextBox 139"), [
        "**DSRL (CoRL 2025).** With DDIM at η = 0 the frozen policy is a deterministic map a = π_{dp}(s, w), so the "
        "initial noise w is the RL action. DSRL-NA learns Q^{A}(s, a) by TD on real transitions and distils "
        "Q^{W}(s, w) from it by querying the policy forward only, so offline data without w can be used.",
        "**Cal-QL (NeurIPS 2023).** Conservative offline RL suppresses over-estimation on unseen actions, but "
        "unbounded pessimism pushes Q far below what the behaviour policy achieved; Cal-QL floors the penalty at a "
        "reference return.",
    ])

    diagram = one("Picture 140")
    dw = 9.2
    place(diagram, L_X + (COL_W - dw) / 2, 12.8, dw, dw / (10.9 / 4.5))
    place(one("TextBox 141"), L_X, 16.65, COL_W, 0.4)

    place(one("TextBox 142"), L_X, 17.15, COL_W, 0.8)
    place(one("TextBox 143"), L_X, 18.0, COL_W, 2.5)
    set_text(one("TextBox 143"), [
        "Pretrain Q^{A} offline with an offline RL method, then run the **unmodified** DSRL online algorithm. "
        "Conservatism lives offline only; Cal-QL's calibration is one line inside the penalty:",
        ("          Q(s, a)   →   max( Q(s, a),  G_{t} )", {"bold_all": True}),
        "G_{t}: Monte-Carlo return-to-go at **chunk granularity**, on the online reward's scale.",
    ])

    arms = one("Table 144")
    place(arms, L_X, 20.45)
    rename_first_cell(arms, 1, 1, "none — DSRL baseline")
    insert_row_after(arms, 2, ["iql", "in-sample expectile V (robomimic)", "does avoiding OOD queries help?"])

    place(one("TextBox 145"), L_X, 23.85, COL_W, 1.15)
    set_text(one("TextBox 145"), [
        "GapReach2D: four arms one factor apart, identical data and online code. Robomimic runs all five; its "
        "CQL/Cal-QL are critic-only adaptations (IQL V target, data action anchored in the log-sum-exp).",
    ], size=19, color=MUTED, align=PP_ALIGN.CENTER, after=0)

    place(one("TextBox 146"), L_X, 25.05, COL_W, 1.3)
    set_text(one("TextBox 146"), [
        "**Controls.** Actor and Q^{W} stay frozen during pretraining; Q^{A} is then distilled into Q^{W}, all the "
        "actor reads. Penalty candidates are physical action chunks π_{dp}(s, w), never latent noise.",
    ])

    add_textbox(slide, L_X, 26.45, COL_W, 0.45, ["**Offline Q-values relative to demonstration returns**"],
                size=21, after=0)
    add_table(slide, L_X, 26.95, COL_W, [
        ["Offline phase", "GapReach2D  Q − G", "Can  E_{w}Q^{A} − G", "Square  E_{w}Q^{A} − G"],
        ["TD warm-start", "+0.28", "−49", "−98"],
        ["CQL", "**−9.06**", "−70", "+2"],
        ["Cal-QL", "−0.22", "−58", "+23"],
    ], col_w=[2.9, 2.7, 2.65, 2.65], row_h=[0.55, 0.5, 0.5, 0.5], size=19)
    add_textbox(slide, L_X, 29.05, COL_W, 0.7, [
        "Before online learning. Scales: GapReach returns ∈ [0, 1]; robomimic G ≈ −100 (Can) / −150 (Square). "
        "Robomimic Q is taken on prior-noise actions and G on demonstration actions: a scale comparison, not an "
        "estimation error.",
    ], size=15, color=MUTED, align=PP_ALIGN.CENTER, after=0, line=1.05)

    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(L_X), Inches(30.0), Inches(COL_W), Inches(2.35))
    box.adjustments[0] = 0.08
    box.fill.solid()
    box.fill.fore_color.rgb = RGBColor(0xEA, 0xF1, 0xF8)
    box.line.color.rgb = TEAL
    box.line.width = Pt(2.25)
    box.shadow.inherit = False
    set_text(box, [
        "**Take-away.** Offline critic pretraining mitigates the early performance dip, but calibration does not "
        "consistently provide additional gains on the tested robotic tasks.",
    ], size=25, color=NAVY, line=1.1, after=0)
    box.text_frame.margin_left = box.text_frame.margin_right = Inches(0.3)
    box.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE


    # ---- right column --------------------------------------------------
    place(one("TextBox 148"), R_X, 5.55, COL_W, 2.75)
    set_text(one("TextBox 148"), [
        "**Task — GapReach2D.** A sparse-reward planar task with DSRL's structure: chunked actions, a frozen DDIM "
        "policy as a black box, the two-critic chain. A wall has a wide (safe) and a narrow (risky) gap; contact fails "
        "the episode, the goal gives +1. Demonstrations succeed 60%; the cloned policy 22.5%.",
        "**Protocol.** 3 seeds per arm with their own demonstrations and base policy; 20,000 online steps. "
        "T_{x} = online steps to first reach x% success.",
    ])
    curve = one("Picture 149")
    cw = 5.9
    place(curve, R_X + (COL_W - cw) / 2, 8.35, cw, cw / (7.5 / 5.5))
    place(one("TextBox 150"), R_X, 12.7, COL_W, 0.4)
    gap = one("Table 151")
    place(gap, R_X, 13.15)
    rename_first_cell(gap, 1, 0, "DSRL baseline")
    place(one("TextBox 152"), R_X, 15.95, COL_W, 0.4)

    for name in ("Table 153", "TextBox 154", "Picture 155", "TextBox 156"):
        remove(one(name))

    add_textbox(slide, R_X, 16.45, COL_W, 1.65, [
        "**Does it transfer? Robomimic Can and Square.** The same offline RL ladder plus IQL; 3–5 seeds, 150k env steps, no "
        "demonstration replay so only the critic differs. Reference: the frozen diffusion policy with N(0, I) noise "
        "(Can 0.405, Square 0.494).",
    ])
    lw = 10.5
    slide.shapes.add_picture(f"{args.figs}/critic_ladder_early.png", Inches(R_X + (COL_W - lw) / 2), Inches(18.15),
                             width=Inches(lw), height=Inches(lw / (10.9 / 5.6)))
    add_textbox(slide, R_X, 23.6, COL_W, 0.6, [
        "Early window, mean ± SE over seeds. Every offline-RL critic lifts the floor of the dip; none removes it.",
    ], size=18, color=MUTED, align=PP_ALIGN.CENTER, after=0, line=1.0)
    add_table(slide, R_X, 24.25, COL_W, [
        ["", "Can", "", "", "Square", "", ""],
        ["Critic", "min", "early AUC", "129k", "min", "early AUC", "127k"],
        ["DSRL baseline (n=5)", "0.24", "0.41±0.03", "0.53±0.06", "0.23", "0.40±0.01", "0.41±0.06"],
        ["IQL (n=5 / 3)", "0.34", "0.49±0.02", "0.56±0.02", "0.37", "0.44±0.01", "0.56±0.02"],
        ["TD (n=3)", "0.39", "0.47±0.03", "0.56±0.03", "0.44", "0.49±0.02", "0.42±0.06"],
        ["CQL-style (n=3)", "0.35", "0.48±0.02", "0.58±0.05", "0.37", "0.47±0.01", "0.33±0.08"],
        ["Cal-QL-style (n=3)", "0.34", "0.43±0.02", "0.51±0.05", "0.40", "0.46±0.03", "0.42±0.05"],
    ], col_w=[2.4, 1.15, 1.6, 1.55, 1.15, 1.6, 1.45], row_h=[0.45, 0.5, 0.46, 0.46, 0.46, 0.46, 0.46],
        size=17, header_rows=2, merges=[(0, 1, 0, 3), (0, 4, 0, 6)])

    place(one("TextBox 157"), R_X, 27.6, COL_W, 0.8)
    place(one("TextBox 158"), R_X, 28.45, COL_W, 4.45)
    set_text(one("TextBox 158"), [
        "**GapReach2D: calibration cuts interaction.** Cal-QL needs 2.7× fewer online steps to 50% success; plain "
        "CQL backfires by flattening Q.",
        "**Robomimic: pretraining helps, calibration adds little.** Every offline-RL critic lifts the floor of the "
        "dip, but Cal-QL-style pretraining provides no consistent additional benefit in the tested settings "
        "(seed-matched vs IQL: early AUC −0.08 ± 0.02 on Can, late −0.15 ± 0.06 on Square).",
        "**Entropy, briefly.** Preliminary ablations suggest the online entropy setting also shapes the transition "
        "(fixed α = 0.3 removes the mean-curve dip on Can, min 0.47 vs 0.24); future work will test whether "
        "preserving critic calibration online improves robustness.",
        "**Limits.** 3–5 seeds, ±0.1 evaluation noise; Square keeps a dip in every setting; robomimic CQL/Cal-QL are "
        "critic-only adaptations.",
    ], size=18.5, after=4)

    prs.save(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
