"""Restructure the two-column A1 poster (v4): offline RL methods for O2O DSRL.

    python poster/restructure_poster.py --in <team Poster.pptx> --figs <dir> --out <draft.pptx>

Works on a copy of the team's poster and never touches the input. Shapes are
addressed by their names in that file (TextBox 137, Table 144, Picture 149, ...),
so it is specific to that layout. <figs> must hold critic_ladder_early.png
(scripts/poster_figs.py) and the two task crops can_crop.png / square_crop.png.

Flow: problem -> approach (pretrain Q^A, online algorithm unchanged) -> five
conditions -> GapReach2D and Can/Square -> conclusion -> follow-up (entropy,
replay, two sentences). Entropy has no panel of its own.

Text markup: **bold**, ^{superscript}, _{subscript} (not nested). A paragraph
may also be given as (text, {"bold_all": True}).

Measured on the team's file: Arial 21 pt in a 10.9 in column wraps at about
68 characters and takes 0.42 in per line; 19 pt about 75 characters / 0.38 in;
17 pt about 85 characters / 0.34 in. Positions below are a running cursor.
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
TINT = RGBColor(0xEA, 0xF1, 0xF8)
TOKEN = re.compile(r"(\*\*.+?\*\*|\^\{.+?\}|_\{.+?\})")

L_X, R_X, COL_W = 0.4, 11.8, 10.9
PAGE_W, PAGE_H = 23.39, 33.11


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


def rename_first_cell(table_shape, row, col, text):
    cell = table_shape.table.cell(row, col)
    runs = cell.text_frame.paragraphs[0].runs
    runs[0].text = text
    for extra in runs[1:]:
        extra._r.getparent().remove(extra._r)


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

    # ---- title (unchanged from v3: the assigned topic) ------------------
    for sh in by_name["제목 1"]:
        if sh.text_frame.text.startswith("Improving") or sh.text_frame.text.startswith("Offline RL"):
            set_text(sh, ["Offline RL Methods for Offline-to-Online Fine-Tuning of Diffusion Policies with DSRL"],
                     size=50, color=NAVY, align=PP_ALIGN.CENTER, line=1.0, after=0, bold_all=True)

    # ---- header block: tighter top margin (allowed by the TA) --------------
    for sh in by_name["제목 1"]:
        t = sh.text_frame.text
        if t.startswith("Offline RL"):
            place(sh, 0.7, 0.1, 22.0, 2.4)
        elif t.startswith("[Your Name]"):
            place(sh, 0.0, 2.7, PAGE_W, 0.55)
        elif t.startswith("Department"):
            place(sh, 0.0, 3.2, PAGE_W, 0.4)
    for name in ("그림 268", "그림 35"):
        for sh in by_name.get(name, []):
            sh.top = Inches(2.55)
    TOP = 4.0

    # ============================ left column ============================
    y = TOP
    place(one("TextBox 136"), L_X, y, COL_W, 0.8); y += 0.85
    set_text(one("TextBox 136"), ["Objective"], size=46, color=NAVY, bold_all=True, after=0, line=1.0)
    place(one("TextBox 137"), L_X, y, COL_W, 2.0)
    set_text(one("TextBox 137"), [
        "Can offline RL improve DSRL's offline-to-online transition? We keep the frozen diffusion policy and the "
        "online algorithm unchanged and pretrain the action-space critic Q^{A} offline with **TD, IQL, CQL-style "
        "or Cal-QL-style** objectives, comparing them on a 2D surrogate and on robomimic Can and Square.",
    ])
    y += 2.05

    place(one("TextBox 138"), L_X, y, COL_W, 0.8); y += 0.85
    set_text(one("TextBox 138"), ["Approach"], size=46, color=NAVY, bold_all=True, after=0, line=1.0)
    place(one("TextBox 139"), L_X, y, COL_W, 4.2)
    set_text(one("TextBox 139"), [
        "**DSRL (CoRL 2025)** fine-tunes a frozen diffusion policy by learning its input noise: with DDIM at η = 0, "
        "a = π_{dp}(s, w), so the noise w is the RL action. Q^{A}(s, a) is learned by TD on real transitions and "
        "Q^{W}(s, w) is distilled from it through forward queries of the policy. We pretrain Q^{A} offline with one "
        "offline RL objective, distil it into Q^{W}, then run the **unmodified** online algorithm.",
        "**Cal-QL (NeurIPS 2023)** differs from CQL-style only inside the conservative penalty: each sampled "
        "unseen-action value enters it as",
        ("          Q(s, a)   →   max( Q(s, a),  G_{t} )", {"bold_all": True}),
        "with G_{t} the demonstration's Monte-Carlo return-to-go at chunk granularity. This bounds the penalty's "
        "push-down; it does not constrain the learned Q itself.",
    ])
    y += 4.25
    remove(one("TextBox 142"))
    remove(one("TextBox 143"))

    diagram = one("Picture 140")
    dw = 8.6
    dh = dw / (10.9 / 4.5)
    place(diagram, L_X + (COL_W - dw) / 2, y, dw, dh); y += dh + 0.05
    place(one("TextBox 141"), L_X, y, COL_W, 0.4)
    set_text(one("TextBox 141"), [
        "Q^{A} is the only network pretrained offline; it is the learning target of Q^{W}.",
    ], size=18, color=MUTED, align=PP_ALIGN.CENTER, after=0, line=1.0)
    y += 0.45

    arms = one("Table 144")
    place(arms, L_X, y)
    rename_first_cell(arms, 1, 1, "none — DSRL baseline")
    insert_row_after(arms, 2, ["iql", "in-sample expectile V (robomimic)", "does avoiding OOD queries help?"])
    y += 0.56 * 6 + 0.05

    place(one("TextBox 145"), L_X, y, COL_W, 1.9)
    set_text(one("TextBox 145"), [
        "GapReach2D runs four arms (no IQL); robomimic runs all five. Robomimic CQL/Cal-QL are critic-only "
        "adaptations (IQL in-sample V target, data action anchored in the log-sum-exp). Controls: the latent actor "
        "and Q^{W} stay frozen during pretraining; penalty candidates are physical action chunks π_{dp}(s, w), "
        "never latent noise.",
    ], size=19, color=MUTED, after=0)
    remove(one("TextBox 146"))
    y += 1.95

    add_textbox(slide, L_X, y, COL_W, 0.45, ["**Critic diagnostics**"], size=24, color=NAVY, after=0, line=1.0)
    y += 0.5
    bias = one("Table 153")
    place(bias, L_X, y)
    rename_first_cell(bias, 1, 0, "DSRL baseline")
    rename_first_cell(bias, 0, 0, "GapReach2D  Q − G")
    y += 0.55 * 5 + 0.05
    place(one("TextBox 154"), L_X, y, COL_W, 0.7)
    set_text(one("TextBox 154"), [
        "GapReach2D (returns in [0, 1]): Cal-QL shows the smallest online over-estimation spike; CQL's scale collapsed.",
    ], size=16, color=MUTED, align=PP_ALIGN.CENTER, after=0, line=1.0)
    y += 0.75

    add_table(slide, L_X, y, COL_W, [
        ["Robomimic, before online", "Can  E_{w}Q^{A} − G", "Square  E_{w}Q^{A} − G"],
        ["TD", "−49", "−98"],
        ["CQL-style", "−70", "+2"],
        ["Cal-QL-style", "−58", "+23"],
    ], col_w=[4.3, 3.3, 3.3], row_h=[0.55, 0.5, 0.5, 0.5], size=19)
    y += 2.1
    add_textbox(slide, L_X, y, COL_W, 0.75, [
        "Q on prior-noise actions π_{dp}(s, w) minus the demonstration return G (≈ −100 Can, −150 Square): a scale "
        "comparison on a different return scale, not an estimation error and not comparable to the table above.",
    ], size=15, color=MUTED, align=PP_ALIGN.CENTER, after=0, line=1.05)
    y += 0.8

    add_textbox(slide, L_X, y, COL_W, 2.1, [
        "**Metrics & protocol.** Online step = env step − initial rollout (24,016 Can / 32,016 Square). "
        "min = lowest point of the seed-mean success curve on the common 5k online-step grid within the early "
        "window; early AUC = normalised area over online steps 0–76k (Can) / 0–68k (Square), initial evaluation at "
        "step 0; late = evaluation at env step 129,152 (Can) / 127,136 (Square). 100 episodes per early "
        "evaluation, 200 late; mean ± SE over seeds.",
    ], size=17, line=1.1, after=0)
    left_bottom = y + 2.1

    # ============================ right column ===========================
    y = TOP
    place(one("TextBox 147"), R_X, y, COL_W, 0.8); y += 0.85          # "Experiments"
    place(one("TextBox 148"), R_X, y, COL_W, 2.5)
    set_text(one("TextBox 148"), [
        "**GapReach2D.** Sparse-reward planar reaching with DSRL's structure (chunked actions, frozen DDIM policy, "
        "two-critic chain): a wall with a wide safe gap and a narrow risky gap, re-randomised per episode; contact "
        "fails, the goal gives +1. Demonstrations succeed 60%, the cloned policy 22.5%. 3 seeds per arm, 20,000 "
        "online steps; T_{x} = online steps to first reach x% success.",
    ])
    y += 2.55
    curve = one("Picture 149")
    ch = 3.3
    cw = ch * (7.5 / 5.5)
    place(curve, R_X + (COL_W - cw) / 2, y, cw, ch); y += ch + 0.05
    place(one("TextBox 150"), R_X, y, COL_W, 0.4); y += 0.45
    gap = one("Table 151")
    place(gap, R_X, y)
    rename_first_cell(gap, 1, 0, "DSRL baseline")
    y += 0.55 * 5 + 0.05
    place(one("TextBox 152"), R_X, y, COL_W, 0.4); y += 0.5

    for name in ("Picture 155", "TextBox 156"):
        remove(one(name))

    add_textbox(slide, R_X, y, COL_W, 2.0, [
        "**Robomimic Can and Square.** All five conditions, 3–5 seeds, 150k env steps, no demonstration replay so "
        "that only the critic differs. Dashed reference: the frozen diffusion policy with N(0, I) noise (Can 0.405, "
        "Square 0.494); DSRL baseline = the same policy fine-tuned online from a random critic. Curves: early "
        "window, seed mean ± SE.",
    ])
    y += 2.05

    ph = 2.5
    pw = ph * 1.6
    gap_x = 0.5
    x0 = R_X + (COL_W - (2 * pw + gap_x)) / 2
    add_textbox(slide, x0, y, pw, 0.35, ["**Can: Pick and place**"], size=19, align=PP_ALIGN.CENTER, after=0, line=1.0)
    add_textbox(slide, x0 + pw + gap_x, y, pw, 0.35, ["**Square: Nut assembly**"], size=19, align=PP_ALIGN.CENTER, after=0, line=1.0)
    y += 0.37
    slide.shapes.add_picture(f"{args.figs}/can_crop.png", Inches(x0), Inches(y), width=Inches(pw), height=Inches(ph))
    slide.shapes.add_picture(f"{args.figs}/square_crop.png", Inches(x0 + pw + gap_x), Inches(y), width=Inches(pw), height=Inches(ph))
    y += ph + 0.05
    add_textbox(slide, R_X, y, COL_W, 0.35, ["Example rollouts of the frozen diffusion policy in simulation."],
                size=18, color=MUTED, align=PP_ALIGN.CENTER, after=0, line=1.0)
    y += 0.4

    lw = 9.3
    lh = lw / (10.9 / 5.6)
    slide.shapes.add_picture(f"{args.figs}/critic_ladder_early.png", Inches(R_X + (COL_W - lw) / 2), Inches(y),
                             width=Inches(lw), height=Inches(lh))
    y += lh + 0.05
    add_table(slide, R_X, y, COL_W, [
        ["", "Can", "", "", "Square", "", ""],
        ["Critic", "min", "early AUC", "129k", "min", "early AUC", "127k"],
        ["DSRL baseline (n=5)", "0.24", "0.41±0.03", "0.53±0.06", "0.23", "0.40±0.01", "0.41±0.06"],
        ["IQL (n=5 / 3)", "0.34", "0.49±0.02", "0.56±0.02", "0.37", "0.44±0.01", "0.56±0.02"],
        ["TD (n=3)", "0.39", "0.47±0.03", "0.56±0.03", "0.44", "0.49±0.02", "0.42±0.06"],
        ["CQL-style (n=3)", "0.35", "0.48±0.02", "0.58±0.05", "0.37", "0.47±0.01", "0.33±0.08"],
        ["Cal-QL-style (n=3)", "0.34", "0.43±0.02", "0.51±0.05", "0.40", "0.46±0.03", "0.42±0.05"],
    ], col_w=[2.4, 1.15, 1.6, 1.55, 1.15, 1.6, 1.45], row_h=[0.45, 0.5, 0.46, 0.46, 0.46, 0.46, 0.46],
        size=17, header_rows=2, merges=[(0, 1, 0, 3), (0, 4, 0, 6)])
    y += 0.45 + 0.5 + 0.46 * 5 + 0.25

    place(one("TextBox 157"), R_X, y, COL_W, 0.65); y += 0.68        # "Discussion"
    place(one("TextBox 158"), R_X, y, COL_W, 2.75)
    set_text(one("TextBox 158"), [
        "**GapReach2D.** Cal-QL reaches 50% success in 2.7× fewer online steps; plain CQL is slower than no "
        "pretraining, its Q collapsed to a flat −7.7.",
        "**Can · Square.** Every pretrained critic raises the seed-mean floor of the early dip (0.24 → 0.34–0.44); "
        "Cal-QL-style adds no consistent benefit over IQL (seed-matched early AUC −0.08 ± 0.02 on Can, late "
        "−0.15 ± 0.06 on Square). TD leads early on Square, IQL late. Limits: 3–5 seeds; critic-only adaptations.",
        "Auxiliary ablations suggest that online entropy settings and demonstration replay also affect transition "
        "performance. Future work will examine their interaction with critic pretraining and the persistence of "
        "calibration during online learning.",
    ], size=16.5, after=4, line=1.1)
    right_bottom = y + 2.75

    # ============================ conclusion bar ==========================
    top = max(left_bottom, right_bottom) + 0.12
    bar_h = 1.0
    bar = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(L_X), Inches(top), Inches(PAGE_W - 2 * L_X), Inches(bar_h))
    bar.adjustments[0] = 0.15
    bar.fill.solid()
    bar.fill.fore_color.rgb = TINT
    bar.line.color.rgb = TEAL
    bar.line.width = Pt(2.25)
    bar.shadow.inherit = False
    set_text(bar, [
        "**Conclusion.** Offline RL-based critic pretraining can improve the offline-to-online transition in DSRL, "
        "with benefits depending on the task and evaluation metric.",
        "Cal-QL improves sample efficiency on GapReach2D, but its additional gains are inconsistent on Can and Square.",
    ], size=20, color=NAVY, line=1.05, after=2, align=PP_ALIGN.CENTER)
    bar.text_frame.vertical_anchor = MSO_ANCHOR.MIDDLE
    print("left ends %.2f, right ends %.2f, bar %.2f-%.2f (page %.2f)" % (left_bottom, right_bottom, top, top + bar_h, PAGE_H))

    prs.save(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
