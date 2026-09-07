"""Fill the symposium PPTX template (A1 portrait) with the poster draft.

    .venv/bin/python poster/build_poster.py --template <template.pptx> --figs <dir> --out <draft.pptx>

Keeps the template header (title box, names, logos, rule); removes the example
boxes; places sections, figures and tables at millimetre coordinates given in
LAYOUT. Also writes <out>_layout.png, a wireframe of the boxes, since a PPTX
cannot be rendered here. Text with [tonight] marks values still being run.
"""
import argparse, os
from pptx import Presentation
from pptx.util import Mm, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

NAVY = RGBColor(0x1F, 0x3A, 0x6E); GREY = RGBColor(0x40, 0x40, 0x40); RED = RGBColor(0xB0, 0x20, 0x20)
KEEP_IDS = {132, 133, 134, 136, 36, 3}          # rule, name, department, two logos, title
LX, LW, RX, RW = 10, 276, 300, 284             # columns, mm

TITLE = "Calibrated critic pretraining for offline-to-online diffusion steering: what it buys, and what bounds it"

ABSTRACT = ("DSRL fine-tunes a frozen diffusion policy by learning the noise it is fed. Going from demonstrations to "
            "online interaction, success first falls below the pretrained policy (0.50 → 0.24 on robomimic Can) and "
            "recovers only after ~60k online steps. We ask whether a better-prepared critic stabilizes this handoff: "
            "a controlled 2-D study (Cal-QL) and a robot ladder TD / IQL / CQL-style / Cal-QL-style on Can and Square, "
            "plus demonstration replay. Critic initialization changes the depth of the dip and replay speeds recovery, "
            "but neither moves its timing; 90 runs trace the trigger to SAC's automatic temperature collapsing the "
            "policy's entropy (17 → 0) in the first 15k steps. Holding entropy above ~10 removes the dip on both tasks.")

SETTING = ("Setting. DSRL-NA: frozen π_dp, SAC actor π_W picks the noise w ∈ [−1.5, 1.5]^28; Q_A (action space, from rewards) "
           "and Q_W (noise space, distilled from Q_A). Can: rollout 24k env steps, then 1.25 updates/step, 100-episode evals every 5k.\n"
           "Ladder (critic at step 0). baseline: random. TD: r + γ Q̄_A(s′, π_dp(s′, w′)), w′ ~ N(0,I). IQL: expectile V, target r + γV(s′). "
           "CQL-style: IQL + α·(logsumexp{Q_A(s, π_dp(s,w_k)), Q_A(s,a_data)} − Q_A(s,a_data)). Cal-QL-style: same with the sampled Q "
           "floored at the demonstration return G. All distilled into Q_W for 25k steps; online: no demo replay, same α, random actor.\n"
           "† Robomimic uses an actor-free, anchored CQL-style objective: an adaptation, not an exact reproduction of standard CQL.")

P1_CAP = ("[colleague] GapReach2D: TD warm-start does not beat the baseline (T50 2,099 vs 2,160); CQL over-penalizes "
          "(AUC 0.674, Q ≈ −9 against returns in [0,1]); Cal-QL keeps the scale and is fastest (AUC 0.884, T80 2,929 vs 6,320). "
          "Conservative values help only when their scale remains calibrated.")
P2_CAP = ("Can, online-relative x axis (initial evaluation at 0), mean ± SE over seeds, early window 0–76k. "
          "IQL / warm-up-critic shallow the dip (min 0.24 → 0.33, AUC 0.41 → 0.49/0.51) but not its timing (≈35k) nor the final (129k 0.53–0.61). "
          "TD / CQL-style / Cal-QL-style: [tonight]. Offline diagnostic: E_w Q(s, π_dp(s,w)) − G = −20 (Can), −15 (Square): the prior policy "
          "is worse than the demonstrations, so Cal-QL's floor (policy ≥ behaviour) does not bind here.")
P3_CAP = ("Demo prefill (upstream `load_offline_data`, off in the published config) lifts early AUC 0.41 → 0.68 and 129k success 0.53 → 0.86, "
          "yet the momentary minimum is deeper (0.09). Explicit fixed / linear schedules do not beat prefill; IQL + prefill does not add (0.60).")
TRANSITION = ("Critic initialization changes dip severity, and replay accelerates recovery — "
              "but neither explains why the collapse occurs at the same moment.   What triggers the dip?  →  ACT II")

P4_CAP = ("With target entropy 0, auto-α drives the policy's entropy 17 → 0 within 15k steps (log π: −17@2k → −6@6k → 0@16k online); "
          "every auto-α condition bottoms in that window (Can ≈11k online, Square ≈10k). Q-scale levers do not move it: reward ×0.25–×2 and "
          "hard backup shift the first drop by < 2.5k (refuted). Dose–response, both tasks — entropy floor 0 / ~6 / ~11 → "
          "Can min 0.24 (recovery 84k) / 0.22 (34k) / no dip; Square 0.21 (62k) / 0.28 (52k) / no dip.")
P5_CAP = ("Fixed α = 0.3 (entropy 10–14) or target entropy 12 with initial α 0.3 (tent12i): no dip (min 0.47 / 0.46 above the π_dp line 0.405), "
          "AUC 0.66 / 0.59, 129k 0.74 / 0.70 vs baseline 0.41 / 0.53. Cal-QL-style + tent12i: [tonight].")
P6_CAP = ("Square (γ = 0.999): holding entropy removes the 42k dip (0.41 vs 0.21) but loses the late game (127k 0.40 vs 0.47): the α needed "
          "to hold entropy (0.4 → 15) inflates the critic target's entropy bonus (Q_W 300–600 → 23,000). Causal test: target entropy 12 with the "
          "bonus removed from the critic target — [tonight, 3 seeds].")
DISCUSSION = ("Critic calibration sets the starting point; the online trigger is the temperature. Limits: n = 3–5, eval noise ±0.1, Square "
              "seed variance; α scale on Square; offline pretraining is an actor-free CQL adaptation. Related: LP-DS (trust region on the noise, "
              "same collapse, different lever), TES-SAC (entropy schedules). Next: entropy-target schedule; critic-bonus-free target on Square.")
REFS = ("Wagenmaker et al. 2025 (DSRL, CoRL). Nakamoto et al. 2023 (Cal-QL, NeurIPS). Kumar et al. 2020 (CQL). Kostrikov et al. 2022 (IQL). "
        "Simsir & Oguz 2026 (LP-DS). Haarnoja et al. 2018 (SAC).")

LAYOUT = [  # (kind, x, y, w, h, payload, size)
    ("head", LX, 118, LW, 12, "Abstract", 34),
    ("body", LX, 131, LW, 46, ABSTRACT, 17),
    ("banner", LX, 180, LW, 13, "ACT I — Can better value estimates stabilize the O2O handoff?", 26),
    ("body", LX, 195, LW, 44, SETTING, 14),
    ("head", LX, 242, LW, 12, "1. Controlled clue: calibration — not pretraining alone — accelerates fine-tuning", 22),
    ("placeholder", LX, 256, LW, 108, "GapReach2D success curves (colleague) + inset: offline calibration violation", 18),
    ("cap", LX, 366, LW, 26, P1_CAP, 14),
    ("head", LX, 396, LW, 12, "2. Stress test: does calibration transfer to high-dimensional manipulation?", 22),
    ("fig", LX, 410, 200, None, "success_critic_early.png", None),
    ("placeholder", LX + 204, 410, 72, 114, "Can · Square summary dots: early AUC, dip depth [tonight]", 14),
    ("cap", LX, 528, LW, 40, P2_CAP, 14),
    ("head", LX, 572, LW, 12, "3. Persistent demonstrations are another anchor: replay is powerful, schedules are not", 22),
    ("fig", LX, 586, LW, None, "replay_bars.png", None),
    ("cap", LX, 718, LW, 26, P3_CAP, 14),
    ("transition", LX, 748, LW, 22, TRANSITION, 18),
    ("head", LX, 776, LW, 12, "Discussion", 26),
    ("body", LX, 790, LW, 44, DISCUSSION, 14),
    ("head", RX, 118, RW, 12, "Table 1 — Can, critic ladder (online-relative, early window 0–76k)", 20),
    ("table", RX, 132, RW, 62, None, 13),
    ("banner", RX, 200, RW, 13, "ACT II — What triggers the dip? Entropy collapse", 26),
    ("head", RX, 217, RW, 12, "4. The dip is auto-α collapsing the policy's entropy; Q scale is not the trigger", 22),
    ("fig", RX, 231, RW, None, "success_sweep_early.png", None),
    ("cap", RX, 395, RW, 40, P4_CAP, 14),
    ("head", RX, 439, RW, 12, "5. Holding entropy removes the dip (Can)", 22),
    ("fig", RX, 453, RW, None, "success_adaptive_early.png", None),
    ("cap", RX, 617, RW, 26, P5_CAP, 14),
    ("head", RX, 647, RW, 12, "6. Square: no dip, but a late-game trade-off — and the causal test", 22),
    ("fig", RX, 661, 214, None, "success_square_early.png", None),
    ("placeholder", RX + 218, 661, 66, 122, "tent12 + hard backup: 42k / 127k [tonight]", 14),
    ("cap", RX, 786, RW, 30, P6_CAP, 14),
    ("body", RX, 820, RW, 16, REFS, 11),
]
TABLE = [["method", "dip min", "T80 (online)", "early AUC", "129k"],
         ["baseline (n=5)", "0.24", "never (0/5)", "0.41 ± 0.03", "0.53"],
         ["TD (FQE of prior)", "[tonight]", "", "", ""],
         ["IQL (n=5)", "0.34", "never (0/5)", "0.49 ± 0.02", "0.56"],
         ["CQL-style †", "[tonight]", "", "", ""],
         ["Cal-QL-style †", "[tonight]", "", "", ""]]


def add_text(slide, x, y, w, h, text, size, bold=False, color=GREY, align=PP_ALIGN.LEFT, fill=None, italic=False):
    box = slide.shapes.add_textbox(Mm(x), Mm(y), Mm(w), Mm(h))
    tf = box.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = Mm(1.5); tf.margin_top = tf.margin_bottom = Mm(0.8)
    if fill is not None:
        box.fill.solid(); box.fill.fore_color.rgb = fill
    for i, para in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run(); r.text = para
        r.font.size = Pt(size); r.font.bold = bold; r.font.italic = italic; r.font.color.rgb = color; r.font.name = "Arial"
    return box


def build(template, figs, out):
    prs = Presentation(template)
    slide = prs.slides[0]
    for sh in list(slide.shapes):
        if sh.shape_id not in KEEP_IDS:
            sh._element.getparent().remove(sh._element)
    for sh in slide.shapes:
        if sh.shape_id == 3:
            tf = sh.text_frame; p = tf.paragraphs[0]
            for r in list(p.runs)[1:]:
                r._r.getparent().remove(r._r)
            p.runs[0].text = TITLE; p.runs[0].font.size = Pt(44)
    boxes = []
    for kind, x, y, w, h, payload, size in LAYOUT:
        if kind == "head":
            add_text(slide, x, y, w, h, payload, size, bold=True, color=NAVY)
        elif kind == "banner":
            add_text(slide, x, y, w, h, payload, size, bold=True, color=RGBColor(0xFF, 0xFF, 0xFF), fill=NAVY)
        elif kind in ("body", "cap"):
            add_text(slide, x, y, w, h, payload, size, color=GREY)
        elif kind == "transition":
            add_text(slide, x, y, w, h, payload, size, bold=True, color=RED)
        elif kind == "placeholder":
            add_text(slide, x, y, w, h, payload, size, italic=True, color=RED, fill=RGBColor(0xF2, 0xF2, 0xF2), align=PP_ALIGN.CENTER)
        elif kind == "fig":
            path = os.path.join(figs, payload)
            pic = slide.shapes.add_picture(path, Mm(x), Mm(y), width=Mm(w))
            h = pic.height / 36000.0
        elif kind == "table":
            rows, cols = len(TABLE), len(TABLE[0])
            tbl = slide.shapes.add_table(rows, cols, Mm(x), Mm(y), Mm(w), Mm(h)).table
            for i, row in enumerate(TABLE):
                for j, val in enumerate(row):
                    cell = tbl.cell(i, j); cell.text = val
                    for p in cell.text_frame.paragraphs:
                        for r in p.runs:
                            r.font.size = Pt(size); r.font.bold = (i == 0 or j == 0); r.font.name = "Arial"
                    cell.margin_left = cell.margin_right = Mm(1); cell.margin_top = cell.margin_bottom = Mm(0.5)
        boxes.append((kind, x, y, w, h, str(payload)[:40]))
    prs.save(out)
    # wireframe preview
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt; import matplotlib.patches as pa
    fig, ax = plt.subplots(figsize=(5.94, 8.41)); ax.set_xlim(0, 594); ax.set_ylim(841, 0); ax.set_aspect("equal")
    ax.add_patch(pa.Rectangle((0, 0), 594, 110, fill=False, ls=":")); ax.text(297, 55, "header (kept)", ha="center", fontsize=6)
    col = {"head": "#1F3A6E", "banner": "#1F3A6E", "body": "#999", "cap": "#bbb", "fig": "#2a9d8f", "placeholder": "#e76f51", "table": "#8a5", "transition": "#b02020"}
    for kind, x, y, w, h, name in boxes:
        ax.add_patch(pa.Rectangle((x, y), w, h, fill=(kind in ("fig", "placeholder", "banner")), alpha=0.35 if kind in ("fig", "placeholder", "banner") else 1, color=col[kind], lw=0.6))
        ax.text(x + 1, y + 3.5, "%s: %s" % (kind, name), fontsize=3.2, va="top")
    ax.axis("off"); fig.tight_layout(); fig.savefig(os.path.splitext(out)[0] + "_layout.png", dpi=220); plt.close(fig)
    print("wrote", out, "and layout preview")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--template", required=True); ap.add_argument("--figs", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); build(a.template, a.figs, a.out)
