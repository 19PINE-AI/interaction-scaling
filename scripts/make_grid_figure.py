"""Build a NeurIPS-style single-shot vs reviewed comparison grid for the paper."""
import base64, io, json
from PIL import Image, ImageDraw, ImageFont

D = {r["task_id"]: r for r in json.load(open("results/hard_benchmarks/paperfig_geom_run1.json"))}
# pick clean before/after pairs (ss defects > 0, rv == 0), diverse architectures
ROWS = [
    ("figure_007", "Latent Diffusion\nU-Net"),
    ("figure_003", "GAN\n(Gen. + Disc.)"),
    ("figure_014", "Seq2Seq\n+ Attention"),
    ("figure_006", "Mask R-CNN"),
]
THUMB_W = 760
PAD = 18
LABEL_H = 34
HEADER_H = 44
ROWLABEL_W = 220


def load(b64):
    im = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    return im.resize((THUMB_W, round(im.height * THUMB_W / im.width)))


def font(sz, bold=False):
    for p in ([
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]):
        try: return ImageFont.truetype(p, sz)
        except Exception: pass
    return ImageFont.load_default()


thumbs = [(load(D[t]["ss_screenshot_b64"]), load(D[t]["rv_screenshot_b64"]), lbl, D[t])
          for t, lbl in ROWS]
TH = max(max(s.height, r.height) for s, r, _, _ in thumbs)
W = ROWLABEL_W + 2 * THUMB_W + 3 * PAD
H = HEADER_H + len(ROWS) * (TH + LABEL_H + PAD) + PAD
canvas = Image.new("RGB", (W, H), "white")
dr = ImageDraw.Draw(canvas)
fb, fr, fs = font(26, True), font(20, True), font(17)

# column headers
dr.text((ROWLABEL_W + PAD + THUMB_W // 2 - 90, 10), "Single-shot", font=fb, fill=(180, 40, 40))
dr.text((ROWLABEL_W + 2 * PAD + THUMB_W + THUMB_W // 2 - 70, 10), "Reviewed", font=fb, fill=(30, 130, 60))

y = HEADER_H
for ss, rv, lbl, rec in thumbs:
    # row label (multi-line, centered vertically)
    nlines = lbl.count("\n") + 1
    dr.multiline_text((PAD, y + TH // 2 - 14 * nlines), lbl, font=fr, fill=(20, 20, 30), spacing=6)
    # thumbnails with colored borders
    x1 = ROWLABEL_W + PAD
    x2 = ROWLABEL_W + 2 * PAD + THUMB_W
    canvas.paste(ss, (x1, y)); canvas.paste(rv, (x2, y))
    dr.rectangle([x1 - 2, y - 2, x1 + THUMB_W + 1, y + ss.height + 1], outline=(180, 40, 40), width=3)
    dr.rectangle([x2 - 2, y - 2, x2 + THUMB_W + 1, y + rv.height + 1], outline=(30, 130, 60), width=3)
    # defect annotations
    ssn, rvn = rec["ss_n_defects"], rec["rv_n_defects"]
    dr.text((x1 + 8, y + ss.height + 6), f"{ssn} geometric defect{'' if ssn == 1 else 's'}", font=fs, fill=(180, 40, 40))
    rv_lbl = f"{rvn} defect{'' if rvn == 1 else 's'}" + (" (clean)" if rvn == 0 else "")
    dr.text((x2 + 8, y + rv.height + 6), rv_lbl, font=fs, fill=(30, 130, 60))
    y += TH + LABEL_H + PAD

canvas.save("paper/figures/comparison_grid.png", "PNG", optimize=True)
print("wrote paper/figures/comparison_grid.png", canvas.size)
