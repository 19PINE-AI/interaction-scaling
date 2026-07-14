"""Quick probe: can Sonnet 4 single-shot draw real academic-paper architecture
figures (Transformer, U-Net, ViT) cleanly, or do they overlap/clip? Renders +
rubric-scores + saves screenshots for visual audit."""
import argparse
import base64, json
from src.config import ModelConfig
from src.evaluation.checklist_judge import checklist_score
from src.rendering.browser import BrowserRenderer
from src.utils.code_utils import extract_code
from src.utils.llm_client import get_client
from scripts.run_diagram_benchmark import DIAGRAM_SYSTEM_PROMPT

TASKS = [
  {"task_id":"paper_transformer","requirements":[
    "Both the encoder stack (left) and decoder stack (right) are drawn as labelled boxes, fully visible, no clipping",
    "Every sublayer box (Multi-Head Attention, Add & Norm, Feed Forward, Masked Multi-Head Attention, etc.) has its label fully inside the box with no text overflow",
    "No two boxes overlap each other",
    "The 'Nx' multiplier annotation appears beside each stack and does not overlap a box",
    "Positional Encoding is shown being combined (circle-plus) with the input/output embeddings, and the symbol does not overlap text",
    "Residual/skip arrows are drawn and do not pass through any box that is not their endpoint",
    "Cross-attention arrows go from the encoder output into the decoder's cross-attention sublayer and are visually distinguishable",
    "The top shows Linear then Softmax then 'Output Probabilities', stacked and labelled, none clipped at the top edge",
    "No arrow label or box is within 10px of the viewport edge and no scrollbar appears",
    "All text is horizontal and legible (no overlapping labels anywhere in the figure)"],
   "description":"Reproduce the canonical Transformer architecture figure from 'Attention Is All You Need' (Vaswani et al. 2017), Figure 1, as a single 1920x1080 self-contained HTML file using inline CSS + inline SVG. Show the full encoder (left) and decoder (right) stacks each wrapped with an 'Nx' multiplier. Encoder sublayers bottom-to-top: Input Embedding (+) Positional Encoding, Multi-Head Attention, Add & Norm, Feed Forward, Add & Norm. Decoder sublayers: Output Embedding (+) Positional Encoding, Masked Multi-Head Attention, Add & Norm, Multi-Head Attention (cross, fed by encoder output), Add & Norm, Feed Forward, Add & Norm. Above the decoder: Linear, Softmax, Output Probabilities. Draw residual connections and the encoder->decoder cross-attention arrows. Use the same overall visual structure as the paper."},
  {"task_id":"paper_unet","requirements":[
    "The contracting (encoder) path on the left and expanding (decoder) path on the right form a U shape, all blocks visible with no clipping",
    "Each conv block is a labelled box showing its channel/resolution annotation (e.g. '64', '128 28x28') fully inside the box",
    "There are at least 4 resolution levels on each side and they are vertically aligned by level across the two paths",
    "Horizontal skip-connection arrows connect each encoder level to the SAME-level decoder block and do not pass through intermediate blocks",
    "Down-sampling (max pool) and up-sampling (up-conv) arrows are drawn between levels and labelled",
    "No two blocks overlap, and no skip arrow overlaps a block label",
    "The channel-count numbers do not overlap each other or the block borders",
    "A legend explains the arrow types (conv, copy/skip, max-pool, up-conv) and does not overlap the figure",
    "Nothing is within 10px of the viewport edge and no scrollbar appears",
    "All text is legible with no overlapping labels"],
   "description":"Reproduce the U-Net architecture figure (Ronneberger et al. 2015) as a single 1920x1080 self-contained HTML file using inline CSS + inline SVG. Draw the contracting path (repeated conv + max-pool, channels doubling 64->128->256->512->1024) descending on the left, the expanding path (up-conv + conv, channels halving) ascending on the right, and the horizontal copy-and-crop skip connections joining matching levels. Annotate each stage with channel counts and feature-map sizes, and include a legend of the arrow types as in the paper."},
  {"task_id":"paper_vit","requirements":[
    "An input image split into a grid of equal-size patches is shown at the bottom-left, fully visible",
    "Each patch flows into a 'Linear Projection of Flattened Patches' box; the projection box label fits inside it",
    "Patch + Position embeddings are shown being added, with a learnable [class] token prepended, labelled, not overlapping",
    "The Transformer Encoder block is drawn (Norm, Multi-Head Attention, +, Norm, MLP, +) with every sublayer labelled inside its box and an 'L x' multiplier beside it",
    "An MLP Head box at the top maps the class token to output classes, labelled and not clipped",
    "Arrows connect patches->projection->encoder->MLP head without passing through unrelated boxes",
    "No two boxes overlap and no label overflows its box",
    "The patch-embedding sequence tokens are evenly spaced and aligned",
    "Nothing is within 10px of the viewport edge and no scrollbar appears",
    "All text is horizontal and legible with no overlapping labels"],
   "description":"Reproduce the Vision Transformer (ViT) architecture figure (Dosovitskiy et al. 2020), Figure 1, as a single 1920x1080 self-contained HTML file using inline CSS + inline SVG. Show: an image divided into patches; flattening + linear projection of patches; prepended [class] token; added position embeddings (0,1,2,...,9); the stack feeding a Transformer Encoder (expanded on the right showing Norm/Multi-Head Attention/MLP with residual adds and an 'L x' multiplier); and an MLP Head producing class output. Match the paper's layout."},
]

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="claude-sonnet-4-20250514",
                        help="model id that draws the figures (default: %(default)s)")
    parser.add_argument("--out", default="/tmp/paper_figures_probe.json",
                        help="output JSON path (default: %(default)s)")
    args = parser.parse_args()

    cfg = ModelConfig(provider=ModelConfig.claude_sonnet().provider,
                      model_id=args.model, max_tokens=8192, temperature=0.0)
    judge = ModelConfig.claude_sonnet()
    r = BrowserRenderer(); client = get_client()
    out = []
    for t in TASKS:
        client.reset_counters()
        resp = client.generate(config=cfg, system=DIAGRAM_SYSTEM_PROMPT,
                               messages=[{"role":"user","content":t["description"]}])
        html = extract_code(resp.content, "html")
        if not html.strip().startswith("<"): html = resp.content
        png = r.render_html(html); b64 = base64.b64encode(png).decode()
        open(f"/tmp/paper_{t['task_id']}.png","wb").write(png)
        res = checklist_score(b64, t["requirements"], model_config=judge, client=client)
        viol = [v for v in res["verdicts"] if not v["satisfied"]]
        print(f"{t['task_id']}: rubric={res['score']:.2f} ({res['n_met']}/{res['n_total']}) violations:")
        for v in viol: print(f"    - req{v['index']}: {v['evidence'][:100]}")
        out.append({"task_id":t["task_id"],"score":res["score"],"html":html,"b64":b64,"verdicts":res["verdicts"]})
    json.dump(out, open(args.out, "w"))

if __name__ == "__main__":
    main()
