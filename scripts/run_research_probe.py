"""Headroom probe for the hardened (web-verified) research suite.

Generates a single-shot report per task and scores it with the binary fact
rubric (checklist_score_text). If single-shot fails the hardened facts, there
is genuine headroom for the Type-3d fact-verification loop (unlike the original
suite, which was saturated at 0.945 single-shot under the rubric).
"""
import argparse
import json, logging
from pathlib import Path
from src.config import ModelConfig
from src.evaluation.checklist_judge import checklist_score_text
from src.utils.llm_client import get_client

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("research_probe"); logger.setLevel(logging.INFO)

SYS = ("You are a meticulous research analyst. Write a precise, factual report "
       "answering the prompt. State specific figures, dates, and names exactly. "
       "Do not hedge; commit to specific values.")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", default="data/hard_benchmarks/research/research_tasks_hard.json",
                        help="hardened research task JSON (default: %(default)s)")
    parser.add_argument("--out", default="results/hard_benchmarks/research_hard_probe.json",
                        help="output JSON path (default: %(default)s)")
    parser.add_argument("--model", default="claude-sonnet-4-20250514",
                        help="model id for the single-shot probe (default: %(default)s)")
    args = parser.parse_args()

    tasks = json.load(open(args.tasks))
    client = get_client()
    cfg = ModelConfig(provider=ModelConfig.claude_sonnet().provider,
                      model_id=args.model, max_tokens=4096, temperature=0.0)
    out = Path(args.out); records = []
    for t in tasks:
        resp = client.generate(config=cfg, system=SYS,
                               messages=[{"role": "user", "content": t["description"]}])
        res = checklist_score_text(resp.content, t["requirements"])
        viol = [v["index"] for v in res["verdicts"] if not v["satisfied"]]
        logger.info("%s: rubric=%.2f (%d/%d) wrong/missing facts=%s",
                    t["task_id"], res["score"], res["n_met"], res["n_total"], viol)
        records.append({"task_id": t["task_id"], "score": res["score"],
                        "n_met": res["n_met"], "n_total": res["n_total"],
                        "violations": viol, "report": resp.content})
        out.write_text(json.dumps(records, indent=2))
    import statistics as st
    logger.info("MEAN single-shot rubric on hardened facts = %.3f (%d/%d tasks perfect)",
                st.mean(r["score"] for r in records),
                sum(r["score"] >= 1 for r in records), len(records))


if __name__ == "__main__":
    main()
