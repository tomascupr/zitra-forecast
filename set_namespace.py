#!/usr/bin/env python3
"""Update namespaces across the Zitra v2 release bundle cleanly before publishing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def rewrite_namespace(hf_org: str, gh_org: str, org_display: str) -> None:
    targets = [
        ROOT / "README.md",
        ROOT / "hf-cards/Zitra-Titan-585M/README.md",
        ROOT / "hf-cards/Zitra-Apex-585M/README.md",
        ROOT / "hf-cards/Zitra-209M-v2/README.md",
        ROOT / "hf-space/app.py",
        ROOT / "hf-space/requirements.txt",
        ROOT / "leaderboard-prs/fev-bench/models/zitra/model.py",
        ROOT / "leaderboard-prs/fev-bench/models/zitra/requirements.txt",
        ROOT / "leaderboard-prs/fev-bench/models_yaml_snippet.yaml",
        ROOT / "leaderboard-prs/gift-eval/notebooks/gift_eval.py",
    ]
    for p in targets:
        if not p.exists():
            continue
        txt = p.read_text()
        txt = txt.replace("https://huggingface.co/rohlik/", f"https://huggingface.co/{hf_org}/")
        txt = txt.replace("https://huggingface.co/spaces/rohlik/", f"https://huggingface.co/spaces/{hf_org}/")
        txt = txt.replace('"rohlik/Zitra-', f'"{hf_org}/Zitra-')
        txt = txt.replace('os.environ.get("ZITRA_HF_ORG", "rohlik")', f'os.environ.get("ZITRA_HF_ORG", "{hf_org}")')
        txt = txt.replace("github.com/tomascupr/zitra-forecast", f"github.com/{gh_org}/zitra-forecast")
        txt = txt.replace("raw.githubusercontent.com/tomascupr/zitra-forecast", f"raw.githubusercontent.com/{gh_org}/zitra-forecast")
        txt = txt.replace("organization: Rohlik / Zitra", f"organization: {org_display}")
        p.write_text(txt)

    for m in ["Zitra-Titan-585M", "Zitra-209M-v2"]:
        cfg_p = ROOT / "leaderboard-prs/gift-eval/results" / m / "config.json"
        if cfg_p.exists():
            cfg = json.loads(cfg_p.read_text())
            cfg["model_link"] = f"https://huggingface.co/{hf_org}/{m}"
            cfg["code_link"] = f"https://github.com/{gh_org}/zitra-forecast/blob/main/leaderboard-prs/gift-eval/notebooks/gift_eval.py"
            cfg["org"] = org_display
            cfg_p.write_text(json.dumps(cfg, indent=4) + "\n")

    for csv_p in (ROOT / "leaderboard-prs/fev-bench/benchmarks/fev_bench/results").glob("*.csv"):
        txt = csv_p.read_text()
        txt = txt.replace('"rohlik/Zitra-', f'"{hf_org}/Zitra-')
        csv_p.write_text(txt)

    print(f"Updated all release files to HF org='{hf_org}', GitHub org='{gh_org}', display='{org_display}'.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-org", required=True)
    ap.add_argument("--gh-org", required=True)
    ap.add_argument("--org-display", required=True)
    args = ap.parse_args()
    rewrite_namespace(args.hf_org, args.gh_org, args.org_display)
