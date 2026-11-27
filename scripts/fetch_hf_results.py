#!/usr/bin/env python3
"""Fetch experiment evaluation JSONs and summaries from your Hugging Face Dataset cache repo."""

import json
import os
import sys
import urllib.request


def main() -> None:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token and os.path.isfile(".hf_token"):
        token = open(".hf_token", encoding="utf-8").read().strip()
    repo_id = os.environ.get("HF_CACHE_REPO", "").strip()
    if not token or not repo_id:
        print("Usage: set HF_TOKEN and HF_CACHE_REPO environment variables before running.", file=sys.stderr)
        sys.exit(2)

    headers = {"Authorization": f"Bearer {token}"}

    def get_json(url: str):
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=25) as r:
            return json.loads(r.read().decode("utf-8"))

    run_items = get_json(f"https://huggingface.co/api/datasets/{repo_id}/tree/main/runs?recursive=true")
    out_dir = os.environ.get("RESULTS_OUT_DIR", os.path.join(os.path.dirname(__file__), "..", "results_synced"))
    os.makedirs(out_dir, exist_ok=True)

    print(f"=== Synced Metrics from {repo_id} ===")
    for it in run_items:
        path = it["path"]
        if path.endswith("eval_results.json") or path.endswith("summary.json"):
            url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{path}"
            data = get_json(url)
            rel = path.replace("runs/", "").replace("/", "_")
            with open(os.path.join(out_dir, rel), "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            if path.endswith("eval_results.json"):
                print(f"\n[{path}]")
                print(json.dumps(data.get("metrics", {}), ensure_ascii=False, indent=2))
            elif path.endswith("summary.json"):
                print(
                    f"\n[{path}] train_sec={data.get('train_sec')} eval_sec={data.get('eval_sec')} "
                    f"gpu={data.get('gpu')} profile={data.get('vram_profile')}"
                )


if __name__ == "__main__":
    main()
