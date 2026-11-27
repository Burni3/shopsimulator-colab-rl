import glob
import json
import os
import subprocess

rc_path = "/content/exp_runner.rc"
log_path = "/content/exp_runner.log"

rc = open(rc_path).read().strip() if os.path.exists(rc_path) else "RUNNING"
print(f"STATUS: rc={rc}")

try:
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader"],
        text=True,
        timeout=10,
    ).strip()
    print(f"GPU: {gpu}")
except Exception as e:
    print(f"GPU: err={e}")

summaries = sorted(glob.glob("/content/*_summary.json"))
if summaries:
    print("COMPLETED_SUMMARIES:")
    for sp in summaries:
        d = json.load(open(sp, encoding="utf-8"))
        m = d.get("metrics", {})
        print(
            f"  - {d['experiment']}: train={d['train_sec']}s eval={d['eval_sec']}s "
            f"r_loose={m.get('r_loose')} r_hard={m.get('r_hard')} r_success={m.get('r_success')} "
            f"r_opt={m.get('r_option')} r_price={m.get('r_price')}"
        )

if os.path.exists(log_path):
    with open(log_path, encoding="utf-8", errors="replace") as f:
        tail = f.readlines()[-18:]
    print("LOG_TAIL:")
    for line in tail:
        print("  " + line.rstrip())
