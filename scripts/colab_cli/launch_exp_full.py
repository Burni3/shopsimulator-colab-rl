import os
import subprocess
import sys

exp_list = [l.strip() for l in open("/content/exp_queue.txt") if l.strip()] if os.path.exists("/content/exp_queue.txt") else []
if not exp_list:
    print("ERROR: No experiments in /content/exp_queue.txt", file=sys.stderr)
    sys.exit(2)

rc_path = "/content/exp_runner.rc"
log_path = "/content/exp_runner.log"
if os.path.exists(rc_path):
    os.remove(rc_path)

seq = " && ".join(
    f"EXP_NAME={e} TRAIN_TASKS=500 EVAL_TASKS=200 RUN_SUITE=full_rl500_v1 SAVE_INTERVAL=20 bash /content/run_experiment.sh"
    for e in exp_list
)
cmd = (
    "bash -lc 'sed -i \"s/\\r$//\" /content/run_experiment.sh /content/shop_innovations.py /content/restore_from_hf.sh /content/prune_vocab_qwen35.py 2>/dev/null || true; "
    f"({seq}) > {log_path} 2>&1; rc=$?; echo $rc > {rc_path}; exit $rc'"
)
proc = subprocess.Popen(
    ["bash", "-c", cmd],
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    start_new_session=True,
)
print(f"LAUNCHED_FULL_500_QUEUE pid={proc.pid} queue={exp_list} train_tasks=500 eval_tasks=200 suite=full_rl500_v1")
