import os
import subprocess
import sys

# Usage on Colab via launcher wrapper: sets EXP_QUEUE env/file
exp_list = os.environ.get("EXP_QUEUE", "").split(",")
exp_list = [e.strip() for e in exp_list if e.strip()]
if not exp_list and os.path.exists("/content/exp_queue.txt"):
    exp_list = [l.strip() for l in open("/content/exp_queue.txt") if l.strip()]
if not exp_list:
    print("ERROR: No experiments specified in /content/exp_queue.txt", file=sys.stderr)
    sys.exit(2)

rc_path = "/content/exp_runner.rc"
log_path = "/content/exp_runner.log"
if os.path.exists(rc_path):
    os.remove(rc_path)

train_tasks = os.environ.get("TRAIN_TASKS", "50")
eval_tasks = os.environ.get("EVAL_TASKS", "50")
seq = " && ".join(
    f"EXP_NAME={e} TRAIN_TASKS={train_tasks} EVAL_TASKS={eval_tasks} bash /content/run_experiment.sh"
    for e in exp_list
)
cmd = (
    "bash -lc 'sed -i \"s/\\r$//\" /content/run_experiment.sh /content/shop_innovations.py /content/restore_from_hf.sh 2>/dev/null || true; "
    f"({seq}) > {log_path} 2>&1; rc=$?; echo $rc > {rc_path}; exit $rc'"
)
proc = subprocess.Popen(
    ["bash", "-c", cmd],
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    start_new_session=True,
)
print(f"LAUNCHED_EXP_QUEUE pid={proc.pid} queue={exp_list} train_tasks={train_tasks} eval_tasks={eval_tasks}")
