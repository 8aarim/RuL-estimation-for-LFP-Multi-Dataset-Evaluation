"""Run steps 1-6 for one dataset (resume from a step with --from):

    python run_all.py matr
    python run_all.py hust --from s3_tune
"""
import subprocess
import sys

STEPS = ['s1_preprocess', 's2_benchmark', 's3_tune', 's4_final_model', 's5_shap', 's6_topk']

ds = sys.argv[1] if len(sys.argv) > 1 else sys.exit(__doc__)
start = STEPS.index(sys.argv[sys.argv.index('--from') + 1]) if '--from' in sys.argv else 0
for step in STEPS[start:]:
    print(f'\n===== {ds}: {step} =====', flush=True)
    subprocess.run([sys.executable, '-m', f'steps.{step}', ds], check=True)
