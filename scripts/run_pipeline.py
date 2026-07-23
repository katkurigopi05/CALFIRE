"""
Run the full CALFIRE analysis pipeline in order, in one command.

Equivalent to running each script by hand:
    train_model.py -> threshold_analysis.py -> timeseries_analysis.py ->
    historical_trends.py -> build_dashboard_data.py

Stops at the first failure (subprocess exit code != 0) instead of pressing
on with stale/missing inputs for later steps. Does not run fetch_weather.py,
predict.py, or the R scripts — those need live internet access or specific
CLI args, not just "run me."

Usage: python scripts/run_pipeline.py
"""
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent

STEPS = [
    "train_model.py",
    "threshold_analysis.py",
    "timeseries_analysis.py",
    "historical_trends.py",
    "build_dashboard_data.py",
]


def main():
    start = time.time()
    for i, script in enumerate(STEPS, 1):
        print(f"\n{'=' * 60}\n[{i}/{len(STEPS)}] {script}\n{'=' * 60}")
        result = subprocess.run([sys.executable, str(SCRIPTS_DIR / script)])
        if result.returncode != 0:
            print(f"\nPipeline stopped: {script} exited with code {result.returncode}")
            sys.exit(result.returncode)

    elapsed = time.time() - start
    print(f"\n{'=' * 60}\nPipeline complete in {elapsed:.0f}s — {len(STEPS)}/{len(STEPS)} steps ok\n{'=' * 60}")


if __name__ == "__main__":
    main()
