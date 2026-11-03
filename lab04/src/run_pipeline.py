from pathlib import Path
import subprocess
import sys


def main():
    src_dir = Path(__file__).resolve().parent
    repo_dir = src_dir.parents[1]

    scripts = [
        "ingest.py",
        "validate_data.py",
        "train_baselines.py",
        "quality_check.py",
        "hash_processed.py",
    ]

    for script in scripts:
        print(f"\n=== Running {script} ===", flush=True)
        subprocess.run(
            [sys.executable, "-u", str(src_dir / script)],
            cwd=repo_dir,
            check=True,
        )

    print("\nLab 4 pipeline completed successfully.", flush=True)


if __name__ == "__main__":
    main()
