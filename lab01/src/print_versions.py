import sys
from importlib.metadata import version

packages = [
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "matplotlib",
    "seaborn",
    "torch",
    "torchvision",
    "torchinfo",
    "thop",
    "onnx",
    "onnxruntime",
    "mlflow",
    "memory-profiler",
    "psutil",
    "codecarbon",
    "fastapi",
    "uvicorn",
    "pytest",
    "httpx",
    "locust",
    "requests",
    "pyarrow",
    "joblib",
    "tqdm",
]

print(f"Python: {sys.version.split( )[0]}")

for package in packages:
    print(f"{package}: {version(package)}")
