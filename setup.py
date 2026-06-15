"""Minimal setup.py to make the 'src' package installable in editable mode.

Run once from the project root:
    pip install -e .

After this, 'from src.data.ingest import ...' works from anywhere
(notebooks, tests, CLI scripts) without manual sys.path manipulation.
"""

from setuptools import setup, find_packages

setup(
    name="stp-natality",
    version="0.1.0",
    description="MLOps solution for newborn birth weight prediction",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    python_requires=">=3.11",
    install_requires=[
        "pandas>=2.1.0",
        "numpy>=1.26.0",
        "scipy>=1.11.0",
        "pyarrow>=14.0.0",
    ],
)
