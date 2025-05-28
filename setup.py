#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name="newaibench",
    version="0.1.0",
    description="NewAIBench Framework",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
)
