"""Setup script for NetSeer — AI-Based Network Attack Forecasting."""

import os
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))


def read_requirements() -> list:
    with open(os.path.join(here, "requirements.txt")) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


setup(
    name="netseer",
    version="1.0.0",
    description="AI-Based Network Attack Forecasting using World Models",
    long_description=open(os.path.join(here, "README.md"), encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="NetSeer",
    license="MIT",
    packages=find_packages(exclude=("tests", "scripts", "data", "models", "results")),
    include_package_data=True,
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "netseer-train=scripts.train:main",
            "netseer-evaluate=scripts.evaluate:main",
            "netseer-generate-synthetic=scripts.generate_synthetic:main",
        ]
    },
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Security",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)