from setuptools import setup, find_packages

setup(
    name="pandemic-simulation",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "tensorboard",
        "psutil",
        "pyyaml",
        "matplotlib",
        "pandas",
        "seaborn",
    ],
)