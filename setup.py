from setuptools import setup, find_packages

setup(
    name="srini_einops",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
    ],
    author="Srini",
    description="Pure NumPy implementation of tensor operations",
    python_requires=">=3.7",
)