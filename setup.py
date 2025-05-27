from setuptools import setup, find_packages

setup(
    name="liza",
    version="0.1.0",
    description="LineariZe Attention injection (LiZA)",
    author="Ton Nom",
    packages=find_packages(),
    install_requires=[
        "torch",
        "einops",
        "pydantic",
        "flash-linear-attention"
    ],
    python_requires=">=3.8",
)
