"""
Setup para instalação do Business-RL.
Framework de Reinforcement Learning para Decisões de Negócio.
"""

from setuptools import setup, find_packages
import os

# Lê o README
readme_path = os.path.join(os.path.dirname(__file__), "README.md")
with open(readme_path, "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="business-rl",
    version="0.1.0",
    author="Business-RL Team",
    description="Framework customizado de RL para construir modelos ultra complexos de forma intuitiva",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cbaracho200/Reinforcement_Learning_version_000",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
        "numpy>=1.19.0",
        "gymnasium>=0.26.0",
        "flask>=2.0.0",
        "flask-cors>=3.0.0",
        "plotly>=5.0.0",
        "pyyaml>=5.4.0",
        "tqdm>=4.60.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0",
            "flake8>=3.9.0",
            "mypy>=0.900",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
