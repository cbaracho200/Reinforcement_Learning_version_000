"""
Setup para instalação do Business-RL.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="business-rl",
    version="0.1.0",
    author="Business-RL Team",
    description="Framework de RL para decisões empresariais complexas",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/business-rl",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
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
    entry_points={
        "console_scripts": [
            "business-rl=business_rl.cli:main",
        ],
    },
)