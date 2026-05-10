"""Setup configuration for PrimeTRADE package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="primetrade",
    version="0.1.0",
    author="JARVIS7786",
    description="Institutional-grade Bitcoin Sentiment & Trader Performance Prediction System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JARVIS7786/PrimeTRADE",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "lightgbm>=4.0.0",
        "mlflow>=2.8.0",
        "statsmodels>=0.14.0",
        "imbalanced-learn>=0.11.0",
        "plotly>=5.17.0",
        "pyyaml>=6.0",
        "python-dateutil>=2.8.2",
        "pytz>=2023.3",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
            "ipython>=8.14.0",
            "jupyter>=1.0.0",
        ],
    },
)
