from setuptools import setup, find_packages

setup(
    name="research_classifier",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.0.0",
        "numpy>=1.20.0",
        "django>=4.0.0",
    ],
    extras_require={
        "test": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
        ],
    },
)
