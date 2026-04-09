"""
Setup script for the Multimodal RAG Financial Document Analyzer.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="multimodal-rag-finance-analyzer",
    version="1.0.0",
    author="Financial RAG Team",
    description="A production-ready Multimodal RAG system for financial document analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/multimodal-rag-finance-analyzer",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "finance-analyzer=backend.main:main",
        ],
    },
)
