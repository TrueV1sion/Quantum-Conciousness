from setuptools import setup, find_packages

setup(
    name="quantum_consciousness",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "torchaudio>=0.9.0",
        "transformers>=4.5.0",
        "networkx>=2.6.0",
        "matplotlib>=3.4.0",
        "numpy>=1.19.0"
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A Meta-Cognitive Pipeline for quantum consciousness processing",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/quantum-consciousness",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.8",
) 