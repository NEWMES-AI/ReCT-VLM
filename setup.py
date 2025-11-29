"""
ReCT-VLM: Reasoning-Enhanced CT Vision-Language Model
Setup script for package installation
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Version
__version__ = "0.1.0"

setup(
    name="rect-vlm",
    version=__version__,
    author="NEWMES-AI",
    author_email="contact@newmes-ai.com",  # Update with actual email
    description="Reasoning-Enhanced CT Vision-Language Model for Multi-task Medical Image Understanding",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NEWMES-AI/ReCT-VLM",
    project_urls={
        "Bug Tracker": "https://github.com/NEWMES-AI/ReCT-VLM/issues",
        "Documentation": "https://github.com/NEWMES-AI/ReCT-VLM/docs",
        "Source Code": "https://github.com/NEWMES-AI/ReCT-VLM",
    },
    packages=find_packages(exclude=["tests", "docs", "examples"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.0.0",
            "flake8>=6.1.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
        ],
        "distributed": [
            "deepspeed>=0.12.0",
            "fairscale>=0.4.13",
        ],
        "export": [
            "onnx>=1.15.0",
            "onnxruntime>=1.16.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "rect-vlm-train=rect_vlm.training.train_multitask:main",
            "rect-vlm-eval=rect_vlm.scripts.evaluate:main",
            "rect-vlm-infer=rect_vlm.scripts.inference:main",
        ],
    },
    include_package_data=True,
    package_data={
        "rect_vlm": [
            "configs/*.yaml",
            "model/configs/*.yaml",
        ],
    },
    zip_safe=False,
    keywords=[
        "medical imaging",
        "computer vision",
        "natural language processing",
        "vision-language model",
        "multi-task learning",
        "CT scan",
        "radiology",
        "disease classification",
        "lesion localization",
        "report generation",
        "deep learning",
        "pytorch",
        "transformers",
    ],
)
