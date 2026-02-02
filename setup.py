from setuptools import setup, find_packages

setup(
    name="airbnb_finetune",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.30.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "jupyter>=1.0.0",
        "tqdm>=4.65.0",
        "pillow>=10.0.0",
        "datasets>=2.14.0",
    ],
    python_requires=">=3.8",
    author="Japulgarin",
    description="Deep learning fine-tuning for Airbnb listings price prediction",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
