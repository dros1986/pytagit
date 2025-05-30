# setup.py

from setuptools import setup, find_packages

setup(
    name="pytagit",
    version="0.1.0",
    author="Il tuo nome",
    author_email="tuo@email.com",
    description="Interactive tool for image tagging with the human in the loop",
	keywords="image tagging, interactive tool, human in the loop, pytorch, PyQt6, t-SNE, CNN training",
	license="MIT",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/dros1986/pytagit",
    packages=find_packages(),  # Cerca automaticamente i moduli Python
    include_package_data=True,
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.23",
        "PyQt6>=6.5",
        "scikit-learn>=1.3",
        "matplotlib>=3.7",
        "albumentations>=1.3",
        "einops>=0.6",
        "opencv-python>=4.8",
        "Pillow>=9.5",
        "tqdm>=4.65",
        "regex>=2023",
        "pandas>=2.0",
        "transformers>=4.50",
        "pytorch-lightning>=2.5",
        "openTSNE>=1.0",
        "seaborn>=0.13",
        "rich>=12.0.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Environment :: X11 Applications :: Qt",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'pytagit=pytagit.main:main',  # Consente di eseguire il tool da terminale
        ],
    },
)