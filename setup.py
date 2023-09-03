from setuptools import setup

setup(
    name="nucliseg",
    author="Tamás Gyenis",
    author_email="tamgyen@gmail.com",
    version="1.0",
    description="hybrid model for cell nucleus segmentation on stained slides.",
    url="https://github.com/tamgyen/nucliseg",
    packages=["nucliseg"],
    install_requires=[
        "torch>=0.10",
        "torchvision>=0.11",
        "pytorch-lightning>=1.5.10,<=1.9.4",
        "tqdm",
        "scikit-image",
    ],
)