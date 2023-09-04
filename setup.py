from setuptools import setup, find_packages

setup(
    name="nucliseg",
    author="Tamás Gyenis",
    author_email="tamgyen@gmail.com",
    version="1.0",
    description="hybrid model for cell nucleus segmentation on stained slides.",
    url="https://github.com/tamgyen/nucliseg",
    packages=find_packages(),
    include_package_data=True,
)
