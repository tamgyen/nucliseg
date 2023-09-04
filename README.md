# nucliseg
This repo contains a python package for instance segmentation of cell nuclei on pathology slides using a lightweight neural network and classical CV approaches.


## How to install

1. Clone the repo
2. ```cd <repo_root>```
3. Create a new conda environment using ```conda env create --file environment.yaml``` - this might take a while
4. Activate ```conda activate nucliseg```
5. Install package using ```cd <repo_root>``` then ```pip install .```
6. Done!

## How to use

Easiest way to try is to run ```python predict.py <source_path> <dest_path>```
