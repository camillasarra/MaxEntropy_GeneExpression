# Maximum entropy models for patterns of gene expression
<a href="https://www.arxiv.org/abs/2408.08037" style='vertical-align:middle; display:inline;'><img
							src="https://img.shields.io/badge/physics.bio--ph-arXiv%3A2408.08037-B31B1B.svg" class="plain" style="height:25px;" /></a>


## Inverse Ising problem 

Goal: find the maximum entropy model compatible with experimental means and correlations of binarized gene expression levels. <br>

Steps: <br>
- start from experimental raw counts x
- get binarized expression $\sigma = \pm 1$ depending on whether x>0 or x=0
- compute experimental means and correlations, and concatenate in a vector $f_{data} = (\langle\sigma_i\rangle, \langle\sigma_i\sigma_j\rangle) $
- define the Ising model $P(\sigma) = \frac{1}{Z} e^{-H(\sigma)}$, with $H(\sigma) = \sum_i h_i \sigma_i + \frac{1}{2} \sum_{ij}  J_{ij} \sigma_i\sigma_j$
- find h,J so that the model averages match the experimental averages (using gradient descent and MonteCarlo simulations)


## Experimental Data

Data and loading functions from: https://alleninstitute.github.io/abc_atlas_access/descriptions/MERFISH-C57BL6J-638850.html 

Experimental data paper: https://www.biorxiv.org/content/10.1101/2023.03.06.531121v1

##  FILES
- tutorial.ipynb

- function.py
- find_ising.py: given a file containing f_data returns the parameters of the ising model
- A_generate_data.ipynb: preprocess data for part C
- B_find_ising.ipynb: get the model, explore and check
- C_Figures.ipynb: generate paper figures


## Requirements
- numpy 
- matplotlib
- tensorflow

