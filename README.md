# Maximum entropy models for patterns of gene expression
<a href="https://www.arxiv.org/abs/2408.08037" style='vertical-align:middle; display:inline;'><img
							src="https://img.shields.io/badge/physics.bio--ph-arXiv%3A2408.08037-B31B1B.svg" class="plain" style="height:25px;" /></a>


This code reproduces the results of the paper [_Maximum entropy models for patterns of gene expression_](https://www.arxiv.org/abs/2408.08037) by Camilla Sarra, Leopoldo Sarra, Luca Di Carlo, Trevor GrandPre, Yaojun Zhang, Curtis G. Callan Jr., William Bialek.

## Inverse Ising problem 

**Goal**: find the maximum entropy model compatible with experimental means and correlations of binarized gene expression levels. <br>

Steps: <br>
- start from experimental raw counts x
- get binarized expression $\sigma = \pm 1$ depending on whether x>0 or x=0
- compute experimental means and correlations, and concatenate in a vector $f_{data} = (\langle\sigma_i\rangle, \langle\sigma_i\sigma_j\rangle)$
- define the Ising model $P(\sigma) = \frac{1}{Z} e^{-H(\sigma)}$, with $H(\sigma) = \frac{1}{d} \left(\sum_i h_i \sigma_i + \frac{1}{2} \sum_{ij}  J_{ij} \sigma_i\sigma_j \right)$
- find h,J so that the model averages match the experimental averages (using gradient descent and MonteCarlo simulations)


## Getting started

Install the requirements in the `requirements.txt` file with the command

```
pip install -r requirements.txt
```

Note: this code has been tested on python 3.8.18 and tensorflow 2.4.1. <br>
Note: this code uses the tensorflow package. It's recommended to install it manually according to the system specifics as described [here](https://www.tensorflow.org/install)


Please take a look at the `tutorial.ipynb` notebook for an example of how to find the Ising model parameters given the experimental averages.


Given a file `f_data_D_LABEL.dat`, where D is the number of variables and LABEL a label, you can get the Ising model parameters using for example
```
find_ising.py -d D -label LABEL -n_thermalize 2000 -n_steps 5000 -adam_step 1e-3
```

This will produce 
-  `q_D_LABEL.dat` with the concatenated Ising parameters (size $D+\frac{D(D-1)}{2}$, where the first D lines are the external fields $h_1, \ldots h_D$,  and the remaining are the interactions $J_{1,2}, ... J_{d-1, d}$ ). The parameters can be obtained using
```python
q = np.loadtxt('q_D_LABEL.dat')
h, J = get_hJ(q,D)
```
-  `evolutions_D_LABEL.pdf` with some plots of the paramaters evolution



##  FILES
- `tutorial.ipynb`: Example of how to go from experimental averages to Ising model parameters using the `find_ising.py` file 
- `find_ising.py`: given a file containing f_data (in the format `f_data_{d}_{label}.dat`) returns the parameters of the ising model in the file `q_{d}_{label}.dat`
- `function.py`
- `A_generate_data.ipynb`: preprocess experimental data for part C
- `B_find_ising.ipynb`: get the model, explore and check
- `C_Figures.ipynb`: generate the plots presented in the paper



## Experimental Data
Dataset used in the paper: https://alleninstitute.github.io/abc_atlas_access/descriptions/MERFISH-C57BL6J-638850.html 

Experimental data paper: https://www.biorxiv.org/content/10.1101/2023.03.06.531121v1

Please specify the correct data path in the `DATA_PATH` variable in the `functions.py` file.

