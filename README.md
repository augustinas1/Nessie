<img align="right" src="https://user-images.githubusercontent.com/26488673/175932056-2fdfd47a-7b61-42f7-a7c1-c51a40f873f2.svg" alt="logo" width="200"/>

# Neural Estimation of Stochastic Simulations for Inference and Exploration

This repository contains the code for the paper

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;A. Sukys, K. Öcal and R. Grima, "Approximating Solutions of the Chemical Master Equation using Neural Networks"

Running the MAPK Pathway inference example requires data from [[1]](#1) which can be obtained from the authors.

If you have any questions or comments, please feel free to contact the authors or open a pull request on GitHub.

### Workflow:
* Define a reaction network (easiest using [Catalyst.jl](https://github.com/SciML/Catalyst.jl))
* Create training, validation and (ideally) testing data using `generate_data.jl`
* Create a `MNBModel` to house your neural network in `nnet.jl`
* Train the neural network using `train_NN.jl`
* Explore and enjoy!

Examples that were used in the paper can be found below. The above steps are distributed between two files in each example: `gen.jl` and `train.jl`.

### Examples:
- Autoregulatory Feedback Loop (folder `afl`)
- Genetic Toggle Switch (folder `ts`)
- MAPK Pathway (folder `mapk`)
- Model of mRNA Turnover (folder `mrnad`)

## References:

<a id="1">[1]</a> C. Zechner, J. Ruess et al., "Moment-based inference predicts bimodality in transient gene expression", PNAS 109(21), 2012.
