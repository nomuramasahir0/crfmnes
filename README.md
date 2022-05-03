# CR-FM-NES

[CR-FM-NES](https://arxiv.org/abs/2201.11422) [1] implementation.
This is an extension of [FM-NES (Fast Moving Natural Evolution Strategy)](https://arxiv.org/abs/2108.09455) [2] to be
applicable in high-dimensional problems. Please e-mail at masahironomura5325@gmail.com if you have any issue.

If you find this code useful in your research then please cite:
```bibtex
@article{nomura2022fast,
  title={Fast Moving Natural Evolution Strategy for High-Dimensional Problems},
  author={Nomura, Masahiro and Ono, Isao},
  journal={arXiv preprint arXiv:2201.11422},
  year={2022}
}
```


## Getting Started


### Prerequisites

You need only [NumPy](http://www.numpy.org/) that is the package for scientific computing.

### Installing

Please run the following command.

```bash
$ pip install crfmnes
```

## Example

This is a simple example that objective function is sphere function.
Note that the optimization problem is formulated as **minimization** problem.

```python
import numpy as np
from crfmnes import CRFMNES

dim = 3
f = lambda x: np.sum(x**2)
mean = np.ones([dim, 1]) * 0.5
sigma = 0.2
lamb = 6
crfmnes = CRFMNES(dim, f, mean, sigma, lamb)

x_best, f_best = crfmnes.optimize(100)
print("x_best:{}, f_best:{}".format(x_best, f_best))
# x_best:[1.64023896e-05 2.41682149e-05 3.40657594e-05], f_best:2.0136169613476005e-09
```

## For Implicitly Constrained Problems

CR-FM-NES can be applied to (implicitly) constrained black-box optimization problems.
Please set the objective function value of the infeasible solution to `np.inf`.
CR-FM-NES reflects the information and performs an efficient search. 
Please refer to [3] for the details of the constraint handling methods implemented in this repository.

## Other Versions of CR-FM-NES

I really appreciate that CR-FM-NES is implemented in other settings.

* C# Implementation: https://github.com/bakanaouji/CRFMNES_CS


## References
* [1] [M. Nomura, I. Ono, Fast Moving Natural Evolution Strategy for High-Dimensional Problems, IEEE CEC, 2022.](https://arxiv.org/abs/2201.11422)
* [2] [M. Nomura, I. Ono, Natural Evolution Strategy for Unconstrained and Implicitly Constrained Problems with Ridge Structure, IEEE SSCI, 2021.](https://arxiv.org/abs/2108.09455)
* [3] [M. Nomura, N. Sakai, N. Fukushima, and I. Ono, Distance-weighted Exponential Natural Evolution Strategy for Implicitly Constrained Black-Box Function Optimization, IEEE CEC, 2021.](https://ieeexplore.ieee.org/document/9504865)
