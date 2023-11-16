# CR-FM-NES [[slide]](slide_cec2022.pdf)

[CR-FM-NES](https://arxiv.org/abs/2201.11422) [1] is an efficient black-box optimization method for high-dimensional problems.
In CR-FM-NES, both time and space complexity are linear, with partially considering variable dependencies.
Therefore, it is especially suitable for high-dimensional problems (about hundreds to thousands of dimensions).
On the other hand, it often achieves high performance even on low-dimensional problems.
This is an extension of [FM-NES (Fast Moving Natural Evolution Strategy)](https://arxiv.org/abs/2108.09455) [2] to be applicable in high-dimensional problems.
Please e-mail at masahironomura5325@gmail.com if you have any issue.

<img width="1215" alt="188303830-aa7b11d0-c6ff-4d1a-9bd8-2ccbf4d7e2dd" src="https://user-images.githubusercontent.com/10880858/211967554-65d632bd-3e77-4725-998c-20f69bb8f5ce.png">

If you find this code useful in your research then please cite:
```bibtex
@INPROCEEDINGS{nomura2022fast,
  title={Fast Moving Natural Evolution Strategy for High-Dimensional Problems},
  author={Nomura, Masahiro and Ono, Isao},
  booktitle={2022 IEEE Congress on Evolutionary Computation (CEC)}, 
  pages={1-8},
  year={2022},
}
```

## News
* **(2023/09)** CR-FM-NES has been integrated into [EvoJax](https://github.com/google/evojax/tree/main/evojax/algo) with the [promising performance](https://github.com/nomuramasahir0/crfmnes/issues/13) on EvoJax Benchmarks. Additionally there are quite [interesting results when using it as part of an QD-algorithm](https://github.com/google/evojax/pull/52). Thanks [@dietmarwo](https://github.com/dietmarwo)!
* **(2023/01)** We released the first major version of `crfmnes` in January 2023. You can install it via `pip install crfmnes`.
* **(2022/12)** CR-FM-NES has been integrated into [evosax](https://github.com/RobertTLange/evosax), which provides JAX-based evolution strategies implementation. Thanks [@RobertTLange](https://github.com/RobertTLange) and [@Obliman](https://github.com/Obliman)!
* **(2022/07)** The paper [Fast Moving Natural Evolution Strategy for High-Dimensional Problems](https://arxiv.org/abs/2201.11422) has been accepted at IEEE CEC'22.
* **(2022/05)** CR-FM-NES has been integrated into [fast-cma-es](https://github.com/dietmarwo/fast-cma-es), which faster C++/Eigen based implementations and a coordinated parallel retry mechanism. Thanks [@dietmarwo](https://github.com/dietmarwo)!
* **(2018/10)** CR-FM-NES has been integrated into [CRFMNES_CS](https://github.com/bakanaouji/CRFMNES_CS), which is CR-FM-NES C# implementation. Thanks [@bakanaouji](https://github.com/bakanaouji)!


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

## For Constrained Problems

CR-FM-NES can be applied to (implicitly) constrained black-box optimization problems.
Please set the objective function value of the infeasible solution to `np.inf`.
CR-FM-NES reflects the information and performs an efficient search. 
Please refer to [3] for the details of the constraint handling methods implemented in this repository.

## Other Versions of CR-FM-NES

I really appreciate that CR-FM-NES is implemented in other settings.

* C# Implementation: [bakanaouji/CRFMNES_CS](https://github.com/bakanaouji/CRFMNES_CS)
* C++ Implementation: [dietmarwo/fast-cma-es](https://github.com/dietmarwo/fast-cma-es/blob/master/_fcmaescpp/crfmnes.cpp)
* Jax(Python) Implementation: [RobertTLange/evosax](https://github.com/RobertTLange/evosax/blob/main/evosax/strategies/cr_fm_nes.py)
* Jax(Python) Implementation: [google/evojax](https://github.com/google/evojax/tree/main/evojax/algo)


## References
* [1] [M. Nomura, I. Ono, Fast Moving Natural Evolution Strategy for High-Dimensional Problems, IEEE CEC, 2022.](https://arxiv.org/abs/2201.11422)
* [2] [M. Nomura, I. Ono, Natural Evolution Strategy for Unconstrained and Implicitly Constrained Problems with Ridge Structure, IEEE SSCI, 2021.](https://arxiv.org/abs/2108.09455)
* [3] [M. Nomura, N. Sakai, N. Fukushima, and I. Ono, Distance-weighted Exponential Natural Evolution Strategy for Implicitly Constrained Black-Box Function Optimization, IEEE CEC, 2021.](https://www.researchgate.net/publication/353787484_Distance-weighted_Exponential_Natural_Evolution_Strategy_for_Implicitly_Constrained_Black-Box_Function_Optimization)
