# crfmnes

CR-FM-NES implementation

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


## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/nmasahiro/crfmnes/tags). 


## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/nmasahiro/crfmnes/blob/master/LISENCE) file for details
