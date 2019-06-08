# Single Shell Free Water Elimination Diffusion Tensor Model 

This is a Python implementation of a Free Water Elimination Model for the
analysis of Single Shell Diffusion Tensor MRI Images. We would like to
[separate out the signal decay of extracellular free water from  water that is
in the vicinity of cellular
tissue](http://pnl.bwh.harvard.edu/portfolio-item/free-water/). This is an
ill-posed problem because there are infinitely many solutions. So constraints
are imposed via the time evolution of a gradient flow on a Riemannian manifold
to get a unique solution. The resulting optimization problem is solved via
gradient descent.  **This software is written for educational purposes only**.


| ![Free water corrected FA](./fw_fa.png) |
|:---:| 
| *Fig 1. Free Water corrected Fractional Anisotropy* |

## Software requirements

This code was developed with the following packages.

* Python 3.6.3
* [Dipy 0.14.0](http://nipy.org/dipy/index.html)

## Getting Started

* Download with git. `git checkout https://github.com/sameerd/DiffusionTensorImaging.git`
* Follow one of the two examples
  * [example.py](example.py) : python script that can be used in a pipeline with local data.
  * [Example.ipynb](notebooks/Example.ipynb) : Jupyter Notebook with an interactive overview of how to use this repository with publicly available data
* Check that the first panel (marked `loss` below) converges

| ![Loss function](./loss_function.png) |
|:---:| 
| *Fig 2. Loss function panels* |


If the model isn't converging, you can try reducing the time step `dt` or
increase the number of iterations. Once you are sure about convergence, you can
use the code's convenience functions to return the `free water map`, the free
water corrected `mean diffusivity`, and the free water corrected `Fractional
Anisotropy` (see Fig 1. above).

## Documentation

* This [README](README.md) file has an overview of how to use this repository. 
* The [SingleShellFreewater.pdf](./doc/SingleShellFreeWater.pdf) file helps describe what the code is doing in the language of mathematics and ties it to the reference papers.
* The core of the implementation is in the [freewater.py](./pymods/freewater.py) file and the interface class in [freewater\_runner.py](./pymods/freewater_runner.py).

## Caveats
* **This software is untested. Please do your own testing before using it.**

* There is a separate effort by @mvgolub and @RafaelNH to put in an expanded
single shell model into the [Dipy](http://nipy.org/dipy/index.html)
repository. Please follow along at the [Dipy Issue Tracker
#827](https://github.com/nipy/dipy/issues/827). When the Dipy version is ready,
it is likely to be better tested, more user friendly, efficient and have more
features than the version in this repository.

## Memory Usage

On a recent test, a volume of size 100x100x10 with approximately 200 gradient
directions consumed around 3GB of memory and took approximately 30 minutes to
run using a 2GHz AMD opteron 6100 Series processor. A more realistically sized
larger volume will take longer to run and consume more memory. Try to make sure
that there is enough memory to prevent swapping to disk.


## References

* Pasternak, O. , Sochen, N. , Gur, Y. , Intrator, N. and Assaf, Y. (2009), Free water elimination and mapping from diffusion MRI. Magn. Reson. Med., 62: 717-730. doi:[10.1002/mrm.22055](https://doi.org/10.1002/mrm.22055)

* Pasternak O., Maier-Hein K., Baumgartner C., Shenton M.E., Rathi Y., Westin CF. (2014) The Estimation of Free-Water Corrected Diffusion Tensors. In: Westin CF., Vilanova A., Burgeth B. (eds) Visualization and Processing of Tensors and Higher Order Descriptors for Multi-Valued Data. Mathematics and Visualization. Springer, Berlin, Heidelberg doi:[10.1007/978-3-642-54301-2\_11](https://doi.org/10.1007/978-3-642-54301-2_11)


