[![build](https://github.com/edwinnglabs/karpiu/actions/workflows/build.yaml/badge.svg?event=push)](https://github.com/edwinnglabs/karpiu/actions/workflows/build.yaml)
[![format](https://github.com/edwinnglabs/karpiu/actions/workflows/format.yaml/badge.svg?event=push)](https://github.com/edwinnglabs/karpiu/actions/workflows/format.yaml)
[![pages](https://github.com/edwinnglabs/karpiu/actions/workflows/pages.yaml/badge.svg?branch=main)](https://github.com/edwinnglabs/karpiu/actions/workflows/pages.yaml)
# Karpiu

**Karpiu** is a package designed for Marketing Mix Modeling (MMM) by calling [Orbit](https://github.com/uber/orbit) from the backend. Karpiu is still in its beta version.  Please use it at your own risk.

# Tutorials & Documentation

For usage, users can refer to this [documentation](https://edwinng.com/karpiu/).

# Installation

To access the development version, please follow the below instructions or simply use `make install-dev` after cloning the repository.  Due to the dependencies of `pystan==2.19.1.1`, it is recommended to fulfill condition `python<=3.8` to have the best experience.

```bash
$ git clone https://github.com/edwinnglabs/karpiu.git
$ cd karpiu
$ pip install pystan==2.19.1.1
$ pip install -r requirements.txt
$ pip install -r requirements-test.txt
$ pip install -r requirements-docs.txt
$ pip install -e .
```

# Related Work

## Codebase

1. [Robyn](https://github.com/facebookexperimental/Robyn) - Robyn is an automated Marketing Mix Modeling (MMM) code. It aims to reduce human bias by means of ridge regression and evolutionary algorithms, enables actionable decision making providing a budget allocator and diminishing returns curves and allows ground-truth calibration to account for causation.

2. [Lightweight (Bayesian) Marketing Mix Modeling](https://github.com/google/lightweight_mmm) - LMMM is a python library that helps organisations understand and optimise marketing spend across media channels.

3. [Orbit](https://github.com/uber/orbit) - A Python package for Bayesian forecasting with object-oriented design and probabilistic models under the hood.

## Research
- [Ng, E., Wang, Z. & Dai, A. (2021) Bayesian Time Varying Coefficient Model with Applications to
Marketing Mix Modeling](http://papers.adkdd.org/2021/papers/adkdd21-ng-bayesian.pdf)
- [Jin, Y., Wang, Y., Sun, Y., Chan, D., & Koehler, J. (2017). Bayesian Methods for Media Mix Modeling with Carryover and Shape Effects. Google Inc.](https://research.google/pubs/pub46001/)
- [Chan, D., & Perry, M. (2017). Challenges and Opportunities in Media Mix Modeling.](https://research.google/pubs/pub45998/)
- [Sun, Y., Wang, Y., Jin, Y., Chan, D., & Koehler, J. (2017). Geo-level Bayesian Hierarchical Media Mix Modeling.](https://research.google/pubs/pub46000/)

## Community Articles

- [Are Marketing Mix Models Useful? I Spent My Own Money To Find Out](https://forecastegy.com/posts/marketing-mix-models/) by Mario Filho.
- [How Google LightweightMMM Works](https://getrecast.com/google-lightweightmmm/) by Mike Taylor.
- [
Bayesian Media Mix Modeling for Marketing Optimization](https://www.pymc-labs.io/blog-posts/bayesian-media-mix-modeling-for-marketing-optimization/)
by Benjamin Vincent.

# Fun Fact

**Karpiu** is a term invented the author's daugther while she just reached her terrible two. In the early development of this project, "Karpiu" became the dominating term in the author's home office. The author later dedicated this term to the package.
