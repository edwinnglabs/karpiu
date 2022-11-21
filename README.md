[![build](https://github.com/edwinnglabs/karpiu/actions/workflows/build.yaml/badge.svg?event=push)](https://github.com/edwinnglabs/karpiu/actions/workflows/build.yaml)

[![format](https://github.com/edwinnglabs/karpiu/actions/workflows/format.yaml/badge.svg)](https://github.com/edwinnglabs/karpiu/actions/workflows/format.yaml)

# Karpiu

**Karpiu** is a package designed for Marketing Mix Modeling (MMM) by calling [Orbit](https://github.com/uber/orbit) from the backend. Karpiu is still in its beta version.  Please use it at your own risk.

# Documentation

For usage in details, users can refer to this [documentation](https://edwinng.com/karpiu/).

# Installation

You need both `karpiu` and `orbit-ml` installed from dev branch

```bash
$ pip install --upgrade git+https://github.com/edwinnglabs/karpiu.git
$ pip install --upgrade git+https://github.com/uber/orbit.git
```

# Quick Start

Load data
```python
import import pandas as pd

RAW_DATA_FILE = 'data.csv'
df = pd.read_csv(RAW_DATA_FILE, parse_dates=['date'])

adstock_df = pd.read_csv('./adstock.csv')
adstock_df = adstock_df.sort_values(by=['regressor'])
adstock_df = adstock_df.set_index('regressor')
paid_channels = ['tv', 'radio', 'social', 'promo', 'search']
```

Build a basic MMM
```python
from karpiu.models import MMM
mmm = MMM(
    kpi_col='sales',
    date_col='date', 
    spend_cols=paid_channels,
    event_cols=[],
    seed=2022,
    adstock_df=adstock_df,
)
mmm.optim_hyper_params(df)
mmm.fit(df)
```

Extract attribution from model
```python
from karpiu.explainer import Attributor
ATTR_START = '2019-03-01'
ATTR_END = '2019-03-31'
attributor = Attributor(model=mmm, start=ATTR_START, end=ATTR_END)
activities_attr_df, spend_attr_df, spend_df, cost_df = attributor.make_attribution()
```

A visualization of attribution
```python
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1, figsize=(16, 8))
ax.stackplot(
    activities_attr_df['date'].values, 
    activities_attr_df[['organic'] + paid_channels].values.transpose(), 
    labels=['organic'] + paid_channels
)
ax.set_title("Attribution by Activities Date", fontdict={'fontsize': 24})
ax.set_xlabel("date", fontdict={'fontsize': 18})
ax.set_ylabel("sales", fontdict={'fontsize': 18})
fig.legend()
fig.tight_layout();
```

# Related Work

## Codebase

1. [Robyn](https://github.com/facebookexperimental/Robyn) - Robyn is an automated Marketing Mix Modeling (MMM) code. It aims to reduce human bias by means of ridge regression and evolutionary algorithms, enables actionable decision making providing a budget allocator and diminishing returns curves and allows ground-truth calibration to account for causation.

2. [Lightweight (Bayesian) Marketing Mix Modeling](https://github.com/google/lightweight_mmm) - LMMM is a python library that helps organisations understand and optimise marketing spend across media channels.

## Research

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
