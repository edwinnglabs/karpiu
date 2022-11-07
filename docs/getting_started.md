# Getting Started

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
