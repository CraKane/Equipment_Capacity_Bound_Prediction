import datetime as dt
import pandas as pd

data = pd.Series([1, 2, 3])
returns = 100 * data.pct_change().dropna()
print(returns)

from arch import arch_model
am = arch_model(returns, p=4)
res = am.fit().params
sum_ = sum(res) - res['mu'] - res['omega'] - res['beta[1]']
print(res)
print(sum_)