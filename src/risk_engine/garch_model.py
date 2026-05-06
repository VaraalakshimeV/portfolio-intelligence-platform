"""
GARCH(1,1) Volatility Forecaster
Industry-standard model for financial volatility — captures volatility
clustering (calm periods followed by volatile periods) that a Random Forest
structurally cannot model because it treats each observation independently.

GARCH(1,1) equation:
  sigma²_t = omega + alpha * epsilon²_(t-1) + beta * sigma²_(t-1)
  persistence = alpha + beta  (close to 1 = long volatility memory)
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path

class GARCHVolatilityModel:

    def __init__(self, p: int = 1, q: int = 1):
        self.p = p
        self.q = q
        self.fitted_models  = {}
        self.forecast_cache = {}

    def fit(self, ticker: str, returns: pd.Series) -> dict:
        from arch import arch_model
        # Scale returns to % — arch library works better on that scale
        model  = arch_model(returns * 100, vol='Garch', p=self.p, q=self.q,
                            dist='t')          # Student-t captures fat tails
        result = model.fit(disp='off', show_warning=False)
        self.fitted_models[ticker] = result

        metrics = {
            'aic':         round(result.aic, 2),
            'bic':         round(result.bic, 2),
            'omega':       round(float(result.params['omega']),    6),
            'alpha':       round(float(result.params['alpha[1]']), 4),
            'beta':        round(float(result.params['beta[1]']),  4),
            'persistence': round(float(result.params['alpha[1]'] +
                                       result.params['beta[1]']),  4),
            'nu':          round(float(result.params.get('nu', 0)), 2),
        }
        return metrics

    def forecast_volatility(self, ticker: str, horizon: int = 1) -> float:
        """Return annualized volatility forecast for next `horizon` days."""
        result  = self.fitted_models[ticker]
        fc      = result.forecast(horizon=horizon, reindex=False)
        var_1d  = fc.variance.iloc[-1, 0] / 10000   # undo *100 scaling
        return float(np.sqrt(var_1d * 252))           # annualize

    def fit_and_forecast(self, ticker: str, returns: pd.Series) -> dict:
        metrics  = self.fit(ticker, returns)
        ann_vol  = self.forecast_volatility(ticker)
        return {**metrics, 'forecast_ann_vol': round(ann_vol, 4)}

    def save(self, path: str = 'models/garch_model.pkl'):
        Path(path).parent.mkdir(exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str = 'models/garch_model.pkl'):
        with open(path, 'rb') as f:
            return pickle.load(f)
