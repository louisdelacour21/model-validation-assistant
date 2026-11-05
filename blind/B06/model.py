import numpy as np


class Model:
    name = "blind-B06"
    conf = 0.99
    window = 500
    lookback = 250

    def forecast(self, history):
        x = np.asarray(history, dtype=float)[-self.lookback:]
        losses = -x
        var = float(np.quantile(losses, self.conf))
        tail = losses[losses >= var]
        es = float(tail.mean())
        return var, max(es, var)
