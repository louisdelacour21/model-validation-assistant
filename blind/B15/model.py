import numpy as np


class Model:
    name = "blind-B15"
    conf = 0.99
    window = 500

    def forecast(self, history):
        r = np.asarray(history, dtype=float)[-self.window:]
        losses = np.sort(-r)[::-1]
        n = len(losses)
        k = int(np.ceil(n * (1.0 - self.conf)))
        var = float(np.quantile(losses, self.conf))
        es = float(np.mean(losses[1:k + 1]))
        return max(var, 0.0), max(es, var, 0.0)
