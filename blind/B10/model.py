import numpy as np


class Model:
    name = "blind-B10"
    conf = 0.99
    window = 500

    def forecast(self, history):
        x = np.asarray(history, dtype=float)[-self.window:]
        losses = -x
        q = 0.95
        var = float(np.quantile(losses, q))
        tail = losses[losses >= var]
        es = float(tail.mean()) if tail.size else var
        return max(var, 0.0), max(es, var, 0.0)
