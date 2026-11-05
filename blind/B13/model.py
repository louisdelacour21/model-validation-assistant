import numpy as np


class Model:
    name = "blind-B13"
    conf = 0.99
    window = 500

    def forecast(self, history):
        r = np.asarray(history, dtype=float)[-self.window:]
        losses = -r
        q = 0.95
        var = float(np.quantile(losses, q))
        tail = losses[losses >= var]
        es = float(tail.mean()) if tail.size else var
        return max(var, 0.0), max(es, var, 0.0)
