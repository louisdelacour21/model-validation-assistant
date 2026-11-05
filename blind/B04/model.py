import numpy as np


class Model:
    name = "blind-B04"
    conf = 0.99
    window = 500

    def forecast(self, history):
        losses = -np.asarray(history, dtype=float)[-self.window:]
        var = float(np.quantile(losses, self.conf))
        tail = losses[losses >= var]
        es = float(tail.mean()) if tail.size else var
        return var, max(es, var)
