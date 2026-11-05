import numpy as np


class Model:
    name = "blind-B14"
    conf = 0.99
    window = 500

    def forecast(self, history):
        r = np.asarray(history, dtype=float)[-self.window:]
        losses = -r
        var = float(np.quantile(losses, 0.95))
        tail = losses[losses >= var]
        es = float(tail.mean())
        return var, es
