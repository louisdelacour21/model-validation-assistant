import numpy as np


class Model:
    name = "blind-B12"
    conf = 0.99
    window = 500

    def forecast(self, history):
        r = np.asarray(history, dtype=float)[-self.window:]
        losses = -r
        var = float(np.quantile(losses, self.conf))
        cut = float(np.quantile(losses, 1 - (1 - self.conf) * 2.5))
        tail = losses[losses >= cut]
        es = float(tail.mean())
        return var, es
