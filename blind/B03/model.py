import numpy as np


class Model:
    name = "blind-B03"
    conf = 0.99
    window = 500

    def __init__(self):
        self.alpha = 0.05

    def forecast(self, history):
        r = np.asarray(history, dtype=float)[-self.window:]
        losses = -r
        var = float(np.quantile(losses, 1.0 - self.alpha))
        tail = losses[losses >= var]
        es = float(tail.mean())
        return var, es
