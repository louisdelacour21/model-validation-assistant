import numpy as np


class Model:
    name = "blind-B16"
    conf = 0.99
    window = 500

    def __init__(self, tail_conf=0.975):
        self.tail_conf = tail_conf

    def forecast(self, history):
        x = np.asarray(history, dtype=float)[-self.window:]
        losses = -x
        var = float(np.quantile(losses, self.conf))
        cut = np.quantile(losses, self.tail_conf)
        tail = losses[losses >= cut]
        es = float(tail.mean())
        return max(var, 0.0), max(es, var, 0.0)
