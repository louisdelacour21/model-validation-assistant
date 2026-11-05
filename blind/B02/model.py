import numpy as np


class Model:
    name = "blind-B02"
    conf = 0.99
    window = 500

    def forecast(self, history):
        r = np.asarray(history, dtype=float)[-self.window:]
        losses = -r
        var = float(np.quantile(losses, self.conf))
        recent = losses[-250:]
        tail = recent[recent >= np.quantile(recent, self.conf)]
        es = float(max(tail.mean(), var))
        return var, es
