import numpy as np


class Model:
    name = "blind-B05"
    conf = 0.99
    window = 500

    def forecast(self, history):
        r = np.asarray(history, dtype=float)[-self.window:]
        loss = np.sort(-r)[::-1]
        var = float(np.quantile(loss, self.conf))
        k = max(int(round(len(loss) * (1 - self.conf))), 2)
        es = float(np.mean(loss[1:k]))
        return var, max(es, var)
