import visdom
import numpy as np

class LinePlotter(object):
    def __init__(self, env_name="office"):
        self.vis = visdom.Visdom()
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.vis.line(
                X = np.array([x, x]),
                Y = np.array([y, y]),
                env = self.env,
                opts = dict(
                    legend=[split_name],
                    title=var_name,
                    xlabel="Iters",
                    ylabel=var_name
                    )
                )
        else:
            self.vis.updateTrace(
                X = np.array([x, x]),
                Y = np.array([y, y]),
                env = self.env,
                win = self.plots[var_name],
                name = split_name
                )

    def save(self):
        self.vis.save([self.env])
