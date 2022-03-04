import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def y_fmt(y, pos):
    decades = [1e9, 1e6, 1e3, 1e0, 1e-3, 1e-6, 1e-9]
    suffix = ["G", "M", "K", "", "m", "u", "n"]
    if y == 0:
        return str(0)
    for i, d in enumerate(decades):
        if np.abs(y) >= d:
            val = y / float(d)
            val = np.round(val, 6)
            signf = len(str(val).split(".")[1])
            if signf == 0:
                return f'{int(val):d}{suffix[i]}'
            else:
                if signf == 1:
                    if str(val).split(".")[1] == "0":
                        return f'{int(round(val)):d}{suffix[i]}'
                tx = f"{{val:.{signf}f}}{{suffix}}"
                return tx.format(val=np.round(val, 3), suffix=suffix[i])
    return str(y)


plt.rcParams.update({'axes.formatter.limits': (-4, 4)})
formatter = plt.FuncFormatter(y_fmt)
unique_legends = []


def legend_without_duplicate_labels(ax, loc):
    global unique_legends
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    unique_legends = unique
    if len(unique):
        ax.legend(*zip(*unique), loc=loc)
    return unique


def setup(ax=None, *, loc=None, title=None, formatter=formatter):
    if ax is None:
        ax = plt.gca()
    if loc != 'no':
        legend_without_duplicate_labels(ax, loc)
    if formatter is not None:
        ax.xaxis.set_major_formatter(formatter)
    ax.grid()
    if title is not None:
        ax.set_title(title)


def export_legends(ax, handles_and_labels, filename=None):
    legend = ax.figlegend(*zip(*handles_and_labels), loc=3, framealpha=1, frameon=False, ncol=len(handles_and_labels))
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    bbox.y1 += 0.1
    print(bbox)
    if filename is not None:
        fig.savefig(filename, dpi="figure", bbox_inches=bbox)
    fig.show()


def merge_curves(runs, *, key):
    from .database import resolve_runs

    runs = resolve_runs(runs)

    lengths = []
    curves = []
    for run in runs:
        df = run.read_value(key)
        lengths.append(len(df))
        curves.append(df['mean'])

    lengths = sorted(set([0] + lengths))

    dfs = []
    for start, end in zip(lengths, lengths[1:]):
        cur_ys = [ys[start:end] for ys in curves if len(ys) >= end]
        step = cur_ys[0].index

        cur_ys = [ys.to_numpy() for ys in cur_ys]
        mean = np.mean(cur_ys, axis=0)
        std = np.std(cur_ys, axis=0)
        n = np.full(end - start, len(cur_ys))
        df = pd.DataFrame({'step': step, 'mean': mean, 'std': std, 'n': n})
        dfs.append(df)

    ret = pd.concat(dfs)
    ret.set_index('step', inplace=True)
    return ret


def color_gen():
    while True:
        for i in range(10):
            yield f'C{i}'


__all__ = ['legend_without_duplicate_labels', 'setup', 'unique_legends', 'merge_curves', 'color_gen']
