import argparse
import subprocess
import shlex
import os
import glob
import json
import time
from matplotlib import pyplot as plt
import numpy as np


LOGDIR = os.path.expanduser('~/bigtrans_logs')
GRAPHDIR = os.path.expanduser('~/bigtrans_graphs')
BUCKET = f'gs://nmt-models/{os.environ["OPENAI_USER"]}/'


os.makedirs(LOGDIR, exist_ok=True)
os.makedirs(GRAPHDIR, exist_ok=True)


EPOCH_VARS = frozenset(['epoch', 'n_epochs'])
x_var_mapping = {
    'epoch': 'n_epochs',
    'step': 'n_updates'
}
y_var_mapping = {
    'eval_loss': {
        'loss': 'valid_gen_loss',
        'loss_clf': 'valid_clf_loss',
        'acc_clf': 'valid_acc',
    },
    'train_loss': {
        'loss_avg': 'train_gen_loss',
        'loss_clf_avg': 'train_clf_loss',
    }
}


class Series(object):
    def __init__(self, logpath, model_name, series_id, x_var, y_var, average, base=None, convert_to_epochs=False, legend=None):
        self.name = model_name
        if legend:
            self.name += ":" + legend
        with open(logpath, 'r') as f:
            lines = f.readlines()
        identifier = json.loads(lines[0])
        img_gen_repr_learn = False
        if 'code' in identifier:
            img_gen_repr_learn = True

        if img_gen_repr_learn:
            x_var = x_var_mapping[x_var]
            y_var = y_var_mapping[series_id][y_var]

        data = []
        epoch_length = None
        for l in lines[1:]:
            try:
                parse = json.loads(l)
                if epoch_length is None and 'n_updates_per_epoch' in parse:
                    epoch_length = float(parse['n_updates_per_epoch'])
                if img_gen_repr_learn:
                    data.append(parse)
                elif 'series' in parse and parse['series'] == series_id:
                    data.append(parse)
            except json.JSONDecodeError:
                pass

        data = [d for d in data if x_var in d and y_var in d]
        self.x = np.array([l[x_var] for l in data]).astype(np.float64)
        self.y = np.array([l[y_var] for l in data]).astype(np.float64)
        if convert_to_epochs and x_var not in EPOCH_VARS:
            self.x /= epoch_length
        if base is not None:
            self.y /= np.log(base)
        if average:
            out_y = []
            for j in range(1, len(self.y) + 1):
                mini = max(0, j - args.average)
                out_y.append(self.y[mini:j].mean())
            self.y = np.array(out_y)

        if len(self.x) > 0 and len(self.y) > 0:
            max_idx = np.argmax(self.y)
            min_idx = np.argmin(self.y)
            self.xmax = self.x[max_idx]
            self.ymax = self.y[max_idx]
            self.xmin = self.x[min_idx]
            self.ymin = self.y[min_idx]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # comma-separated model name substrings
    parser.add_argument('--model', type=str)
    parser.add_argument('--title', type=str, default=None)
    parser.add_argument('--skip_cp', action="store_true")
    parser.add_argument('--ylim', type=str, default="")
    parser.add_argument('--xlim', type=str, default="")
    parser.add_argument('--series', type=str, default="eval_loss:epoch:loss")
    parser.add_argument('--average', type=int, default=None)

    parser.add_argument('--train', action="store_true")
    parser.add_argument('--valid', action="store_true")
    parser.add_argument('--acc', action="store_true")
    parser.add_argument('--clf_loss', action="store_true")
    parser.add_argument('--train_valid', action="store_true")
    parser.add_argument('--max', action="store_true")

    parser.add_argument('--base', type=float)

    parser.add_argument('--logy', action="store_true")
    parser.add_argument('--logx', action="store_true")

    parser.add_argument('--show', action='store_true')

    args = parser.parse_args()
    if not args.title:
        args.title = args.model

    # Basic sanity-checks
    if args.acc and args.base is not None:
        raise ValueError("Converting to other units is supported only for generative losses")

    legends = [None]
    if args.train:
        args.series = 'train_loss:step:loss_avg'
    if args.valid:
        args.series = 'eval_loss:epoch:loss'
    if args.acc:
        args.series = 'eval_loss:epoch:acc_clf'
        args.max = True
    if args.clf_loss:
        args.series = 'eval_loss:epoch:loss_clf'
    if args.train_valid:
        legends = ["valid", "train"]
        if args.acc:
            args.series = 'eval_loss:epoch:acc_clf,train_loss:step:loss_acc'
            args.max = True
        elif args.clf_loss:
            args.series = 'eval_loss:epoch:loss_clf,train_loss:step:loss_clf_avg'
            args.max = False
        else:
            args.series = 'eval_loss:epoch:loss,train_loss:step:loss_avg'

    os.makedirs(LOGDIR, exist_ok=True)

    strs = args.model.split(',')
    print('Plotting models with names', strs)

    prefix = BUCKET
    suffix = '/log.jsonl'

    names = []
    sps = []
    if not args.skip_cp:
        files = []
        for s in strs:
            modelstr = f'{prefix}{s}{suffix}'
            cmd = f'gsutil ls {modelstr}'
            try:
                o = subprocess.check_output(shlex.split(cmd))
                files += [a.decode('utf-8') for a in o.split()]
            except subprocess.CalledProcessError:
                print(f'ls failed for {modelstr}')
        for f in files:
            name = f[len(prefix):-len(suffix)]
            p = os.path.join(LOGDIR, name, 'log.jsonl')
            cmd = f'gsutil cp {f} {p}'
            sps.append(subprocess.Popen(shlex.split(cmd)))
        while sps:
            for proc in sps:
                retcode = proc.poll()
                if retcode is not None:
                    sps.remove(proc)
                else:
                    time.sleep(0.1)
    localpaths = []
    for s in strs:
        prefix = f'{LOGDIR}/'
        suffix = 'log.jsonl'
        for fp in glob.glob(os.path.join(prefix, s, suffix)):
            localpaths.append((fp, fp[len(prefix):-len(suffix) - 1]))

    # Series types define what to show as the train and validation curves.
    series_types = args.series.split(',')
    assert len(series_types) > 0
    series = [[] for _ in series_types]
    print('series to print:', series_types)
    convert_to_epochs = set(srs.split(':')[1] in EPOCH_VARS for srs in series_types) == {True, False}
    for logpath, model_name in localpaths:
        for idx, (series_str, legend) in enumerate(zip(series_types, legends)):
            series_id, x_var, y_var = series_str.split(':')
            s = Series(logpath, model_name, series_id, x_var, y_var, args.average, base=args.base, convert_to_epochs=convert_to_epochs, legend=legend)
            if len(s.x) > 0 and len(s.y) > 0:
                series[idx].append(s)

    assert len(series) > 0 and len(series[0]) > 0
    cm = plt.cm.gist_rainbow
    colors = cm(np.linspace(0, 1, len(series[0])))
    if args.show:
        plt.figure(figsize=(5, 5))
    else:
        plt.figure(figsize=(20, 20))

    # # sort to keep colors consistent across plottings
    for idx in range(len(series_types)):
        series[idx].sort(key=lambda x: x.name)

    ymin_data = []
    ymax_data = []
    # For --train_valid, validation curve will be shown in solid line by
    # default.
    linestyles = ["-", "--"]
    for srs_list, style in zip(reversed(series), reversed(linestyles[:len(series)])):
        for idx, srs in enumerate(srs_list):
            alpha = 0.7 if style == '--' else 1.0
            plt.plot(srs.x, srs.y, linestyle=style, color=colors[idx], label=srs.name, alpha=alpha)
            ymax_data.append(srs.y.max())
            ymin_data.append(srs.y.min())

    plt.grid(linestyle="--")
    if args.logy:
        plt.yscale('log')
    if args.logx:
        plt.xscale('log')
    if args.ylim:
        ymin, ymax = [float(x) for x in args.ylim.split(',')]
        plt.ylim(ymin, ymax)
        plt.yticks(np.arange(ymin, ymax, (ymax - ymin) / 50))
    else:
        ymin, ymax = min(ymin_data), max(ymax_data)
        plt.yticks(np.arange(ymin, ymax, (ymax - ymin) / 50))

    if args.xlim:
        xmin, xmax = [float(x) for x in args.xlim.split(',')]
        plt.xlim(xmin, xmax)
    os.makedirs(GRAPHDIR, exist_ok=True)
    fname = args.title + args.series.replace(":", "-").replace(",", "-")
    outpath = os.path.join(GRAPHDIR, fname[:100] + '.png')

    plt.title(f"{args.series} for {args.model}")

    plt.legend()
    plt.savefig(outpath)
    if args.max:
        for idx in range(len(series)):
            series[idx].sort(key=lambda x: x.ymax)
            for s in series[idx]:
                print(s.ymax, s.xmax, s.name)
    else:
        for idx in range(len(series)):
            series[idx].sort(key=lambda x: x.ymin)
            for s in series[idx]:
                print(s.ymin, s.xmin, s.name)

    if args.show:
        plt.show()
    else:
        print('Opening.')
        subprocess.call(['open', outpath])
