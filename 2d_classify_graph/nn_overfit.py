from collections import defaultdict
import json
import os
import re
from time import time
import warnings

from keras import backend as K
from keras.callbacks import Callback, ModelCheckpoint
from keras.layers import Dense
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.utils import to_categorical
import numpy as np

import demo_dataset
from draw import draw_plot
# hack after import/backend in draw
import matplotlib.pyplot as plt
from pylab import rcParams


HISTORY_PATH = 'history'
FIGURE_PATH = 'figures'
EXP1_MODEL_PATH_TMPL = os.path.join(
    HISTORY_PATH, 'exp1_d{ds}_bch{batch}_ep{{epoch}}_{round}-model.hdf5')
EXP1_HISTORY_PATH_TMPL = os.path.join(
    HISTORY_PATH, 'exp1_d{ds}_bch{batch}_{round}-history.json')
EXP1_TIMESPENT_PATH_TMPL = os.path.join(
    HISTORY_PATH, 'exp1_d{ds}_bch{batch}_{round}-time.txt')

EPOCHS_MAX = 5000
TEST_ROUND_N = 9


class OverfittedCheck(Callback):
    def __init__(self, verbose=0):
        super(OverfittedCheck, self).__init__()
        self.stopped_epoch = 0
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get('acc')
        if current is None:
            warnings.warn('Early stopping requires "acc" available!',
                          RuntimeWarning)

        if current == 1.0:
            self.stopped_epoch = epoch
            self.model.stop_training = True

        if self.verbose > 0 and epoch % 50 == 0:
            print('epoch {}, Acc: {}'.format(epoch, current))

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Stop since training accuracy reaches 100%: epoch {}'.format(
                self.stopped_epoch))


def build_mlp_relu_model(n_class, hidden_dim=2, layer_num=1):
    model = Sequential()
    model.add(Dense(hidden_dim, activation='relu', input_dim=2))
    for i in range(layer_num-1):
        model.add(Dense(hidden_dim, activation='relu'))
    model.add(Dense(n_class, activation='softmax'))

    return model


def exp1(hyper_params):
    optimizer = Adam()

    for test_round in range(TEST_ROUND_N):
        print('Test round {}'.format(test_round))
        for hyper_param in hyper_params:
            data_size_param = hyper_param['data_size_param']
            batch_size = hyper_param['batch_size']
            print('Data size param {}, batch size {}'.format(data_size_param,
                                                             batch_size))

            X, y = demo_dataset.get_many_nested_squares(
                3, edge_n_func=lambda p: p+data_size_param)
            n_class = len(set(y))
            model = build_mlp_relu_model(n_class, 100, 3)
            model.summary()

            model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                          metrics=['accuracy'])

            model_path_tmpl = EXP1_MODEL_PATH_TMPL.format(
                **{'ds': data_size_param, 'batch': batch_size,
                   'round': test_round})
            history_path = EXP1_HISTORY_PATH_TMPL.format(
                **{'ds': data_size_param, 'batch': batch_size,
                   'round': test_round})
            timespent_path = EXP1_TIMESPENT_PATH_TMPL.format(
                **{'ds': data_size_param, 'batch': batch_size,
                   'round': test_round})

            start_time = time()
            history = model.fit(
                X, to_categorical(y, num_classes=n_class),
                batch_size=batch_size, epochs=EPOCHS_MAX,
                callbacks=[ModelCheckpoint(period=50, filepath=model_path_tmpl),
                           OverfittedCheck(verbose=1)],
                shuffle=True, verbose=0)
            end_time = time()
            time_spent = end_time - start_time

            with open(history_path, 'w') as f_in:
                json.dump(history.history, f_in)
            with open(timespent_path, 'w') as f_in:
                print(time_spent, file=f_in)


def draw_epoch_metrics(data_size_param, b2metrics, ylabel, png_prefix):
    d = data_size_param

    fig, ax = plt.subplots()
    lines = []
    color_iter = iter(['red', 'orange', 'yellow', 'green', 'blue', 'magenta',
                       'purple', 'cyan', 'black'])
    for b in sorted(b2metrics.keys()):
        metric_list = b2metrics[b]
        color = next(color_iter)
        p, = ax.plot(range(len(metric_list)), metric_list, color=color,
                     label='batch size {}'.format(b))
        lines.append(p)

    ax.set_xlabel('#epoch')
    ax.set_ylabel(ylabel)
    ax.legend(lines, [l.get_label() for l in lines])
    plt.title('{}-point data set'.format(24*(d+1)))
    plt.savefig('{}-d{}.png'.format(png_prefix, d))
    plt.close()


def draw_time(data_size_param, b2epoch, b2time):
    d = data_size_param

    epoch_list = [v for _, v in sorted(b2epoch.items())]
    time_list = [v for _, v in sorted(b2time.items())]

    ind = np.arange(len(b2epoch))
    width = 0.35
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    rect1, = ax1.bar(ind, epoch_list, width, color='orange')
    rect2, = ax2.bar(ind+width, time_list, width, color='green')
    ax1.set_xlabel('batch size')
    ax1.set_ylabel('#epoch')
    ax2.set_ylabel('time spent')
    ax1.set_xticks(ind + width / 2)
    ax1.set_xticklabels(sorted(b2epoch.keys()))

    ax1.legend((rect1, rect2), ('#epoch', 'time spent'),
               loc='upper center')
    plt.title('{}-point data set'.format(24*(d+1)))
    plt.savefig('time-d{}.png'.format(d))
    plt.close()


def draw_exp1_heat(hp2med_epoch_index):
    for hp, med_i in hp2med_epoch_index.items():
        (data_size_param, batch_size) = hp
        # only when data set is deterministic
        X, y = demo_dataset.get_many_nested_squares(
            3, edge_n_func=lambda p: p+data_size_param)

        # Draw prediction heat map
        moi_regex = re.compile(
            'exp1_d{ds}_bch{batch}_.*_{round}-model.hdf5'.format(
                **{'ds': data_size_param, 'batch': batch_size, 'round': med_i}))
        model_paths = [fn for fn in os.listdir(HISTORY_PATH)
                       if moi_regex.match(fn)]
        for model_path in model_paths:
            model = load_model(os.path.join(HISTORY_PATH, model_path))
            model_name = model_path.split('.', 1)[0]

            plot_name = os.path.join(FIGURE_PATH, model_name + '-heat.png')
            draw_plot(model, model_name, plot_name, X, y, reso_step=0.005,
                      draw_class=False)
            print(plot_name)


def get_patience_list(loss_list):
    max_patience = wait = 0
    best = np.Inf
    patience_list = []

    for loss in loss_list:
        if loss < best:
            best = loss
            wait = 0
        else:
            wait = wait + 1
            max_patience = max(max_patience, wait)
        patience_list.append(max_patience)

    return patience_list


def summarize_exp1(hyper_params):
    hp2med_epoch_index = {}
    d2b2med_time = defaultdict(dict)
    d2b2med_epoch = defaultdict(dict)
    d2b2med_accs = defaultdict(dict)
    d2b2med_losses = defaultdict(dict)

    for hyper_param in hyper_params:
        data_size_param = hyper_param['data_size_param']
        batch_size = hyper_param['batch_size']
        epoch_list = []
        time_list = []
        accs_list = []
        losses_list = []

        for test_round in range(TEST_ROUND_N):
            history_path = EXP1_HISTORY_PATH_TMPL.format(
                **{'ds': data_size_param, 'batch': batch_size,
                   'round': test_round})
            timespent_path = EXP1_TIMESPENT_PATH_TMPL.format(
                **{'ds': data_size_param, 'batch': batch_size,
                   'round': test_round})

            with open(history_path) as f:
                history = json.load(f)
                epoch_list.append(len(history['acc']))
                accs_list.append(history['acc'])
                losses_list.append(history['loss'])
            with open(timespent_path) as f:
                t = f.readline().rstrip()
                time_list.append(t)

        med_i = sorted(range(TEST_ROUND_N),
                       key=lambda i: epoch_list[i])[int((TEST_ROUND_N-1)/2)]
        hp2med_epoch_index[(data_size_param, batch_size)] = med_i

        d2b2med_epoch[data_size_param][batch_size] = epoch_list[med_i]
        d2b2med_time[data_size_param][batch_size] = time_list[med_i]
        d2b2med_accs[data_size_param][batch_size] = accs_list[med_i]
        d2b2med_losses[data_size_param][batch_size] = losses_list[med_i]

        print('{} {} {}'.format(
            (data_size_param+1)*24, batch_size,
            ' '.join([str(i) for i in sorted(epoch_list)])))

    draw_exp1_heat(hp2med_epoch_index)

    for d in d2b2med_epoch.keys():
        b2med_epoch = d2b2med_epoch[d]
        b2med_time = d2b2med_time[d]
        draw_time(d, b2med_epoch, b2med_time)

    w, h = rcParams['figure.figsize']
    rcParams['figure.figsize'] = w*2, h
    for d, b2med_accs in d2b2med_accs.items():
        draw_epoch_metrics(d, b2med_accs, ylabel='training acc.',
                           png_prefix='acc')
    hack_accs = d2b2med_accs[5]
    del hack_accs[32]
    del hack_accs[64]
    draw_epoch_metrics(5, hack_accs, ylabel='training acc.',
                       png_prefix='acc_trunc')

    for d, b2med_losses in d2b2med_losses.items():
        draw_epoch_metrics(d, b2med_losses, ylabel='loss', png_prefix='loss')

    for d in d2b2med_losses.keys():
        b2patience = {b: get_patience_list(losses)
                      for b, losses in d2b2med_losses[d].items()}
        draw_epoch_metrics(d, b2patience,
                           ylabel='patience (on loss) to continue',
                           png_prefix='patience')


def set_exp1_hyperparams():
    d2b = {
        1: [1, 2, 4, 8, 16, 32],
        5: [1, 2, 4, 8, 16, 32, 64],
        17: [1, 2, 4, 8, 16, 32, 64, 128, 256],
        53: [1, 2, 4, 8, 16, 32, 64, 128, 256]
    }
    hyper_params = []
    for d in sorted(d2b.keys()):
        for b in d2b[d]:
            hyper_params.append({'data_size_param': d, 'batch_size': b})

    return hyper_params


if __name__ == '__main__':
    hp = set_exp1_hyperparams()
    exp1(hp)
    summarize_exp1(hp)
    K.clear_session()
