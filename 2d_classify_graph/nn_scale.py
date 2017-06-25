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
from sklearn.preprocessing import MinMaxScaler

import demo_dataset
from draw import draw_plot


HISTORY_PATH = 'history2'
FIGURE_PATH = 'figures2'
EXP1_MODEL_PATH_TMPL = os.path.join(
    HISTORY_PATH, 'exp2_{dn}_{smin}_{smax}_ep{{epoch:04d}}_{round}-model.hdf5')
EXP1_HISTORY_PATH_TMPL = os.path.join(
    HISTORY_PATH, 'exp2_{dn}_{smin}_{smax}_{round}-history.json')
EXP1_TIMESPENT_PATH_TMPL = os.path.join(
    HISTORY_PATH, 'exp2_{dn}_{smin}_{smax}_{round}-time.txt')

EPOCHS_MAX = 300
PERIOD = 10
TEST_ROUND_N = 2


class OverfittedCheck(Callback):
    def __init__(self, verbose=0, verbose_period=50):
        super(OverfittedCheck, self).__init__()
        self.stopped_epoch = 0
        self.verbose = verbose
        self.verbose_period = verbose_period

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get('acc')
        if current is None:
            warnings.warn('Early stopping requires "acc" available!',
                          RuntimeWarning)

        if current == 1.0:
            self.stopped_epoch = epoch
            self.model.stop_training = True

        if self.verbose > 0 and epoch % self.verbose_period == 0:
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


def exp2(X, y, data_name, scale_min, scale_max):
    optimizer = Adam()
    batch_size = 16

    n_class = len(set(y))

    for test_round in range(TEST_ROUND_N):
        model = build_mlp_relu_model(n_class, 100, 3)
        model.summary()

        model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                      metrics=['accuracy'])

        model_path_tmpl = EXP1_MODEL_PATH_TMPL.format(
            **{'dn': data_name, 'smin': scale_min, 'smax': scale_max,
               'round': test_round})
        history_path = EXP1_HISTORY_PATH_TMPL.format(
            **{'dn': data_name, 'smin': scale_min, 'smax': scale_max,
               'round': test_round})
        timespent_path = EXP1_TIMESPENT_PATH_TMPL.format(
            **{'dn': data_name, 'smin': scale_min, 'smax': scale_max,
               'round': test_round})

        start_time = time()
        history = model.fit(
            X, to_categorical(y, num_classes=n_class),
            batch_size=batch_size, epochs=EPOCHS_MAX,
            callbacks=[ModelCheckpoint(period=PERIOD, filepath=model_path_tmpl),
                       OverfittedCheck(verbose=1, verbose_period=PERIOD)],
            shuffle=True, verbose=0)
        end_time = time()
        time_spent = end_time - start_time

        with open(history_path, 'w') as f_in:
            json.dump(history.history, f_in)
        with open(timespent_path, 'w') as f_in:
            print(time_spent, file=f_in)

        total_epoch_n = len(history.history['loss'])
        model.save(model_path_tmpl.format(**{'epoch': total_epoch_n}))


def draw_heat(X, y, data_name, scale_min, scale_max):
    # Draw prediction heat map
    for test_round in range(TEST_ROUND_N):
        moi_regex = re.compile(
            'exp2_{dn}_{smin}_{smax}_.*_{round}-model.hdf5'.format(
                **{'dn': data_name, 'smin': scale_min, 'smax': scale_max,
                   'round': test_round}))
        model_paths = [fn for fn in os.listdir(HISTORY_PATH)
                       if moi_regex.match(fn)]
        for model_path in model_paths:
            model = load_model(os.path.join(HISTORY_PATH, model_path))
            model_name = model_path[:-5]

            plot_name = os.path.join(FIGURE_PATH, model_name + '-heat.png')
            draw_plot(model, model_name, plot_name, X, y, reso_step=0.01,
                      draw_class=False)
            print(plot_name)


if __name__ == '__main__':
    #data_name = 'squares'
    #X, y = demo_dataset.get_many_nested_squares(
    #    3, edge_n_func=lambda p: p+5)
    data_name = 'breast'
    X, y = demo_dataset.get_breast_cancer_last2()

    for smin, smax in ((0, 1), (-1, 0), (-0.5, 0.5)):
        scaler = MinMaxScaler(feature_range=(smin, smax))
        X = scaler.fit_transform(X)
        exp2(X, y, data_name, smin, smax)
        draw_heat(X, y, data_name, smin, smax)

    K.clear_session()
