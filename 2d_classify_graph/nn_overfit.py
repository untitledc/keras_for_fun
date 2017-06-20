import json
import os
from time import time
import warnings

from keras import backend as K
from keras.callbacks import Callback, ModelCheckpoint
from keras.layers import Dense
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.utils import to_categorical

import demo_dataset
from draw import draw_plot


HISTORY_PATH = 'history'
EXP1_MODEL_PATH_TMPL = os.path.join(
    HISTORY_PATH, 'exp1_d{ds}_bch{batch}_ep{{epoch}}_{round}-model.hdf5')
EXP1_HISTORY_PATH_TMPL = os.path.join(
    HISTORY_PATH, 'exp1_d{ds}_bch{batch}_{round}-history.json')
EXP1_TIMESPENT_PATH_TMPL = os.path.join(
    HISTORY_PATH, 'exp1_d{ds}_bch{batch}_{round}-time.txt')

EPOCHS_MAX = 5000


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


def exp1():
    optimizer = Adam()

    d2b = {
        1: [1, 2, 4, 8, 16, 32],
        5: [1, 2, 4, 8, 16, 32, 64],
        17: [1, 2, 4, 8, 16, 32, 64, 128, 256],
        53: [1, 2, 4, 8, 16, 32, 64, 128, 256]
    }
    hyper_params = []
    for d in d2b:
        for b in d2b[d]:
            hyper_params.append({'data_size_param': d, 'batch_size': b})

    for test_round in range(10):
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

    # Draw prediction heat map
    model_paths = [fn for fn in os.listdir(HISTORY_PATH)
                   if fn.endswith('-model.hdf5')]
    for model_path in model_paths:
        model = load_model(os.path.join(HISTORY_PATH, model_path))
        model_name = model_path.split('.', 1)[0]

        plot_name = os.path.join(HISTORY_PATH, model_name + '-class.png')
        draw_plot(model, model_name, plot_name, X, y, reso_step=0.005)
        plot_name = os.path.join(HISTORY_PATH, model_name + '-heat.png')
        draw_plot(model, model_name, plot_name, X, y, reso_step=0.005,
                  draw_class=False)


if __name__ == '__main__':
    exp1()
    K.clear_session()
