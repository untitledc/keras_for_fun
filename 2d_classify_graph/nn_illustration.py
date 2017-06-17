from keras.models import Sequential
from keras.layers import Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import to_categorical
import numpy as np

import demo_dataset
from draw import draw_plot


def build_simple_linear_model(model, n_class):
    model.add(Dense(2, activation='linear', input_dim=2))
    model.add(Dense(n_class, activation='softmax'))


def build_simple_relu_model(model, n_class, hidden_dim=2):
    model.add(Dense(hidden_dim, activation='relu', input_dim=2))
    model.add(Dense(n_class, activation='softmax'))


def build_simple_regressor(model, hidden_dim=10):
    model.add(Dense(hidden_dim, activation='relu', input_dim=2))
    #model.add(Dense(hidden_dim, activation='sigmoid', input_dim=2))
    model.add(Dense(1, activation='linear'))


def my_init_w1(shape, dtype=None):
    return np.array([[1, 1, 1, 1, 1, 1, 1],
                     [0, 0, 0, 0, 0, 0, 0]], dtype=float)


def my_init_b1(shape, dtype=None):
    return np.array([0, -0.2, -0.4, -0.4, -0.6, -0.8, -0.8])


def my_init_w2(shape, dtype=None):
    return np.array([5, -10, 5, 5, -10, 5, 5], dtype=float).reshape(7, 1)


def build_my_init_regressor(model):
    model.add(Dense(10, activation='relu', input_dim=2,
                    kernel_initializer=my_init_w1, bias_initializer=my_init_b1))
    model.add(Dense(1, activation='linear', kernel_initializer=my_init_w2))


def build_mlp_relu_model(model, n_class, hidden_dim=2, layer_num=1):
    model.add(Dense(hidden_dim, activation='relu', input_dim=2))
    for i in range(layer_num-1):
        model.add(Dense(hidden_dim, activation='relu'))
    model.add(Dense(n_class, activation='softmax'))


def build_check_weight_model(model, n_class):
    model.add(Dense(3, activation='linear', input_dim=2))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(2, activation='softmax'))


def build_mess_around_model(model, n_class):
    model.add(Dense(20, input_dim=2))
    model.add(LeakyReLU(0.1))
    model.add(Dense(20))
    model.add(LeakyReLU(0.1))
    model.add(Dense(20))
    model.add(LeakyReLU(0.1))
    model.add(Dense(20, activation='tanh'))
    model.add(Dense(n_class, activation='softmax'))


def build_mess_around_model2(model, n_class):
    model.add(Dense(500, input_dim=2))
    model.add(LeakyReLU(0.1))
    model.add(Dense(300))
    model.add(LeakyReLU(0.1))
    model.add(Dense(200))
    model.add(LeakyReLU(0.1))
    model.add(Dense(200, activation='sigmoid'))
    model.add(Dense(200, activation='sigmoid'))
    model.add(Dense(200, activation='sigmoid'))
    model.add(Dense(200, activation='sigmoid'))
    model.add(Dense(n_class, activation='softmax'))


def main():
    is_classified = False
    print_weight = True

    #X, y = demo_dataset.get_breast_cancer_last2()
    #X, y = demo_dataset.get_iris()
    #X, y = demo_dataset.get_separable_dummy(4)
    #X, y = demo_dataset.get_nested_squares()
    #X, y = demo_dataset.get_many_nested_squares(3)
    X, y = demo_dataset.get_interleaved_1d()
    n_class = len(set(y))

    model = Sequential()
    #build_check_weight_model(model, n_class)
    #build_mess_around_model2(model, n_class)
    #build_simple_linear_model(model, n_class)
    #build_simple_relu_model(model, n_class)
    #build_simple_relu_model(model, n_class, 20)
    #build_mlp_relu_model(model, n_class, 200, 3)
    #build_mlp_relu_model(model, n_class, 200, 3)
    #build_simple_regressor(model, 10)
    build_my_init_regressor(model)

    print('initial weights')
    for model_weight in model.get_weights():
        print(model_weight.shape)
        print(model_weight)

    if is_classified:
        model.compile(optimizer='sgd', loss='categorical_crossentropy',
                      metrics=['accuracy'])
        model.fit(X, to_categorical(y, num_classes=n_class),
                  batch_size=1,
                  epochs=50,
                  shuffle=True)
    else:
        model.compile(optimizer='rmsprop', loss='mean_squared_error')
        model.fit(X, y,
                  batch_size=1,
                  epochs=50,
                  shuffle=True,
                  verbose=0)

    print('model summary')
    model.summary()
    model_weight_list = model.get_weights()

    if print_weight:
        print('model weights')
        for model_weight in model_weight_list:
            print(model_weight.shape)
            print(model_weight)

    # title for the plots
    title = 'FNN linear layer'

    plot_name = 'fnn-lin.png'
    draw_plot(model, title, plot_name, X, y, reso_step=0.005)

    #print(model.predict(np.array([[0.3, 0.4]])))
    print('prediction')
    print(model.predict(np.array([[0.0, 0.0]])))
    print(model.predict(np.array([[0.2, 0.0]])))
    print(model.predict(np.array([[0.4, 0.0]])))
    print(model.predict(np.array([[0.6, 0.0]])))
    print(model.predict(np.array([[0.8, 0.0]])))
    print(model.predict(np.array([[1.0, 0.0]])))

    #print(model_weight_list[0].shape)
    #print(model_weight_list[1].shape)
    #print(model_weight_list[2].shape)
    #print(model_weight_list[3].shape)

    #for test_in in np.arange(0, 1.2, 0.2):
    #    o1_before = test_in * model_weight_list[0][0] + model_weight_list[1]
    #    o1 = np.array([b if b > 0 else 0 for b in o1_before])
    #    print(o1.dot(model_weight_list[2])+model_weight_list[3])


if __name__ == '__main__':
    main()
