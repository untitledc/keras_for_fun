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
    #X, y = demo_dataset.get_breast_cancer_last2()
    #X, y = demo_dataset.get_iris()
    #X, y = demo_dataset.get_separable_dummy(4)
    #X, y = demo_dataset.get_nested_squares()
    X, y = demo_dataset.get_many_nested_squares(3)
    n_class = len(set(y))

    model = Sequential()
    #build_check_weight_model(model, n_class)
    #build_mess_around_model2(model, n_class)
    #build_simple_linear_model(model, n_class)
    #build_simple_relu_model(model, n_class)
    #build_simple_relu_model(model, n_class, 200)
    build_mlp_relu_model(model, n_class, 200, 3)
    #model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
    model.compile(optimizer='rmsprop', loss='mean_squared_error',
                  metrics=['accuracy'])
    model.fit(X, to_categorical(y, num_classes=n_class),
              batch_size=1,
              epochs=50,
              shuffle=True)

    print('model summary')
    model.summary()
    print('model weights')
    model_weight_list = model.get_weights()
    for model_weight in model_weight_list:
        print(model_weight.shape)
        print(model_weight)

    # title for the plots
    title = 'FNN linear layer'

    plot_name = 'fnn-lin.png'
    draw_plot(model, title, plot_name, X, y, reso_step=0.005)

    #print(model.predict(np.array([[0.3, 0.4]])))


if __name__ == '__main__':
    main()
