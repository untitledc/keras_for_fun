from keras.models import Sequential
from keras.layers import Dense
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import to_categorical

import demo_dataset
from draw import draw_plot


#X, y = demo_dataset.get_breast_cancer_last2()
X, y = demo_dataset.get_iris()
#X, y = demo_dataset.get_separable_dummy(10)
n_class = len(set(y))

model = Sequential()
if True:
    model.add(Dense(2, activation='linear', input_dim=2))
    model.add(Dense(n_class, activation='softmax'))
else:
    model.add(Dense(20, input_dim=2))
    model.add(LeakyReLU(0.1))
    model.add(Dense(20))
    model.add(LeakyReLU(0.1))
    model.add(Dense(20))
    model.add(LeakyReLU(0.1))
    model.add(Dense(20, activation='tanh'))
    model.add(Dense(n_class, activation='softmax'))
model.compile(optimizer='sgd', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X, to_categorical(y, num_classes=n_class),
          batch_size=1,
          epochs=20,
          shuffle=True)

# title for the plots
title = 'FNN linear layer'

plot_name = 'fnn-lin.png'
draw_plot(model, title, plot_name, X, y, reso_step=0.005)


