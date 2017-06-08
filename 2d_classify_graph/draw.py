import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def draw_subplot(predictor, title, filename, X, y, x_min, x_max, y_min, y_max,
                 reso_step=0.01):
    xx, yy = np.meshgrid(np.arange(x_min, x_max, reso_step),
                         np.arange(y_min, y_max, reso_step))
    pred_flat = predictor.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    pred = pred_flat.reshape(xx.shape)
    plt.imshow(pred, extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               origin='lower', interpolation='nearest',
               cmap=plt.cm.Paired, alpha=0.5)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title)
