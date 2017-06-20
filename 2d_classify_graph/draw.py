import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np


def get_pred(predictor, X, draw_class=True):
    pred = predictor.predict(X)
    if pred.ndim == 1:
        return pred

    if pred.ndim == 2:
        if pred.shape[1] == 1:
            return pred[:, 0]

        if draw_class:
            return pred.argmax(axis=1)
        else:
            return pred[:, 1]
    else:
        raise Exception('cannot recognize output of the predictor')


def draw_plot(predictor, title, filename, X, y, reso_step=0.01,
              draw_class=True):
    n_class = len(set(y))
    if not draw_class and n_class > 2:
        print('drawing real output value is not supported in multi-class')
        return -1

    cmap = plt.cm.Paired if n_class > 2 else plt.cm.RdBu
    # create a mesh to plot in
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, reso_step),
                         np.arange(y_min, y_max, reso_step))
    pred_flat = get_pred(predictor, np.c_[xx.ravel(), yy.ravel()],
                         draw_class=draw_class)
    #print('pred min: {0}, max: {1}'.format(min(pred_flat), max(pred_flat)))

    # Put the result into a color plot
    pred = pred_flat.reshape(xx.shape)
    color_norm = matplotlib.colors.Normalize(0.0, 1.0, clip=True)
    img = plt.imshow(pred, extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                     origin='lower', interpolation='nearest', cmap=cmap,
                     norm=color_norm, alpha=0.5)
    if not draw_class:
        plt.colorbar(img, cmap=cmap, norm=color_norm)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title)

    plt.savefig(filename)
    plt.close()


def main():
    size = 30
    dot_size = 20
    color1 = plt.get_cmap('Reds')
    alpha1 = 1
    color2 = plt.get_cmap('Greens')
    alpha2 = 0.5
    xx, yy = np.meshgrid(np.arange(size), np.arange(size))
    plt.scatter(xx.ravel(), yy.ravel(), c=xx.ravel()/size,
                cmap=color1, s=dot_size)
    plt.savefig('test.png')
    plt.close()
    plt.scatter(xx.ravel(), yy.ravel(), c=yy.ravel()/size,
                cmap=color2, s=dot_size)
    plt.savefig('test2.png')
    plt.close()
    plt.scatter(xx.ravel(), yy.ravel(), c=xx.ravel()/size,
                cmap=color1, alpha=alpha1, s=dot_size)
    plt.scatter(xx.ravel(), yy.ravel(), c=yy.ravel()/size,
                cmap=color2, alpha=alpha2, s=dot_size)
    plt.savefig('test3.png')
    plt.close()


if __name__ == '__main__':
    main()
