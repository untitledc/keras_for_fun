import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from draw import draw_subplot


# import some data and take 2 features to play with
dataset = datasets.load_breast_cancer()
X = dataset.data[:, [16, 17]]
y = dataset.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.9, random_state=42)
scaler = MinMaxScaler()
X = scaler.fit_transform(X_train)
y = y_train

h = .01  # step size in the mesh

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
rbf_svc = [svm.SVC(kernel='rbf', gamma=2**-7, C=2**13).fit(X, y),
           svm.SVC(kernel='rbf', gamma=2**3, C=2**13).fit(X, y),
           svm.SVC(kernel='rbf', gamma=2**-7, C=1).fit(X, y),
           svm.SVC(kernel='rbf', gamma=2**3, C=1).fit(X, y)]

# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

# title for the plots
titles = ['C=2^13, gamma=2^-7',
          'C=2^13, gamma=2^3',
          'C=1, gamma=2^-7',
          'C=1, gamma=2^3']


for i in range(4):
    draw_subplot(rbf_svc[i], titles[i], i+1, X, y, x_min, x_max, y_min, y_max)
    plt.savefig('test{}.png'.format(i))
    plt.close()


