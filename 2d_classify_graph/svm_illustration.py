from sklearn import svm

import demo_dataset
from draw import draw_plot


#X, y = demo_dataset.get_breast_cancer_last2()
#X, y = demo_dataset.get_iris(1, 3)
X, y = demo_dataset.get_iris()
#X, y = demo_dataset.get_separable_dummy(10)

rbf_svc = [svm.SVC(kernel='rbf', gamma=2**-7, C=2**13).fit(X, y),
           svm.SVC(kernel='rbf', gamma=2**4, C=2**13).fit(X, y),
           svm.SVC(kernel='rbf', gamma=2**-7, C=1).fit(X, y),
           svm.SVC(kernel='rbf', gamma=2**4, C=1).fit(X, y)]

# title for the plots
titles = ['C=2^13, gamma=2^-7', 'C=2^13, gamma=2^4',
          'C=1, gamma=2^-7', 'C=1, gamma=2^4']

for i in range(4):
    plot_name = 'svm{}.png'.format(i)
    draw_plot(rbf_svc[i], titles[i], plot_name, X, y)


