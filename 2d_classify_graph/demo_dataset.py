from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def preprocess(X, y, ratio=0.5):
    X_train, _, y_train, _ = train_test_split(
        X, y, train_size=ratio, stratify=y, random_state=42)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_train)

    return X_scaled, y_train


def get_breast_cancer_last2():
    dataset = datasets.load_breast_cancer()
    X = dataset.data[:, [16, 17]]

    return preprocess(X, dataset.target, 0.1)


def get_iris(f1_idx=0, f2_idx=1):
    dataset = datasets.load_iris()
    X = dataset.data[:, [f1_idx, f2_idx]]

    return preprocess(X, dataset.target, 0.8)

