"""Basic KNN; scaling is fitted only when the pipeline fits training data."""
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_classifier():
    return make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5, weights='uniform', metric='minkowski', p=2))
