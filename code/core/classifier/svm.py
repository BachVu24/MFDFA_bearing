"""Basic RBF SVM with fixed, untuned parameters."""
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_classifier():
    return make_pipeline(StandardScaler(), SVC(C=1.0, kernel='rbf', gamma='scale', class_weight=None))
