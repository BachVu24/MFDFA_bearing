"""Basic linear discriminant analysis with empirical class priors."""
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def build_classifier():
    return make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver='svd'))
