"""Fixed random forest; tree splits do not require feature standardization."""
from sklearn.ensemble import RandomForestClassifier


def build_classifier():
    return RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=2,
                                  min_samples_leaf=1, max_features='sqrt',
                                  class_weight=None, bootstrap=True, random_state=9609, n_jobs=1)
