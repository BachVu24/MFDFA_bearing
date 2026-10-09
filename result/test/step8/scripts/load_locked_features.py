"""Step9 entry point: validate the Step8 lock and return only fixed RAW inputs."""
from pathlib import Path
import hashlib
import json
import numpy as np

EXPECTED_FEATURES = ['delta_alpha', 'alpha0']
LOCK_DIRECTORY = Path(__file__).resolve().parents[1] / 'config'
REPOSITORY = Path(__file__).resolve().parents[4]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def load_locked_config(directory=LOCK_DIRECTORY, repository=REPOSITORY):
    directory, repository = Path(directory), Path(repository)
    path = directory / 'locked_feature_set.json'
    expected = (directory / 'locked_feature_set.sha256').read_text().split()[0]
    if digest(path) != expected:
        raise ValueError('Step8 locked config digest mismatch')
    config = json.loads(path.read_text())
    if (config['status'] != 'LOCKED' or config['features'] != EXPECTED_FEATURES
        or config['feature_order'] != EXPECTED_FEATURES or config['representation'] != 'RAW'
        or config['pipeline'] != 'raw' or config['component'] != 'raw'
        or config['feature_reselection_allowed'] is not False
        or config['classification_accuracy_may_change_feature_set'] is not False):
        raise ValueError('Step8 feature/representation policy changed')
    for row in config['upstream_inputs']:
        if digest(repository / row['path']) != row['sha256']:
            raise ValueError(f"Changed upstream evidence: {row['path']}")
    return config

def locked_inputs(frame, channel='output_voltage', config_directory=LOCK_DIRECTORY):
    """Return (eligible metadata, ordered numeric features, excluded RAW metadata).

    Requires explicit boolean QC; do not interpret the string 'False' as true.
    Does not scale, impute, split data, fit models, or change feature selection.
    """
    config = load_locked_config(config_directory)
    if channel not in [config['primary_channel'],config['secondary_channel']]:
        raise ValueError('Unknown channel')
    raw = frame[(frame.pipeline == 'raw') & (frame.component == 'raw') & (frame.channel == channel)].copy()
    if raw.duplicated(['recording','channel']).any():
        raise ValueError('Duplicate full recording/channel observations')
    if not raw.analysis_valid.isin([True,False]).all():
        raise ValueError('QC must contain explicit booleans, not strings or missing values')
    eligible = raw.analysis_valid.eq(True) & np.isfinite(raw[EXPECTED_FEATURES].to_numpy(dtype=float)).all(axis=1)
    metadata = ['recording','state','speed','load','repeat','channel','analysis_valid']
    return (raw.loc[eligible,metadata].copy(), raw.loc[eligible,EXPECTED_FEATURES].copy(),
            raw.loc[~eligible,metadata].copy())
