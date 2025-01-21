from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import numpy as np

def rebalance_classes(X_train, y_train):
    """Rebalance the dataset using SMOTE and undersampling."""
    over = SMOTE(sampling_strategy=0.1)
    under = RandomUnderSampler(sampling_strategy=0.33)
    pipeline = Pipeline(steps=[('o', over), ('u', under)])
    X_resampled, y_resampled = pipeline.fit_resample(X_train, y_train)

    # Shuffle the resampled data
    stacked_resampled_data = np.hstack([X_resampled, y_resampled.reshape(-1, 1)])
    np.random.shuffle(stacked_resampled_data)

    X_resampled = stacked_resampled_data[:, :-1]
    y_resampled = stacked_resampled_data[:, -1]
    return X_resampled, y_resampled
