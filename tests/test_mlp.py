import unittest
import numpy as np
import pytest
from sklearn.datasets import make_classification

@pytest.fixture
def synthetic_data():
    n_samples = 100
    n_features_per_group = 2
    n_longitudinal_groups = 2
    n_non_longitudinal = 2
    random_state = 42

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_longitudinal_groups * n_features_per_group + n_non_longitudinal,
        random_state=random_state,
    )

    features_group = [
        list(range(i * n_features_per_group, (i + 1) * n_features_per_group)) for i in range(n_longitudinal_groups)
    ]

    for i in range(n_longitudinal_groups - 1):
        X[:, features_group[i + 1]] = X[:, features_group[i]] + np.random.normal(
            0, 0.1, size=(n_samples, n_features_per_group)
        )

    return X, y, features_group, n_non_longitudinal


class TestTemporalMLP(unittest.TestCase):
    def test_forward_pass(self, synthetic_data):
        _, _, features_group, _ = synthetic_data

        y_pred = net.predict(X)
        self.assertEqual(y_pred.shape, (8,))
        self.assertTrue(np.all(np.logical_or(y_pred == 0, y_pred == 1)))

if __name__ == '__main__':
    unittest.main()