import unittest

import pandas as pd

from eskom_energy_demand_forecasting.config import CONFIG
from eskom_energy_demand_forecasting.dataset import engineer_target, load_eskom_data
from eskom_energy_demand_forecasting.features import build_ml_features


class SmokeTests(unittest.TestCase):
    def test_import_and_basic_feature_build(self) -> None:
        df = load_eskom_data(CONFIG)
        df = engineer_target(df, CONFIG)
        sample = df.iloc[:500].copy()
        X, y = build_ml_features(sample, CONFIG)

        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        self.assertGreater(len(X), 0)
        self.assertEqual(len(X), len(y))


if __name__ == "__main__":
    unittest.main()
