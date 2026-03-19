import pandas as pd


def load_sample_data():
    return pd.DataFrame(
        [
            {"field_id": "F001", "state": "OH", "year": 2021, "crop": "Corn", "acres": 120, "precipitation": 24.5},
            {"field_id": "F002", "state": "OH", "year": 2021, "crop": "Soybeans", "acres": 95, "precipitation": 23.8},
            {"field_id": "F003", "state": "OH", "year": 2022, "crop": "Corn", "acres": 140, "precipitation": 26.1},
            {"field_id": "F004", "state": "IN", "year": 2021, "crop": "Corn", "acres": 110, "precipitation": 22.4},
            {"field_id": "F005", "state": "MI", "year": 2023, "crop": "Wheat", "acres": 80, "precipitation": 27.0},
            {"field_id": "F006", "state": "MI", "year": 2022, "crop": "Soybeans", "acres": 130, "precipitation": 25.3},
        ]
    )