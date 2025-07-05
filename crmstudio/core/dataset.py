from dataclasses import dataclass
from typing import Optional, Dict
import pandas as pd


@dataclass
class InputDataset:
    data: pd.DataFrame
    model_type: str  # "PD", "LGD", "EAD"

    # Define required columns per model type as class attribute
    _required_columns_map = {
        "PD": ["obligor_id", "observation_date", "default_flag", "pd_predicted"],
        "LGD": ["obligor_id", "observation_date", "default_flag", "lgd_predicted", "lgd_observed"],
        "EAD": ["obligor_id", "observation_date", "default_flag", "ead_predicted", "ead_observed"],
    }

    def __post_init__(self):
        required_id_cols = ["obligor_id", "facility_id"]
        if not any(col in self.data.columns for col in required_id_cols):
            raise ValueError("Input must contain at least one of 'obligor_id' or 'facility_id'")

        if self.model_type not in self._required_columns_map:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        required_columns = self._required_columns_map[self.model_type]
        missing = [col for col in required_columns if col not in self.data.columns]
        if missing:
            raise ValueError(f"Missing required columns for {self.model_type} model: {missing}")

        n = len(self.data)
        for col in required_columns:
            if len(self.data[col]) != n:
                raise ValueError(f"Column '{col}' length does not match others")

    @classmethod
    def from_combined(cls, data: pd.DataFrame, model_type: str):
        """
        Create ValidationDataset from a combined dataframe containing all model data.
        Validates presence of required columns for the given model type.
        """
        if model_type not in cls._required_columns_map:
            raise ValueError(f"Unsupported model type: {model_type}")

        required_columns = cls._required_columns_map[model_type]

        missing = [col for col in required_columns if col not in data.columns]
        if missing:
            raise ValueError(f"Missing required columns for {model_type}: {missing}")

        # Optionally, subset columns to required + risk_driver_* + optional fields
        risk_drivers = [col for col in data.columns if col.startswith("risk_driver_")]
        optional_fields = ["exposure", "segment_id", "model_version", "score", "observation_date", "default_flag"]
        columns_to_keep = set(required_columns + risk_drivers + optional_fields)
        columns_to_keep = [col for col in data.columns if col in columns_to_keep]

        subset_df = data[columns_to_keep].copy()

        return cls(data=subset_df, model_type=model_type)

    @property
    def y_true(self) -> pd.Series:
        return self.data["default_flag"]

    @property
    def y_pred(self) -> pd.Series:
        pred_col = {
            "PD": "pd_predicted",
            "LGD": "lgd_predicted",
            "EAD": "ead_predicted"
        }[self.model_type]
        return self.data[pred_col]

    @property
    def exposure(self) -> Optional[pd.Series]:
        return self.data.get("exposure")

    @property
    def observation_date(self) -> pd.Series:
        return self.data["observation_date"]

    @property
    def risk_drivers(self) -> Dict[str, pd.Series]:
        return {
            col: self.data[col]
            for col in self.data.columns
            if col.startswith("risk_driver_")
        }

    @property
    def segment_id(self) -> Optional[pd.Series]:
        return self.data.get("segment_id")

    @property
    def model_version(self) -> Optional[pd.Series]:
        return self.data.get("model_version")

    @property
    def obligor_id(self) -> Optional[pd.Series]:
        return self.data.get('obligor_id')
    
    @property
    def facility_id(self) -> Optional[pd.Series]:
        return self.data.get('facility_id')