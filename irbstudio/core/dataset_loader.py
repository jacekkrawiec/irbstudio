import pandas as pd
from typing import Dict
from core.dataset import InputDataset


class DatasetLoader:
    def __init__(self, column_mapping: Dict[str, str]):
        """
        :param column_mapping: dict mapping canonical column names
               (e.g., 'default_flag') to user dataset column names (e.g., 'df_flag').
        """
        self.column_mapping = column_mapping

    def load_from_csv(self, path: str, model_type: str) -> InputDataset:
        # Load raw data
        raw_df = pd.read_csv(path)

        # Rename columns according to mapping (inverse mapping for rename)
        rename_map = {v: k for k, v in self.column_mapping.items()}
        df = raw_df.rename(columns=rename_map)

        # Optional: convert date columns to datetime
        if "observation_date" in df.columns:
            df["observation_date"] = pd.to_datetime(df["observation_date"], errors='coerce')

        # Additional type conversions can be added here

        # Create InputDataset instance (runs internal validation)
        data = InputDataset.from_combined(df, model_type=model_type) #I don;t really get the logic with defining model types. IF we keep it like this we force user to load the same csv multiple times
        return data
