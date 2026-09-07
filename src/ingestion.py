from pathlib import Path
import pandas as pd
from typing import Tuple


class IngestData:
    def __init__(self) -> None:
        BASE_DIR = Path.cwd()
        DATA_DIR = BASE_DIR / "data" / "raw"

        self.FILE_MAP = {
            "loan": DATA_DIR / "loan.parquet",
            "payment": DATA_DIR / "payment.parquet",
            "clarity_uv": DATA_DIR / "clarity_underwriting_variables.parquet",
            "clarity_csv": DATA_DIR / "clarity_underwriting_dictionary.csv",
        }

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        missing_files = self.check_missing_files()

        if missing_files:
            raise FileNotFoundError(
                f"Following required files are missing:\n {'\n'.join(missing_files)}"
            )

        return (
            pd.read_parquet(self.FILE_MAP["loan"]),
            pd.read_parquet(self.FILE_MAP["payment"]),
            pd.read_parquet(self.FILE_MAP["clarity_uv"]),
        )

    def check_missing_files(self):
        return [str(path) for path in self.FILE_MAP.values() if not path.is_file()]
