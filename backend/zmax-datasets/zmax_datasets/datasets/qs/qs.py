from functools import cached_property
from pathlib import Path

import pandas as pd

from zmax_datasets.datasets.zmax import (
    Dataset,
    Recording,
)
from zmax_datasets.utils.exceptions import (
    MultipleSleepScoringFilesFoundError,
    SleepScoringFileNotFoundError,
)

_SCORING_MAPPING_FILE = Path(__file__).parent / "qs_scoring_files.csv"
_SCORING_MAPPING_FILE_COLUMNS = ["session_id", "scoring_file"]
_SUBJECT_ID = "s1"


class QS(Dataset):
    @cached_property
    def _scoring_mapping(self) -> pd.DataFrame:
        return pd.read_csv(
            _SCORING_MAPPING_FILE,
            names=_SCORING_MAPPING_FILE_COLUMNS,
        )

    @classmethod
    def _extract_ids_from_zmax_dir(cls, zmax_dir: Path) -> tuple[str, str]:
        return _SUBJECT_ID, zmax_dir.name

    def _get_sleep_scoring_file(self, recording: Recording) -> Path:
        matching_rows = self._scoring_mapping[
            self._scoring_mapping["session_id"] == recording.session_id
        ]

        if matching_rows.empty:
            raise SleepScoringFileNotFoundError(
                f"No scoring file found for {recording}."
            )

        if (scoring_files_count := len(matching_rows)) > 1:
            raise MultipleSleepScoringFilesFoundError(
                f"Multiple scoring files ({scoring_files_count}) found for {recording}."
            )

        return self._sleep_scoring_dir / matching_rows["scoring_file"].iloc[0]
