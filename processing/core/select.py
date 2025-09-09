from ..pipeline import Processor
from ..dataset import EEGDataset

class SelectColumns(Processor):
    def __init__(
        self,
        columns: list[str],
    ):
        self.columns = columns

    def execute(self, input_data):
        self.input_data = input_data
        df = input_data.get_dataset()
        selected_df = df[self.columns]
        return EEGDataset(selected_df, sample_interval=input_data.sample_interval)