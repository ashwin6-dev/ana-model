from ..pipeline import Processor
from ..dataset import EEGDataset

class RenameColumns(Processor):
    def __init__(
        self,
        original_names: list[str],
        new_names: list[str],
    ):
        self.original_names = original_names
        self.new_names = new_names

    def execute(self, input_data):
        self.input_data = input_data
        df = input_data.get_dataset()
        renamed_df = df.rename(columns=dict(zip(self.original_names, self.new_names)))
        return EEGDataset(renamed_df, sample_interval=input_data.sample_interval)