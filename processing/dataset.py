class EEGDataset:
    def __init__(
        self,
        data,
        sample_interval,
    ):
        self.data = data
        self.sample_interval = sample_interval

    def get_dataset(self):
        return self.data

class EEGDatasetCollection:
    def __init__(self, datasets):
        self.datasets = datasets

    def get_datasets(self):
        return self.datasets
