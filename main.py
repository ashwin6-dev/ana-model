import pandas as pd
from processing import dataset, bands

raw_df = pd.read_csv('data/arithmetic/s00.csv')
sample_interval = 60 / len(raw_df)

input_data = dataset.EEGDataset(raw_df, sample_interval)
processor = bands.ComputeBandPowers()
band_power_df = processor.execute(input_data)

print (band_power_df)