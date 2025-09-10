from training import dataset_builder
import pandas as pd

df = pd.read_csv('./training/data/s00.csv')
dataset = dataset_builder.RandomMaskedSequenceDataset(df, sample_interval=1, sequence_length=10, mask_prob=0.3, mask_value=0.0)
x, y = dataset[0]

print (len(dataset))
print (x.shape)
print (y.shape)

print (x)
print (y)