import os
import pandas as pd
from processing import dataset, pipeline
from processing.signal import bands
from processing.core import rename, select

TARGET_INTERVAL = 1
BAND_NAMES = ['delta', 'theta', 'low_alpha', 'high_alpha',
              'low_beta', 'high_beta', 'low_gamma', 'high_gamma']


def run_pipeline(input_csv, output_csv, sample_interval, processors):
    """Generic helper to run a pipeline on EEG data and save output."""
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    df = pd.read_csv(input_csv)
    eeg_dataset = dataset.EEGDataset(data=df, sample_interval=sample_interval)

    data_pipeline = pipeline.Pipeline(processors=processors)
    processed_dataset = data_pipeline.run(eeg_dataset)

    processed_dataset.save(output_csv)


def process_arithmetic():
    arithmetic_dir = './data/arithmetic'
    output_dir = './training/data'
    os.makedirs(output_dir, exist_ok=True)

    csv_files = [f for f in os.listdir(arithmetic_dir) if f.endswith('.csv')]
    for file in csv_files:
        input_csv = os.path.join(arithmetic_dir, file)
        output_csv = os.path.join(output_dir, file)

        df = pd.read_csv(input_csv)
        # sample_interval = duration / num_samples
        sample_interval = 60 / len(df)

        run_pipeline(
            input_csv=input_csv,
            output_csv=output_csv,
            sample_interval=sample_interval,
            processors=[bands.ComputeBandPowers(target_interval=TARGET_INTERVAL)]
        )


def process_sleepy():
    run_pipeline(
        input_csv='./data/sleepy/acquiredDataset.csv',
        output_csv='./training/data/sleepy.csv',
        sample_interval=1,
        processors=[
            rename.RenameColumns(
                original_names=['delta', 'theta', 'lowAlpha', 'highAlpha',
                                'lowBeta', 'highBeta', 'lowGamma', 'highGamma'],
                new_names=BAND_NAMES
            ),
            select.SelectColumns(columns=BAND_NAMES)
        ]
    )


def process_confusion():
    run_pipeline(
        input_csv='./data/confusion/EEG_data.csv',
        output_csv='./training/data/confusion.csv',
        sample_interval=0.5,
        processors=[
            rename.RenameColumns(
                original_names=['Delta', 'Theta', 'Alpha1', 'Alpha2',
                                'Beta1', 'Beta2', 'Gamma1', 'Gamma2'],
                new_names=BAND_NAMES
            ),
            select.SelectColumns(columns=BAND_NAMES)
        ]
    )


if __name__ == "__main__":
    process_arithmetic()
    process_sleepy()
    process_confusion()
