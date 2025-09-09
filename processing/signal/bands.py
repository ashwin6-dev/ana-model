from ..pipeline import Processor
from ..dataset import EEGDataset
from scipy.fft import rfft, rfftfreq
import numpy as np
import pandas as pd

class ComputeBandPowers(Processor):
    BANDS = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'low_alpha': (8, 10),
        'high_alpha': (10, 13),
        'low_beta': (13, 20),
        'high_beta': (20, 30),
        'low_gamma': (30, 60),
        'high_gamma': (60, 100)
    }

    def __init__(self, target_interval=1.0):
        self.target_interval = target_interval
        self.input_data = None

    def execute(self, input_data):
        self.input_data = input_data
        df = input_data.get_dataset()
        band_power_df = self.compute_band_power(df)
        return band_power_df

    def compute_band_power(self, signals: pd.DataFrame) -> pd.DataFrame:
        sample_interval = self.input_data.sample_interval
        samples_per_second = int(int(round(1 / sample_interval)) * self.target_interval)

        all_band_powers = []

        for start in range(0, len(signals), samples_per_second):
            window = signals.iloc[start:start+samples_per_second]
            if len(window) < samples_per_second:
                break

            band_sums = {band: 0.0 for band in self.BANDS}
            n_signals = len(window.columns)

            for column in window.columns:
                signal_window = window[column].values
                band_powers = self.compute_signal_bands(signal_window)
                for band, power in band_powers.items():
                    band_sums[band] += power

            band_avgs = {band: band_sums[band] / n_signals for band in self.BANDS}
            all_band_powers.append(band_avgs)

        return EEGDataset(pd.DataFrame(all_band_powers), sample_interval=1)

    def compute_signal_bands(self, signal):
        fs = 1 / self.input_data.sample_interval
        N = len(signal)
        yf = rfft(signal)
        xf = rfftfreq(N, 1/fs)

        psd = (np.abs(yf) ** 2) / N

        band_powers = {}
        for band, (low, high) in self.BANDS.items():
            idx = np.where((xf >= low) & (xf <= high))
            band_power = np.mean(psd[idx])
            band_powers[band] = band_power

        return band_powers
