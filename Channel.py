import numpy as np
from TX import Transmitter


class Channel:

    def __init__(self, snr):

        self.snr = snr

    def overlay(self, tx_a, tx_b, offset):

        tx = Transmitter(tx_a.bit_length+tx_b.bit_length, tx_a.sample_length, tx_a.amplitude+tx_b.amplitude)
        tx.data_length = offset + tx_b.data_length
        tx.samples_tx = np.zeros(tx.data_length)
        tx.amplitude = tx_a.amplitude + tx_b.amplitude

        tx.samples_tx[0:offset] = tx_a.samples_tx[0:offset]
        tx.samples_tx[offset:tx_a.data_length] = \
            tx_a.samples_tx[offset:tx_a.data_length] + tx_b.samples_tx[0:(tx_b.data_length - offset)]
        tx.samples_tx[tx_a.data_length:tx.data_length] = tx_b.samples_tx[tx_b.data_length - offset:tx_b.data_length]

        return tx

    def add_noise(self, tx):
        snr = 10**(self.snr/10.0)
        signal_power = np.sum(tx.samples_tx**2)/tx.data_length
        noise_power = signal_power/snr
        noise = np.random.randn(tx.data_length)*np.sqrt(noise_power)
        tx.samples_tx = tx.samples_tx + noise








