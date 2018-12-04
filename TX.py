#from pylab import *
import numpy as np
import random
import math
import matplotlib.pyplot as plt

class Transmitter:
    __seq_map = np.array([
        [1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0],
        [1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1],
        [0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1],
        [0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0],
        [1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1],
        [1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1],
        [1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1],
        [1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1],
        [0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        [0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1],
        [1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0]
    ])

    def __init__(self, bit_length, sample_length, amplitude):

        self.dsss_length = 32
        self.bit_length = bit_length
        self.symbol_length = int(self.bit_length / 4)
        self.chip_length = self.bit_length*8
        self.sample_length = sample_length
        self.amplitude = amplitude
        self.data_length = self.chip_length * self.sample_length

        # self.bits = np.array([0 for _ in range(bit_length)], dtype=float)
        self.bits_tx = np.zeros(self.bit_length)
        self.chips_tx = np.zeros(self.chip_length)
        self.samples_tx_ori = np.zeros(self.data_length)
        self.samples_tx = np.zeros(self.data_length)
        self.noise_tx = np.zeros(self.data_length)

        self.bits_rx = np.zeros(self.bit_length)
        self.chips_rx = np.zeros(self.chip_length)
        self.amp_rx = np.zeros(self.chip_length)
        self.samples_rx = np.zeros(self.data_length)

    def generate_data(self):

        self.bits_tx[16:self.bit_length - 16] = [random.randint(0, 1) for _ in range(self.bit_length - 32)]
        for i in range(0, self.symbol_length):
            seq = 0
            for j in range(0, 4):
                seq = seq + self.bits_tx[i * 4 + j] * pow(2, 3 - j)
            seq = int(seq)
            self.chips_tx[i * self.dsss_length:(i + 1) * self.dsss_length] = self.__seq_map[seq, :]
        ''''''
        # for i in range(len(self.chips_tx)):
        #     self.chips_tx[i] = 1
        ''''''
        for i in range(0, self.chip_length):
            self.samples_tx[i * self.sample_length:(i + 1) * self.sample_length] = self.amplitude * self.get_samples(self.chips_tx[i])

        for i in range(len(self.samples_tx_ori)):
            self.samples_tx_ori[i] += self.samples_tx[i]

        '''add gaussian noise'''
        mu = 0.0
        sig = 0
        '''len(self.samples_tx)'''
        for i in range(64):
            tmp = random.gauss(mu, sig)
            self.samples_tx[i] += tmp
            self.noise_tx[i] = tmp


        num = 1000
        t = [i for i in range(1000)]


    def get_samples(self, chip):

        samples = np.zeros(self.sample_length)
        for i in range(0, self.sample_length):
            samples[i] = math.sin(math.pi*i / (self.sample_length-1))
            # delete zero sample
        if chip == 1:
            return samples
        else:
            return -samples

    def draw_samples(self, show_length, samples):

        x = np.arange(0.0, show_length, 1.0)
        plt.figure()
        plt.yticks(np.arange(-self.amplitude, self.amplitude + 1, 2.0))
        plt.plot(x, samples[0:show_length], color="red", lw=1.0, linestyle="-")
        plt.grid()
        plt.show(block=False)















