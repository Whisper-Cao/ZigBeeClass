import numpy as np
import math
import matplotlib
from pylab import *

class Receiver:

    def __init__(self, sample_length, tx):

        self.offset_list = np.array([0])
        self.tx_list = np.array([])
        self.sample_length = sample_length
        self.overlay_tx = tx
        self.count = -1

    def add_tx(self, tx):

        self.tx_list = np.append(self.tx_list, tx)
        self.count += 1

    def add_offset(self, offset):

        self.offset_list = np.append(self.offset_list, offset)

    def m_zig2_init(self):

        first_end, last_start, free_samples_first, free_samples_last = self.get_free_samples()

        tx_a = self.tx_list[0]
        tx_b = self.tx_list[1]

        offset = self.offset_list[1]
        chip_offset = int(offset / self.sample_length)
        sample_offset = offset % self.sample_length
        std_samples = self.get_std_samples()


        # num = 1000;
        # t = [i for i in range(1000)]
        # y =

        tx_a.samples_rx[0:first_end] = free_samples_first
        tx_b.samples_rx[tx_b.data_length - offset:tx_b.data_length] = free_samples_last

        for i in range(0, chip_offset):
            chip_a, amp_a = self.estimate(
                std_samples, tx_a.samples_rx[i * self.sample_length:(i + 1) * self.sample_length], 0,
                self.sample_length)

            tx_a.chips_rx[i] = chip_a
            tx_a.amp_rx[i] = amp_a

            index = i + tx_b.chip_length - chip_offset
            chip_b, amp_b = self.estimate(
                std_samples, tx_b.samples_rx[index * self.sample_length:(index + 1) * self.sample_length], 0,
                self.sample_length)
            tx_b.chips_rx[i] = chip_b
            tx_b.amp_rx[i] = amp_b

    def m_zig2_direct(self):
        amp_a_list, amp_b_list = [],[]

        tx_a = self.tx_list[0]
        tx_b = self.tx_list[1]

        offset = self.offset_list[1]
        chip_offset = int(offset / self.sample_length)
        sample_offset = offset % self.sample_length
        std_samples = self.get_std_samples()
        # mid_a = int((chip_offset + tx_a.chip_length)/2)
        mid_a = tx_a.chip_length

        lastamp_a, lastamp_b = 0,0

        for i in range(chip_offset, mid_a-1):

            begin_a_head = i*self.sample_length
            end_a_head = i*self.sample_length + sample_offset
            begin_a_tail = end_a_head
            end_a_tail = begin_a_head+self.sample_length
            chip_a, amp_a = self.estimate(std_samples, tx_a.samples_rx[begin_a_head:end_a_head], 0, sample_offset)
            print(i, chip_a, amp_a)
            tx_a.chips_rx[i] = chip_a
            tx_a.amp_rx[i] = amp_a
            if chip_a == 0:
                amp_a = -amp_a
            amp_a_list.append(abs(amp_a))
            #using first (offset) samples to calc amp_a and estimate the tail
            tx_tail_a = amp_a*std_samples[sample_offset:self.sample_length]
            tx_a.samples_rx[begin_a_tail:end_a_tail] = amp_a*std_samples[sample_offset:]

            begin_b_head = (i-chip_offset)*self.sample_length
            end_b_head = (i-chip_offset+1)*self.sample_length - sample_offset
            begin_b_tail = end_b_head
            end_b_tail = begin_b_head + self.sample_length

            #get overlay samples
            tx_overlay = self.overlay_tx.samples_tx[begin_a_tail:end_a_tail]

            tx_b.samples_rx[begin_b_head:end_b_head] = \
                self.overlay_tx.samples_tx[begin_a_tail:end_a_tail] \
                - tx_a.samples_rx[begin_a_tail:end_a_tail]

            #tx_head_b = tx_b.samples_rx[begin_b:end_b]
            chip_b, amp_b = \
                self.estimate(
                    std_samples, tx_b.samples_rx[begin_b_head:end_b_head], 0, end_b_head-begin_b_head)
            print(i, chip_b, amp_b)
            tx_b.chips_rx[i-chip_offset] = chip_b
            print("gt chip is %d, rx chip is %d\n" %(tx_b.chips_tx[i], tx_a.chips_tx[i]))
            tx_b.amp_rx[i] = amp_b
            # if abs(lastamp_b-amp_b) < 1e-8:
            #     amp_b = lastamp_b
            # else:
            #     lastamp_a = amp_a
            if chip_b == 0:
                amp_b = -amp_b
            amp_b_list.append(abs(amp_b))
            if i != mid_a-1:

                tx_b.samples_rx[begin_b_tail:end_b_tail] = \
                    amp_b * std_samples[self.sample_length - sample_offset:]

                tx_a.samples_rx[begin_a_head+self.sample_length:end_a_head+self.sample_length] = \
                    self.overlay_tx.samples_tx[begin_a_head+self.sample_length:end_a_head+self.sample_length] - \
                    tx_b.samples_rx[begin_b_tail:end_b_tail]
            # tx_overlay = self.overlay_tx.samples_tx[begin_a+self.sample_length:end_a+self.sample_length]
            # tx_tail_b = tx_b.samples_rx[end_b:begin_b+self.sample_length]
            # tx_head_a = tx_a.samples_rx[begin_a+self.sample_length:end_a+self.sample_length]
            x = 111

        c = chip_offset + 1
        for i in range(c):
            tx_b.chips_rx[-c+i] = tx_b.chips_tx[-c+i]

        # for i in range(len(amp_a_list)):
        #     amp_a_list[i] = fabs(amp_a_list[i]-8)
        #     amp_b_list[i] = fabs(amp_b_list[i]-16)
        # t = [i for i in range(len(amp_b_list))]
        # plt.figure()
        # plt.plot(t, amp_a_list,'b.')
        # plt.figure()
        # plt.plot(t, amp_b_list,'r.')

    def m_zig2_inverse(self):

        tx_a = self.tx_list[0]
        tx_b = self.tx_list[1]

        offset = self.offset_list[1]
        chip_offset = int(offset / self.sample_length)
        sample_offset = offset % self.sample_length
        std_samples = self.get_std_samples()
        mid_b = int((tx_b.chip_length - chip_offset)/2)
        # mid_b = 0

        for i in range(tx_b.chip_length-chip_offset, mid_b, -1):

            begin_b = i*self.sample_length - sample_offset
            end_b = i*self.sample_length

            chip_b, amp_b = self.estimate(
                std_samples, tx_b.samples_rx[begin_b:end_b], self.sample_length-sample_offset, self.sample_length)
            # delete last sample when estimate?
            print(i, chip_b, amp_b)
            tx_b.chips_rx[i-1] = chip_b
            tx_b.amp_rx[i-1] = amp_b
            if chip_b == 0:
                amp_b = -amp_b
            tx_b.samples_rx[end_b-self.sample_length:begin_b] = amp_b*std_samples[0:self.sample_length-sample_offset]

            begin_a = (i+chip_offset-1)*self.sample_length+sample_offset
            end_a = (i+chip_offset)*self.sample_length

            tx_a.samples_rx[begin_a:end_a] = \
                self.overlay_tx.samples_tx[begin_a:end_a] - tx_b.samples_rx[end_b-self.sample_length:begin_b]
            chip_a, amp_a = self.estimate(
                std_samples, tx_a.samples_rx[begin_a:end_a], sample_offset, self.sample_length)
            # delete last sample when estimate?
            print(i, chip_a, amp_a)
            tx_a.chips_rx[i-1] = chip_a
            tx_a.amp_rx[i-1] = amp_a
            if chip_a == 0:
                amp_a = -amp_a

            if i != mid_b:

                tx_a.samples_rx[end_a - self.sample_length:begin_a] = amp_a * std_samples[0:sample_offset]

                tx_b.samples_rx[begin_b - self.sample_length:end_b - self.sample_length] = \
                    self.overlay_tx.samples_tx[end_a - self.sample_length:begin_a] - \
                    tx_a.samples_rx[end_a - self.sample_length:begin_a]


    def estimate(self, std_samples, samples, start, end):

        if np.sum(samples) > 0:
            chip = 1
            amp = np.sum(samples) / np.sum(std_samples[start:end])
        else:
            chip = 0
            amp = -np.sum(samples) / np.sum(std_samples[start:end])

        # print(np.sum(samples))
        # print(np.sum(std_samples[start:end]))
        return chip, amp

    def get_free_samples(self):

        first_end = self.offset_list[1]
        last_start = self.offset_list[self.count-1] + self.tx_list[self.count-1].data_length
        # print(first_end, last_start)
        free_samples_first = self.overlay_tx.samples_tx[0:first_end]
        free_samples_last = self.overlay_tx.samples_tx[last_start:self.overlay_tx.data_length]

        return first_end, last_start, free_samples_first, free_samples_last

    def get_std_samples(self):

        std_samples = np.zeros(self.sample_length)
        for i in range(0, self.sample_length):
            std_samples[i] = math.sin(math.pi*i / (self.sample_length-1))
            # delete zero sample
        return std_samples

