from TX import Transmitter
from Channel import Channel
from RX import Receiver
import numpy as np
import matplotlib.pyplot as plt
from math import log10,sqrt, sin, pi
# from pylab import *
import random

def simulate():
    channel = Channel(5)
    tx1 = Transmitter(1024, 33, 1)
    tx2 = Transmitter(1024, 33, 1)
    tx1.generate_data()
    tx2.generate_data()


    # plt.figure()
    # print(tx1.samples_tx.shape[0])
    # tx1.draw_samples(1024, tx1.samples_tx)
    #
    # plt.figure()
    sample_offset = 106
    tx = channel.overlay(tx1, tx2, sample_offset)
    # #channel.add_noise(tx)
    # tx.draw_samples(2048, tx.samples_tx)
    num = 1024
    b = 1024
    e = b+num
    t = [i for i in range(num)]
    y = tx1.samples_tx[b:e]
    y_nonoise = tx1.samples_tx_ori[b:e]
    plt.figure()
    plt.title("tx samples")
    plt.plot(t,y,'r.',markersize=10)
    plt.plot(t,y_nonoise,'r')

    Ps = 0
    for sample in y_nonoise:
        Ps += (sample ** 2)
    Ps /= len(y_nonoise)

    Pn = 0
    for sample in tx1.noise_tx:
        Pn += (sample ** 2)
    Pn /= len(tx1.noise_tx)
    snr = 10*log10(sqrt((Ps)/Pn))
    print(str(snr)+'dB')
    rx = Receiver(33, tx)
    rx.add_offset(sample_offset)
    rx.add_tx(tx1)
    rx.add_tx(tx2)
    rx.m_zig2_init()
    rx.m_zig2_direct()

    # rx.m_zig2_inverse()

    # tx1.draw_samples(2047, tx1.amp_rx)
    # tx2.draw_samples(2047, tx2.amp_rx)

    print(np.mean(tx1.amp_rx))
    print(np.mean(tx2.amp_rx))

    plt.show()

    zigra_samples_list = []

    err1, err2 = 0,0
    err_distribution_a = np.zeros(17)
    err_distribution_b = np.zeros(17)

    for i in range(len(tx1.chips_rx)):
        if int(tx1.chips_rx[i]) != int(tx1.chips_tx[i]):
            err1 += 1
            err_distribution_a[i//512] += 1


    for i in range(len(tx2.chips_rx)):
        if int(tx2.chips_rx[i]) != int(tx2.chips_tx[i]):
            err2 += 1
            err_distribution_b[i//512] += 1

    print(err1)
    print(err2)
    print(err_distribution_a)
    print(err_distribution_b)

def gen_halfsin(num):
    res = []
    for i in range(num):
        res.append(sin(pi/num *i))
    return np.array(res)

def unit_test():
    a = [1,1,-1,1]
    b = [-1,1,1,-1]
    std_halfsin = gen_halfsin(32)
    sample_a, sample_b = [],[]
    for i in range(len(a)):
        for j in std_halfsin:
            sample_a.append(a[i]*j)
            sample_b.append(b[i]*j)
    sample_offset = 7
    overlap = np.array(sample_a[:sample_offset] + list(np.array(sample_a[sample_offset:])+(np.array(sample_b[:-sample_offset]))))
    print(overlap)
    Aa =0.8
    print(Aa)
    estimate_a = np.array([Aa*x for x in sample_a[7:32]])
    samples_b_slice_head = overlap[7:32]-estimate_a
    Ab = sum(samples_b_slice_head) / sum(std_halfsin[:25])
    print(Ab)
simulate()
# snr = 20# noise = np.random.normal(0, 1, mod_len)
#  noise_power = np.sum(np.multiply(np.abs(noise), np.abs(noise)))/mod_len
#  for i in range(num_of_seq):#     test_zig_seq = []#     for j in range(i, i+num_of_seq):#         test_zig_seq += zigbee.symbol_seq[j % num_of_seq]#     test_zig_seq = np.tile(test_zig_seq, int(symbols_per_chirp/num_of_seq))#     test_zig_samples = zigbee.mod(test_zig_seq)##     zigra_sample = test_zig_samples[:-half_spc][16::32]##     sig_power = np.sum(np.multiply(np.abs(zigra_sample), np.abs(zigra_sample)))/mod_len#     k = (sig_power/noise_power)*np.power(10, -snr/10)#     noise = np.sqrt(k)*noise##     # zigra_sample = np.add(zigra_sample, noise)#     zigra_samples_list.append(zigra_sample)# zigra_result, zigra_fft = lora.de_mod(zigra_samples_list)# for i in range((1 << bits_per_symbol)):#     lora.show_freq(zigra_samples_list, i)#     lora.show_fft(zigra_fft, i)