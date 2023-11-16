from src.BasicJitter import BasicJitter

import numpy as np
import matplotlib.pyplot as plt

NB_OF_BAR = 200
IMAGE_DIR = 'results/figures/'

def main ():
    print("_____Menu_____")
    print("\n\
    1: Get plot of basic jitter \n\
    2: Get plot of convoluted jitter\n\
    3: Jitter Identification\n")
    
    ch = int(input('Select menu: '))
    
    if ch == 1:
        basicjitter = BasicJitter()
        print(basicjitter.name)

        ## tophat distribution
        mean_tophat = 0.5
        Fs_tophat = NB_OF_BAR
        width_tophat = 1/5

        t_tophat, y_tophat = basicjitter.tophat(mean_tophat - width_tophat/2, mean_tophat + width_tophat/2, Fs_tophat)
        tophat_title = '{}, mean: {}, width: {}, Fs: {}'.format('tophat', mean_tophat, width_tophat, Fs_tophat)

        basicjitter.get_figure(t_tophat, y_tophat, Fs_tophat, title = tophat_title)

        ## gaussian distribution
        mean_gauss = 0.5
        std_dev = 0.01
        Fs_gauss = NB_OF_BAR

        t_gauss, y_gauss = basicjitter.gaussian(mean_gauss, std_dev)
        gauss_title = '{}, mean: {}, std_dev: {}, Fs: {}'.format('gauss', mean_gauss, std_dev, Fs_gauss)

        basicjitter.get_figure(t_gauss, y_gauss, Fs_gauss, title = gauss_title)

        ## sinusoidal distribution
        null_freq = 10

        Fs_sine = NB_OF_BAR
        mean_sine = 0.5
        width = 0.765 / null_freq

        t_sine, y_sine = basicjitter.sine(mean_sine, width, Fs_sine)
        sine_title = '{}, mean: {}, null_freq: {}, Fs: {}'.format('gauss', mean_sine, null_freq, Fs_sine)

        basicjitter.get_figure(t_sine, y_sine, Fs_sine, title = sine_title)
    
    if ch == 2:
        basicjitter = BasicJitter()
        print("Convoluted jitter")
        Fs = NB_OF_BAR

        start_time = 0.2
        end_time = 0.8
        mean = 0.5

        std_dev = 0.024

        t, y_gaussian = basicjitter.gaussian(0.5, std_dev, Fs, start_time, end_time)

        null_freq = 10

        width_sine = 0.765 / null_freq
        width_tophat = 0.2 ## from specs, 0.4 - 0.6s

        t, y_sine = basicjitter.sine(mean, width_sine, Fs, start_time, end_time)
        t, y_hat = basicjitter.tophat(mean - width_tophat/2, mean + width_tophat/2, Fs, start_time, end_time)
        
        y_conv_ident = np.convolve(y_sine, y_gaussian, 'same')
        y_conv_ident = np.convolve(y_conv_ident, y_hat, 'same')

        title_gauss = 'Gauss, mean = {}, std_dev = {}'.format(mean, std_dev)
        title_tophat = 'Tophat, mean = {}, width = {}'.format(mean, width_tophat)
        title_sine = 'Sine, mean = {}, null_freq = {}'.format(mean, null_freq)

        title = 'Convoluted Jitter, Fs: {}\n{}\n{}\n{}'.format(Fs, title_gauss, title_tophat, title_sine)

        fig, axs = plt.subplots(nrows=1, figsize=(7.5, 7.5))
        axs.plot(t,y_conv_ident,'r-')

        # axs.plot(t, y_gaussian, 'k-', alpha = 0.2)
        # axs.plot(t, y_hat, 'k--', alpha = 0.2)
        # axs.plot(t, y_sine, 'r--', alpha = 0.2)

        axs.set_title(title)
        axs.set_ylabel('Probability Density [-]')
        axs.set_xlabel('Time [s]')

        plt.show()
    
    if ch == 3:
        print("Jitter identification")
    
if __name__ == '__main__':
    main()