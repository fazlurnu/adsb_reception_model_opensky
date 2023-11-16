from src.BasicJitter import BasicJitter

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

NB_OF_BAR = 200
IMAGE_DIR = 'results/figures/'

def get_cum(freq):
    cum_list = []
    cum = 0
    
    for f in freq:
        cum += f
        cum_list.append(cum)

    cum_list = np.array(cum_list) / cum_list[-1]
        
    return cum_list

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
        sine_title = '{}, mean: {}, null_freq: {}, Fs: {}'.format('sinusoidal', mean_sine, null_freq, Fs_sine)

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
        df1 = pd.read_csv('pos_tables/icao24_4ca8e4_pos_table_1433801924.csv')
        df2 = pd.read_csv('pos_tables/icao24_3461ce_pos_table_1433801924.csv')
        df3 = pd.read_csv('pos_tables/icao24_3461cf_pos_table_1433801924.csv')

        df_list = [df1, df2, df3]

        df_filtered = []

        for df_ in df_list:
            df_filtered_ = df_[['mintime', 'maxtime', 'rawmsg', 'icao24', 'lat', 'lon', 'alt']]
            df_filtered.append(df_filtered_)
            
            # df = pd.concat([df1, df2, df3, df4, df5, df6, df7])

        df = pd.concat(df_filtered)

        df.drop_duplicates(subset=['rawmsg'], inplace = True)
        df.dropna(subset=["lat"], inplace=True)
        df.dropna(subset=["lon"], inplace=True)
        df.dropna(subset=["lat"], inplace=True)

        df_grouped = df.groupby('icao24', group_keys=True).apply(lambda x: x)
        df_grouped['updateinterval'] = df_grouped['mintime'].diff()

        df_grouped = df_grouped[(df_grouped['updateinterval'] > 0.0) & (df_grouped['updateinterval'] < 10)]
        df = df_grouped.reset_index(drop=True)

        for df_ in df_filtered:
            plt.plot(df_['lon'], df_['lat'])

        plt.title('3 Aircrafts Trajectory')
        plt.show()

        ## create histogram
        start = 0.2
        end = 0.8
        nb_of_bar = 100

        input_df = df[(df['updateinterval'] > 0.2) & (df['updateinterval'] < 0.8)]
        input_list = np.array(input_df['updateinterval'].to_list())
        
        bins = np.linspace(start, end, int(((end-start)*nb_of_bar+1)))
        weightsa = np.ones_like(input_list)/float(len(input_list))
        freq_, edge_ = np.histogram(input_list, bins, weights = weightsa)
        print(sum(freq_))
        
        plt.bar(edge_[:-1], freq_*100, width = 1/nb_of_bar, edgecolor = 'k', alpha = 0.8)
        plt.show()

        basicjitter = BasicJitter()
        freq, Y = basicjitter.get_freq_domain(freq_, nb_of_bar)

        plt.plot(freq, abs(Y), 'r-')
        plt.xlabel('Freq (Hz)')
        plt.ylabel('|Y(freq)|')
        plt.xlim([0, nb_of_bar/10])
        plt.title('Find the null frequency from this graph')
        plt.show()

        null_freq = float(input('Null Frequency: '))

        ## create convolution
        start_time = start
        end_time = end
        mean = 0.5

        std_dev = float(input('Gaussian Std Dev (obtained from gradient descent): '))

        Fs = nb_of_bar

        t, y_gaussian = basicjitter.gaussian(mean, std_dev, Fs, start_time, end_time)

        null_freq = 10

        width_sine = 0.765 / null_freq

        width_tophat = float(input('Top Hat Width: '))

        t, y_sine = basicjitter.sine(mean, width_sine, Fs, start_time, end_time)
        t, y_hat = basicjitter.tophat(mean - width_tophat/2, mean + width_tophat/2, Fs, start_time, end_time)
        
        y_conv_ident = np.convolve(y_sine, y_gaussian, 'same')
        y_conv_ident = np.convolve(y_conv_ident, y_hat, 'same')

        title_gauss = 'Gauss, mean = {}, std_dev = {}'.format(mean, std_dev)
        title_tophat = 'Tophat, mean = {}, width = {}'.format(mean, width_tophat)
        title_sine = 'Sine, mean = {}, null_freq = {}'.format(mean, null_freq)

        title = 'Convoluted Jitter, Fs: {}\n{}\n{}\n{}'.format(Fs, title_gauss, title_tophat, title_sine)

        fig, axs = plt.subplots(nrows=3, figsize=(10.0, 7.5))
        plt.subplots_adjust(hspace=0.3)

        axs[0].plot(t,y_conv_ident*100,'r-')
        axs[0].bar(edge_[:-1], freq_*100, width = 1/nb_of_bar, edgecolor = 'k', alpha = 0.8)
        axs[0].set_ylabel('Probability Density [%]')

        y_conv_cum = get_cum(y_conv_ident[:-1])
        y_data_cum = get_cum(freq_)

        diff = abs(y_conv_cum - y_data_cum)*100

        axs[1].plot(t[:-1],y_conv_cum*100,'r-')
        axs[1].plot(t[:-1],y_data_cum*100,'k-')
        axs[1].set_ylabel('Cumulative Density [%]')
        axs[1].set_xlabel('Time [s]')

        axs[2].plot(t[:-1],diff,'r-')
        axs[2].set_ylabel('Difference [%]')
        axs[2].set_xlabel('Time [s]')

        plt.show()
    
if __name__ == '__main__':
    main()