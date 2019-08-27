import matplotlib.pyplot as plt
from scipy import signal as sig
from scipy import fftpack, stats
from pathlib import Path
import pandas as pd
import numpy as np


def main():
    plot_signals() 

    #CL_by_cycle() #Crack length increases linearly

    # average_amplitude() #Possible correlation
    # avg_peak_width() #General downward linear trend, spikes in last sample for T1 T3 T5 T6
    # max_peak_width() #Constant in large majority of cases, drops in T3 
    # dist_between_peaks() #Linear correlation for all except T4, constant in some cases


    # #FFT()

    # fft_amp_max() #No clear correlation
    fft_amp_sum() #Linear trend for all samples except T1 #near constant in most cases, slight upward trend
    # fft_amp_avg() #Possible correlation
    

    # #PSD()
    
    # psd_max_height()
    # psd_height_sum()
    

    # #Test sum/average amplitude of bins, especially to the left, with crack length
    # cross_correlation() #It looks like the amplitudes increase to the left as the crack lengthens

    correlation_coef()
    xc_bins()
    # xc_sub_signals()

    #convolve_signals()

    pass

def xc_sub_signals(show=True):
    p = plot()

    signals, names = plot_signals(show=False)
    crack_lengths = CL_by_cycle(show=False)

    for plate_signals, plate_names, cracks in zip(signals, names, crack_lengths):

        x = []
        y = []

        first = plate_signals.pop(0)
        #first = plate_signals[0]

        for signal, cycle_name, crack in zip(plate_signals, plate_names, cracks):

            # if crack[0,1] == 0:
            #     first = signal
            #     continue

            n = int(cycle_name[3:])

            r, _ = stats.pearsonr(first['ch2'], signal['ch2'])

            x.append(crack[0,1])
            y.append(r)

            #first = signal

        if show: p.plot_one_series(x, y, label = cycle_name[:2])

    if show:
        p.label_axes("Crack length (mm)", "Correlation coefficient")
        p.show_plot()
        p.close()

def convolve_signals(show=True):
    signals, name_list = plot_signals(show=False)

    conv_values = []

    for plate_signals, plate_names in zip(signals, name_list):

        temp = []

        for signal, name in zip(plate_signals, plate_names):

            conv = sig.convolve(signal['ch1'], signal['ch2'], mode='same')
           
            temp.append(conv)

            if show:
                p = plot()
                p.plot_one_series(signal['time'], conv, label=name, scale=True, new_subplot=True)

                p.label_axes("Time", "Voltage")
   
        conv_values.append(temp)

        if show: 
            p.show_plot()
            p.close()

def xc_bins(show=True):
    _, name_list = plot_signals(show=False)
    crack_lengths = CL_by_cycle(show=False)
    values = cross_correlation(show=False)
    

    plate_bin_data = []

    #p = plot()
    p2 = plot()

    for plate_data, plate_names, plate_cracks in zip(values, name_list, crack_lengths):

        x = []
        y = []
        y2 = []

        for corr, name, crack in zip(plate_data, plate_names, plate_cracks):

            max_v = np.max(corr)
            min_v = np.min(corr)

            bin_size = len(corr)//4

            temp = []
            
            #for i in range(bin_size):
            bin_data, corr = np.split(corr, [bin_size])
            #y.append((np.mean(np.abs(bin_data)) - min_v) / (max_v - min_v))
            #y.append(np.mean(bin_data))
            y2.append((np.max(np.abs(bin_data)) - min_v) / (max_v - min_v))

            #y.append(temp)

            x.append(crack[0, 1])


        if show:               
            #p.plot_one_series(x, y, label=name[:2], scale=False, new_subplot=False)
            p2.plot_one_series(x, y2, label=name[:2], scale=False, new_subplot=False)

            #p.label_axes("Crack Length (mm)", "Average amplitude of bin 1")
            p2.label_axes("Crack Length (mm)", "Max amplitude of bin 1")


    if show: 
        p2.show_plot()
        p2.close()

def correlation_coef(show=True):
    p = plot()

    signals, names = plot_signals(show=False)
    crack_lengths = CL_by_cycle(show=False)

    for plate_signals, plate_names, cracks in zip(signals, names, crack_lengths):

        x = []
        y = []

        for signal, cycle_name, crack in zip(plate_signals, plate_names, cracks):
            n = int(cycle_name[3:])

            r, _ = stats.pearsonr(signal['ch1'], signal['ch2'])

            x.append(crack[0,1])
            y.append(r)

        if show: p.plot_one_series(x, y, label = cycle_name[:2])

    if show:
        p.label_axes("Crack length (mm)", "Correlation coefficient")
        p.show_plot()
        p.close()

def cross_correlation(show=True):
    signals, name_list = plot_signals(show=False)

    xcorr_values = []

    for plate_signals, plate_names in zip(signals, name_list):

        temp = []

        for signal, name in zip(plate_signals, plate_names):

            corr = sig.correlate(signal['ch1'], signal['ch2'], mode='same')
           
            temp.append(corr)

            if show:
                p = plot()
                p.plot_one_series(signal['time'], corr, label=name, scale=True, new_subplot=True)

                p.label_axes("Time", "Cross-correlation function")
   
        xcorr_values.append(temp)

        if show: 
            p.show_plot()
            p.close()

    return xcorr_values

def psd_height_sum(show=True):
    p = plot()

    psd_values, _ = PSD(show=False)
    _, names = plot_signals(show=False)
    crack_lengths = CL_by_cycle(show=False)

    for plate_psd, plate_names, cracks in zip(psd_values, names, crack_lengths):

        x = []
        y = []

        for psd, cycle_name, crack in zip(plate_psd, plate_names, cracks):
            n = int(cycle_name[3:])

            peaks, _ = sig.find_peaks(psd)

            x.append(crack[0,1])
            y.append(np.sum(peaks))

        if show: p.plot_one_series(x, y, label = cycle_name[:2])

    if show:
        p.label_axes("Crack length (mm)", "PSD peak height sum")
        p.show_plot()
        p.close()

def psd_max_height(show=True):
    p = plot()

    psd_values, x = PSD(show=False)
    _, names = plot_signals(show=False)
    crack_lengths = CL_by_cycle(show=False)

    for plate_psd, plate_names, cracks in zip(psd_values, names, crack_lengths):

        x = []
        y = []

        for psd, cycle_name, crack in zip(plate_psd, plate_names, cracks):
            n = int(cycle_name[3:])

            x.append(crack[0,1])
            y.append(np.max(psd))

        if show: p.plot_one_series(x, y, label = cycle_name[:2])

    if show:
        p.label_axes("Crack length (mm)", "Max PSD peak height")
        p.show_plot()
        p.close()

def PSD(show=True):
    signals, name_list = plot_signals(show=False)

    psd_values = []

    for plate_signals, plate_names in zip(signals, name_list):

        temp = []

        for signal, name in zip(plate_signals, plate_names):

            f, psd = sig.welch(signal['ch2'])
           
            temp.append(psd)

            if show:
                p = plot()
                p.plot_one_series(f, psd, label=name, scale=True, new_subplot=True)

                p.label_axes("Frequency", "PSD values")
   
        psd_values.append(temp)

        if show: 
            p.show_plot()
            p.close()

    return psd_values, f

def avg_peak_width(show=True):    
    p = plot()

    signals, names = plot_signals(show=False)
    crack_lengths = CL_by_cycle(show=False)

    for plate_signals, plate_names, cracks in zip(signals, names, crack_lengths):

        x = []
        y = []

        for signal, cycle_name, crack in zip(plate_signals, plate_names, cracks):
            n = int(cycle_name[3:])

            peaks, _ = sig.find_peaks(signal['ch2'])
            widths = sig.peak_widths(signal['ch2'], peaks, rel_height=1.0)

            x.append(crack[0,1])
            y.append(np.mean(widths))

        if show: p.plot_one_series(x, y, label = cycle_name[:2])

    if show:
        p.label_axes("Crack length (mm)", "Average peak width")
        p.show_plot()
        p.close()

def max_peak_width(show=True):
    p = plot()

    signals, names = plot_signals(show=False)
    crack_lengths = CL_by_cycle(show=False)

    for plate_signals, plate_names, cracks in zip(signals, names, crack_lengths):

        x = []
        y = []

        for signal, cycle_name, crack in zip(plate_signals, plate_names, cracks):
            n = int(cycle_name[3:])

            peaks, _ = sig.find_peaks(signal['ch2'])
            widths = sig.peak_widths(signal['ch2'], peaks, rel_height=1.0)

            x.append(crack[0,1])
            y.append(np.max(widths))

        if show: p.plot_one_series(x, y, label = cycle_name[:2])

    if show:
        p.label_axes("Crack length (mm)", "Largest peak width")
        p.show_plot()
        p.close()

def dist_between_peaks(show=True):
    p = plot()

    signals, names = plot_signals(show=False)
    crack_lengths = CL_by_cycle(show=False)

    for plate_signals, plate_names, plate_cracks in zip(signals, names, crack_lengths):

        x = []
        y = []

        for signal, cycle_name, crack in zip(plate_signals, plate_names, plate_cracks):
            n = int(cycle_name[3:])

            signal = np.matrix(signal)

            ch1_max = np.argmax(signal[:,1])
            ch2_max = np.argmax(signal[:,2])

            if np.isnan(signal[ch1_max, 0]):
                delta_time = signal[1, 0] - signal[0, 0]
                signal[ch1_max, 0] = ch1_max * delta_time
            
            x.append(crack[0,1])
            y.append(signal[ch2_max, 0] - signal[ch1_max, 0])

        if show: p.plot_one_series(x, y, label = cycle_name[:2])

    if show:
        p.label_axes("Crack length (mm)", "Peak time difference")
        p.show_plot()
        p.close()
        
def fft_amp_max(show=True):
    p = plot()

    _, names = plot_signals(show=False)
    _, _, data = FFT(show=False)
    crack_lengths = CL_by_cycle(show=False)

    for plate, plate_names, cracks in zip(data, names, crack_lengths):

        x = []
        y = []

        for one_sum, cycle_name, crack in zip(plate, plate_names, cracks):
            n = int(cycle_name[3:])

            x.append(crack[0,1])
            y.append(one_sum)

        if show: p.plot_one_series(x, y, label = cycle_name[:2])

    if show:
        p.label_axes("Crack length (mm)", "Max amplitude")
        p.show_plot()
        p.close()

def fft_amp_sum(show=True):
    p = plot()

    _, names = plot_signals(show=False)
    _, data, _ = FFT(show=False)
    crack_lengths = CL_by_cycle(show=False)

    for plate, plate_names, cracks in zip(data, names, crack_lengths):

        x = []
        y = []

        for one_sum, cycle_name, crack in zip(plate, plate_names, cracks):
            n = int(cycle_name[3:])

            x.append(crack[0,1])
            y.append(one_sum)

        if show: p.plot_one_series(x, y, label = cycle_name[:2])

    if show:
        p.label_axes("Crack Length (mm)", "Amplitude sum")
        p.show_plot()
        p.close()

def fft_amp_avg(show=True):
    p = plot()

    av_waves, _, _ = FFT(show=False)
    crack_lengths = CL_by_cycle(show=False)

    plate2="T1"

    #cycles = []
    av_y2 = []

    for y2, plate, _ in av_waves:

        if plate != plate2:
            crack_data = crack_lengths.pop(0)

            cracks = np.squeeze(np.asarray(crack_data[:, 1]))

            if show:
                p.plot_one_series(cracks, av_y2, label=plate2)

            cycles = []
            av_y2 = []

        av_y2.append(y2)
        #cycles.append(cycle)
        plate2 = plate

    if show:
        p.label_axes("Crack length (mm)", "Average amplitude")
        p.show_plot()
        p.close()

def average_amplitude(show=True):
    p = plot()

    crack_lengths = CL_by_cycle(show=False)
    signals, names = plot_signals(show=False)

    for plate, plate_names, cracks in zip(signals, names, crack_lengths):

        x = []
        y = []

        for data, cycle_name, crack in zip(plate, plate_names, cracks):
            n = int(cycle_name[3:])
            data = np.matrix(data)

            max_y = sig.argrelextrema(data[:, 2], np.greater, order=5)
            av_wave_y = np.mean(abs(data[max_y, 2]))

            x.append(crack[0,1])
            y.append(av_wave_y)

        if show: p.plot_one_series(x, y, label = cycle_name[:2])

    if show:
        p.label_axes("Crack length (mm)", "Average amplitude")
        p.show_plot()
        p.close()

def FFT(show=True):
    signals, name_list = plot_signals(show=False)

    av_waves = []
    sums = []
    maxes = []

    for plate, name in zip(signals, name_list):

        temp = []
        temp2 =[]

        for s, n in zip(plate, name):
            half = len(s)//2

            x = s["time"]
            x = fftpack.fftfreq(len(s), x[1] - x[0])[0:half-1]
            #x = fftpack.fftshift(x)

            y1 = fftpack.fft(s["ch1"])[half+1:]
            y2 = fftpack.fft(s["ch2"])[half+1:]

            y1 = fftpack.fftshift(y1)
            y2 = fftpack.fftshift(y2)


            #Sum of amplitudes
            temp.append(np.sum(np.abs(fftpack.fft(s["ch2"])[1:])))

            #Max amplitude
            temp2.append(np.max(y2))

            #Calculates extremes for other functions, pretty inefficent
            maxy1 = sig.argrelextrema(y1, np.greater, order=5)
            maxy2 = sig.argrelextrema(np.abs(y2), np.greater, order=5)

            y1_ex = y1[maxy1]
            y2_ex = y2[maxy2]

            av_wave_y1 = np.mean(np.abs(y1_ex))
            av_wave_y2 = np.mean(np.abs(y2_ex))

            av_waves.append([av_wave_y2, n[:2], int(n[3:])])
            #

            if show:
                p = plot()
                p.plot_one_series(x, np.abs(y1), label= n + "-ch1", scale=True, new_subplot=True)
                p.plot_one_series(x, np.abs(y2), label= n + "-ch2")

                p.label_axes("Frequency", "Amplitude")

        sums.append(temp)
        maxes.append(temp2)
                
        if show: 
            p.show_plot()
            p.close()

    return av_waves, sums, maxes

def plot_signals(show=True):
    prod = process_data()
    data = []
    names = []

    for i in range(1, 7):
        n = []
        one_plate = []

        cycle_dirs = [str(f).replace("PHM_Data_Challenge_19/PHM2019_crack_growth_training_data/", '') 
                    for f in Path("PHM_Data_Challenge_19/PHM2019_crack_growth_training_data/T" + str(i)).iterdir()
                        if f.is_dir()]    

        cycle_dirs.sort(reverse=True)

        for path in cycle_dirs:                        

            d1 = prod.from_csv(path + "/signal_1.csv")
            d2 = prod.from_csv(path + "/signal_2.csv")
            d = prod.average_signals(d1, d2)

            path = path.replace("/", '-')
            
            n.append(path)
            one_plate.append(d)

            if show:
                p = plot()
                p.plot_one_series(d['time'], d['ch1'], label= path + "-ch1", scale=True, new_subplot=True)
                #p.plot_one_series(d['time'], d1['ch2'], label= path + "-ch2", scale=False)
                #p.plot_one_series(d['time'], d2['ch2'], label= path + "-ch2", scale=False)
                p.plot_one_series(d['time'], d['ch2'], label= path + "-ch2", scale=False)

                p.label_axes("Time", "Signal (V)")

        names.append(n)
        data.append(one_plate)

        if show:
            p.show_plot()
            p.close()

    return data, names

def CL_by_cycle(show=True):
    if show: p = plot()
    prod = process_data()

    data = []

    for i in range(1, 7):
        d = prod.from_excel("T" + str(i) + "/Description_T" + str(i) + ".xlsx")
        d = np.matrix(d)
        d = d[5:, 0:2]

        #filters nan values
        for i2 in range(len(d)):
            if pd.isnull(d[i2, 0]):
                d = d[:i2]
                break

        data.append(d)

        if show: p.plot_one_series(np.squeeze(np.asarray(d[:, 0])), np.squeeze(np.asarray(d[:, 1])), label="T" + str(i))

    if show: 
        p.label_axes("Cycle count", "Crack Length (mm)")
        p.show_plot()
        p.close()

    return data


class process_data():

    def __init__(self):
        pass

    def from_csv(self, filename):
        filename = 'PHM_Data_Challenge_19/PHM2019_crack_growth_training_data/' + filename
        return pd.read_csv(filename)

    def from_excel(self, filename):
        filename = 'PHM_Data_Challenge_19/PHM2019_crack_growth_training_data/' + filename
        return pd.read_excel(filename)

    def average_signals(self, df_1, df_2):
        mean_ch2 = (df_1['ch2'] + df_2['ch2']) / 2.0
        df_1['ch2'] = mean_ch2
        
        return df_1


class plot():

    def __init__(self, one_plot=True, num_plots=1):
        self.fig = plt.figure()
        self.subplot_info = [num_plots, 1, 1]
        if one_plot: self.ax = self.fig.add_subplot(1, 1, 1)

    def plot_one_series(self, x_data, y_data, label, scale=False, new_subplot=False):
        if new_subplot: self.new_subplot()
        if scale: plt.axis([min(x_data),
                            max(x_data),
                            min(y_data) + (min(y_data)/5),
                            max(y_data) + (max(y_data)/5)])

        self.ax.plot(x_data, y_data, label=label)

    def label_axes(self, x_label, y_label):
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(loc='upper left')

    def new_subplot(self):
        self.ax = self.fig.add_subplot(self.subplot_info[0], self.subplot_info[1], self.subplot_info[2])

        self.subplot_info[2] += 1

    def show_plot(self):       
        plt.show()
    
    def close(self):
        subplot_info = [1,1,1]
        plt.close()

main()
