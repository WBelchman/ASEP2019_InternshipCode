from scipy import fftpack, signal, stats
import numpy as np

def concatenate_data(set1, set2):
    out = []
    
    for sub1, sub2 in zip(set1, set2):
        temp = []

        for d1, d2 in zip(sub1, sub2):

            if isinstance(d1, np.float64):
                temp.append([d1, d2])

            else:
                d1.append(d2)
                temp.append(d1)

        out.append(temp)
        
    return out

def fft_amp_sums(data):
    sums = []

    for plate in data:
        temp = []

        for s in plate:
            temp.append(np.sum(np.abs(fftpack.fft(s[:, 2])[1:])))

        sums.append(temp)

    return sums

def fft_amp_max(data):

    max_list = []

    for plate in data:

        temp = []

        for s in plate:
            temp.append(np.max(fftpack.fft(s[:, 2])[1:]).real)
        
        max_list.append(temp)

    return max_list

def psd_height_sum(data):
    sums = []

    for plate in data:

        temp = []

        for s in plate:

            _, psd = signal.welch(s[:, 2])

            temp.append(np.sum(psd))

        sums.append(temp)

    return sums

def psd_max_height(data):
    heights = []

    for plate in data:

        temp = []

        for s in plate:

            _, psd = signal.welch(s[:, 2])
            print(psd)
            peaks, _ = signal.find_peaks(psd.reshape(1, -1)[0])

            temp.append(np.max(peaks))

        heights.append(temp)

    return heights

def avg_peak_width(data):

    widths_list = []

    for plate in data:

        temp = []

        for s in plate:

            s = np.array(s[:, 2].flatten())          

            peaks, _ = signal.find_peaks(s[0])
            widths = signal.peak_widths(s[0], peaks, rel_height=1.0)

            temp.append(np.mean(widths))

        widths_list.append(temp)

    return widths_list

def correlation_coef(data):
    coef_list = []

    for plate_data in data:

        temp = []

        for d in plate_data:

            r, _ = stats.pearsonr(d[:, 1], d[:, 2])

            temp.append(r[0])

        coef_list.append(temp)

    return coef_list

def xc_sub_signals(data):

    coef_list = []

    for plate_signals in data:

        temp = []

        first = plate_signals.pop(0)

        for signal in plate_signals:

            r, _ = stats.pearsonr(first[:, 2], signal[:, 2])

            temp.append(r)

        coef_list.append(temp)

    return coef_list

def xc_mean_bin1(data):
    
    plate_bin_data = []

    for plate_data in data:

        temp = []

        for s in plate_data:

            corr = signal.correlate(s[:, 1], s[:, 2], mode='same')

            max_v = np.max(corr)
            min_v = np.min(corr)


            bin_size = len(corr)//4

            #for i in range(bin_size):
            bin_data, corr = np.split(corr, [bin_size])
            temp.append(np.max((np.abs(bin_data)) - min_v) / (max_v - min_v))

        plate_bin_data.append(temp)

    return plate_bin_data
