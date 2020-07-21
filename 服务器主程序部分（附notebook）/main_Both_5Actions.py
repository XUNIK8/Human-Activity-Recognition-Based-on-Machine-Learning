import scipy as sp
import sys
from scipy import signal
from scipy import stats
from scipy.stats import entropy
from scipy.stats import iqr as IQR
import joblib
from detecta import detect_peaks

import numpy as np
from flask import Flask,request
app = Flask(__name__)

accx = []
accy = []
accz = []
gryx = []
gryy = []
gryz = []
feature = []


def mean(inArray):
    array=np.array(inArray)
    mean_value = float(array.mean())
    return mean_value

# std: standard deviation of mag column
def std(inArray):
    array=np.array(inArray)
    std_value = float(array.std()) # std value
    return std_value

# mad: median deviation
def mad(inArray):
    array=np.array(inArray)
    mad_value = float(sp.median(array))# median deviation value of mag_column
    return mad_value

# max
def max(inArray):
    array=np.array(inArray)
    max_value=float(array.max()) # max value
    return max_value
# min
def min(inArray):
    array=np.array(inArray)
    min_value= float(array.min()) # min value
    return min_value

# IQR
def IQR(inArray):
    array=np.array(inArray)
    IQR_value=float(IQR(array))# Q3(column)-Q1(column)
    return IQR_value

# Entropy
def entropy(inArray):
    array=np.array(inArray)
    entropy_value=float(entropy(array)) # entropy signal
    return entropy_value

def sma(inArray):
    array=np.array(inArray)
    sma_axial=float(abs(array).sum()) # sum of areas under each signal
    return sma_axial # return sma value

def energy(inArray):
    array=np.array(inArray)
    energy_vector=(array**2).sum() # energy value of each df column
    return energy_vector # return energy vector energy_X,energy_Y,energy_Z

def skew(inArray):
    array=np.array(inArray)
    skew_value=float(stats.skew(array)) # entropy signal
    return skew_value

def kurt(inArray):
    array=np.array(inArray)
    kurt_value=float(stats.kurtosis(array)) # entropy signal
    return kurt_value

@app.route('/count', methods=['post'])
def count():

    numA = 0
    numB = 0
    numC = 0
    numD = 0
    count = []
    data = request.get_json()
    system = data['system']
    accyAll = data['accyAll']
    activity = data['activity']
    activity.append(5)
    if len(accyAll) > 0:
        if system == 1:
            b1, a1 = signal.butter(2, [0.001, 0.03], 'bandpass')
            accyAll_After = signal.filtfilt(b1, a1, accyAll,padlen = 0)
            print('length:',len(accyAll_After))
            print('activity:',activity)
            print('len of activity:', len(activity))
            peaks = detect_peaks(accyAll_After)
            print('peaks:',peaks)
            print('len of peaks:', len(peaks))
            for i in range(len(peaks)):
                location = int(peaks[i] / 60)
                if location == 0:
                    location = 1
                type = activity[location]
                if type == 1:
                    numA += 1
                if type == 2:
                    numB += 1
                if type == 3:
                    numC += 1
                if type == 4:
                    numD += 1
            count = [numA,numB,numC,numD]
            countStr =  ','.join(str(i) for i in count)
        if system == 2:
            b1, a1 = signal.butter(2, [0.025, 0.09], 'bandpass')
            accyAll_After = signal.filtfilt(b1, a1, accyAll, padlen=0)
            print('length:', len(accyAll_After))
            print('activity:', activity)
            print('len of activity:', len(activity))
            peaks = detect_peaks(accyAll_After)
            print('peaks:', peaks)
            print('len of peaks:', len(peaks))
            for i in range(len(peaks)):
                location = int(peaks[i] / 20)
                if location == 0:
                    location = 1
                type = activity[location]
                if type == 1:
                    numA += 1
                if type == 2:
                    numB += 1
                if type == 3:
                    numC += 1
                if type == 4:
                    numD += 1
            count = [numA,numB,numC,numD]
            countStr =  ','.join(str(i) for i in count)

    print(count)
    print(countStr)
    return countStr

@app.route('/',methods = ['post'])
def action():
    global accx
    global accy
    global accz
    global gryx
    global gryy
    global gryz
    global feature

    data = request.get_json()

    accx1 = data['accx']
    accy1 = data['accy']
    accz1 = data['accz']
    gryx1 = data['gryx']
    gryy1 = data['gryy']
    gryz1 = data['gryz']

    system = data['system']
    print('system:',system)
    if system == 1:


        accx = np.concatenate((accx,accx1))
        accy = np.concatenate((accy,accy1))
        accz = np.concatenate((accz,accz1))
        gryx = np.concatenate((gryx,gryx1))
        gryy = np.concatenate((gryy,gryy1))
        gryz = np.concatenate((gryz,gryz1))

        if len(accx) >100:

            b, a = signal.butter(3, 0.1, 'lowpass')

            accx_After = signal.filtfilt(b, a, accx)
            mean_x1 = mean(accx_After[:])
            std_x1 = std(accx_After[:])
            mad_x1 = mad(accx_After[:])
            max_x1 = max(accx_After[:])
            min_x1 = min(accx_After[:])

            sma_x1 = sma(accx_After[:])
            energy_x1 = energy(accx_After[:])
            skew_x1 = skew(accx_After[:])
            kurt_x1 = kurt(accx_After[:])
        # plt.plot(num, accx[0:1000])
        # plt.title('accx:')
        # python要用show展现出来图
        # plt.show()

            accy_After = signal.filtfilt(b, a, accy)
            mean_y1 = mean(accy_After[:])
            std_y1 = std(accy_After[:])
            mad_y1 = mad(accy_After[:])
            max_y1 = max(accy_After[:])
            min_y1 = min(accy_After[:])

            sma_y1 = sma(accy_After[:])
            energy_y1 = energy(accy_After[:])
            skew_y1 = skew(accy_After[:])
            kurt_y1 = kurt(accy_After[:])

        #   plt.plot(num, accy[0:1000])
        #  plt.title('accy:')
        # python要用show展现出来图
        # plt.show()

            accz_After = signal.filtfilt(b, a, accz)
            mean_z1 = mean(accz_After[:])
            std_z1 = std(accz_After[:])
            mad_z1 = mad(accz_After[:])
            max_z1 = max(accz_After[:])
            min_z1 = min(accz_After[:])

            sma_z1 = sma(accz_After[:])
            energy_z1 = energy(accz_After[:])
            skew_z1 = skew(accz_After[:])
            kurt_z1 = kurt(accz_After[:])
            #  plt.plot(num, accz[0:1000])
            #  plt.title('accz:')
            # python要用show展现出来图
            # plt.show()

            gryx_After = signal.filtfilt(b, a, gryx)
            mean_x2 = mean(gryx_After[:])
            std_x2 = std(gryx_After[:])
            mad_x2 = mad(gryx_After[:])
            max_x2 = max(gryx_After[:])
            min_x2 = min(gryx_After[:])

            sma_x2 = sma(gryx_After[:])
            energy_x2 = energy(gryx_After[:])
            skew_x2 = skew(gryx_After[:])
            kurt_x2 = kurt(gryx_After[:])
            # plt.plot(num, gryx[0:1000])
            # plt.title('gryx:')
            # python要用show展现出来图
            # plt.show()

            gryy_After = signal.filtfilt(b, a, gryy)
            mean_y2 = mean(gryy_After[:])
            std_y2 = std(gryy_After[:])
            mad_y2 = mad(gryy_After[:])
            max_y2 = max(gryy_After[:])
            min_y2 = min(gryy_After[:])

            sma_y2 = sma(gryy_After[:])
            energy_y2 = energy(gryy_After[:])
            skew_y2 = skew(gryy_After[:])
            kurt_y2 = kurt(gryy_After[:])
            # plt.plot(num, gryy[0:1000])
            # plt.title('gryy:',)
            # python要用show展现出来图
            # plt.show()

            gryz_After = signal.filtfilt(b, a, gryz)
            mean_z2 = mean(gryz_After[:])
            std_z2 = std(gryz_After[:])
            mad_z2 = mad(gryz_After[:])
            max_z2 = max(gryz_After[:])
            min_z2 = min(gryz_After[:])

            sma_z2 = sma(gryz_After[:])
            energy_z2 = energy(gryz_After[:])
            skew_z2 = skew(gryz_After[:])
            kurt_z2 = kurt(gryz_After[:])
            # plt.plot(num, gryz[0:1000])
            # plt.title('gryz:')
            # python要用show展现出来图
            # plt.show()

            feature = [mean_x1, std_x1, mad_x1, max_x1, min_x1,
                            sma_x1, energy_x1, skew_x1, kurt_x1,
                            mean_y1, std_y1, mad_y1, max_y1, min_y1,
                            sma_y1, energy_y1, skew_y1, kurt_y1,
                            mean_z1, std_z1, mad_z1, max_z1, min_z1,
                            sma_z1, energy_z1, skew_z1, kurt_z1,
                            mean_x2, std_x2, mad_x2, max_x2, min_x2,
                            sma_x2, energy_x2, skew_x2, kurt_x2,
                            mean_y2, std_y2, mad_y2, max_y2, min_y2,
                            sma_y2, energy_y2, skew_y2, kurt_y2,
                            mean_z2, std_z2, mad_z2, max_z2, min_z2,
                                                                                                                                                                           sma_z2, energy_z2, skew_z2, kurt_z2,
                            ]

            rf = joblib.load('model_RF_IOS_97.1_5Actions.model')
            feature = np.array(feature)
            feature = feature.reshape(1,-1)
            action = rf.predict(feature)
            if action[0] ==0:
                action[0] = 5
            print('action1:',action)
            action = str(action[0])
            print('action1:',action)
            feature = []
            accx = accx[60:]
            accy = accx[60:]
            accz = accx[60:]
            gryx = accx[60:]
            gryy = accx[60:]
            gryz = accx[60:]

        else :
            accx = accx[:]
            accy = accx[:]
            accz = accx[:]
            gryx = accx[:]
            gryy = accx[:]
            gryz = accx[:]
            action = ''

    if system == 2:
        accx = np.concatenate((accx, accx1))
        accy = np.concatenate((accy, accy1))
        accz = np.concatenate((accz, accz1))
        gryx = np.concatenate((gryx, gryx1))
        gryy = np.concatenate((gryy, gryy1))
        gryz = np.concatenate((gryz, gryz1))

        if len(accx) >30:

            b, a = signal.butter(3, 0.1, 'lowpass')

            accx_After = signal.filtfilt(b, a, accx)
            mean_x1 = mean(accx_After[:])
            std_x1 = std(accx_After[:])
            mad_x1 = mad(accx_After[:])
            max_x1 = max(accx_After[:])
            min_x1 = min(accx_After[:])

            sma_x1 = sma(accx_After[:])
            energy_x1 = energy(accx_After[:])
            skew_x1 = skew(accx_After[:])
            kurt_x1 = kurt(accx_After[:])
        # plt.plot(num, accx[0:1000])
        # plt.title('accx:')
        # python要用show展现出来图
        # plt.show()

            accy_After = signal.filtfilt(b, a, accy)
            mean_y1 = mean(accy_After[:])
            std_y1 = std(accy_After[:])
            mad_y1 = mad(accy_After[:])
            max_y1 = max(accy_After[:])
            min_y1 = min(accy_After[:])

            sma_y1 = sma(accy_After[:])
            energy_y1 = energy(accy_After[:])
            skew_y1 = skew(accy_After[:])
            kurt_y1 = kurt(accy_After[:])

        #   plt.plot(num, accy[0:1000])
        #  plt.title('accy:')
        # python要用show展现出来图
        # plt.show()

            accz_After = signal.filtfilt(b, a, accz)
            mean_z1 = mean(accz_After[:])
            std_z1 = std(accz_After[:])
            mad_z1 = mad(accz_After[:])
            max_z1 = max(accz_After[:])
            min_z1 = min(accz_After[:])

            sma_z1 = sma(accz_After[:])
            energy_z1 = energy(accz_After[:])
            skew_z1 = skew(accz_After[:])
            kurt_z1 = kurt(accz_After[:])
            #  plt.plot(num, accz[0:1000])
            #  plt.title('accz:')
            # python要用show展现出来图
            # plt.show()

            gryx_After = signal.filtfilt(b, a, gryx)
            mean_x2 = mean(gryx_After[:])
            std_x2 = std(gryx_After[:])
            mad_x2 = mad(gryx_After[:])
            max_x2 = max(gryx_After[:])
            min_x2 = min(gryx_After[:])

            sma_x2 = sma(gryx_After[:])
            energy_x2 = energy(gryx_After[:])
            skew_x2 = skew(gryx_After[:])
            kurt_x2 = kurt(gryx_After[:])
            # plt.plot(num, gryx[0:1000])
            # plt.title('gryx:')
            # python要用show展现出来图
            # plt.show()

            gryy_After = signal.filtfilt(b, a, gryy)
            mean_y2 = mean(gryy_After[:])
            std_y2 = std(gryy_After[:])
            mad_y2 = mad(gryy_After[:])
            max_y2 = max(gryy_After[:])
            min_y2 = min(gryy_After[:])

            sma_y2 = sma(gryy_After[:])
            energy_y2 = energy(gryy_After[:])
            skew_y2 = skew(gryy_After[:])
            kurt_y2 = kurt(gryy_After[:])
            # plt.plot(num, gryy[0:1000])
            # plt.title('gryy:',)
            # python要用show展现出来图
            # plt.show()

            gryz_After = signal.filtfilt(b, a, gryz)
            mean_z2 = mean(gryz_After[:])
            std_z2 = std(gryz_After[:])
            mad_z2 = mad(gryz_After[:])
            max_z2 = max(gryz_After[:])
            min_z2 = min(gryz_After[:])

            sma_z2 = sma(gryz_After[:])
            energy_z2 = energy(gryz_After[:])
            skew_z2 = skew(gryz_After[:])
            kurt_z2 = kurt(gryz_After[:])
            # plt.plot(num, gryz[0:1000])
            # plt.title('gryz:')
            # python要用show展现出来图
            # plt.show()

            feature = [mean_x1, std_x1, mad_x1, max_x1, min_x1,
                       sma_x1, energy_x1, skew_x1, kurt_x1,
                       mean_y1, std_y1, mad_y1, max_y1, min_y1,
                       sma_y1, energy_y1, skew_y1, kurt_y1,
                       mean_z1, std_z1, mad_z1, max_z1, min_z1,
                       sma_z1, energy_z1, skew_z1, kurt_z1,
                       mean_x2, std_x2, mad_x2, max_x2, min_x2,
                       sma_x2, energy_x2, skew_x2, kurt_x2,
                       mean_y2, std_y2, mad_y2, max_y2, min_y2,
                       sma_y2, energy_y2, skew_y2, kurt_y2,
                       mean_z2, std_z2, mad_z2, max_z2, min_z2,
                       sma_z2, energy_z2, skew_z2, kurt_z2,
                       ]

            rf = joblib.load('model_RF_Android_97.0_5Actions.model')
            feature = np.array(feature)
            feature = feature.reshape(1, -1)
            action = rf.predict(feature)
            if action[0] ==0:
                action[0] = 5
            print('action1:', action)
            action = str(action[0])
            print('action1:', action)
            feature = []
            accx = accx[20:]
            accy = accx[20:]
            accz = accx[20:]
            gryx = accx[20:]
            gryy = accx[20:]
            gryz = accx[20:]

        else:
            accx = accx[:]
            accy = accx[:]
            accz = accx[:]
            gryx = accx[:]
            gryy = accx[:]
            gryz = accx[:]
            action = ''

    return action


if __name__ == '__main__':
    context = (sys.path[0] + '/Nginx/1_www.inifyy.cn_bundle.crt', sys.path[0] + '/Nginx/2_www.inifyy.cn.key')
    app.run(debug=1, host='172.17.0.3', port=8000, ssl_context=context)