# coding: utf-8

# # 利用thchs30为例建立一个语音识别系统
#
#
# - 数据处理
# - 搭建模型
#     - DFCNN
#






# In[ ]:


#!tar zxvf data_thchs30.tgz


# ## 1. 特征提取
#
# input为输入音频数据，需要转化为频谱图数据，然后通过cnn处理图片的能力进行识别。
#
# **1. 读取音频文件**  MP3文件读取需要安装FFMPEG

# In[5]:
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import os

import librosa
import librosa.display

import librosa
import librosa.display
import matplotlib.pyplot as plt
# y, sr = librosa.load('./1A -  128 -  Stoppin (Wyvern bounce Edit)(djoffice.cn).mp3', sr=None)
# y, sr = librosa.load('./4A -  129 -  IGNITION X HIPSTA (LOCKETT MASHUP)(djoffice.cn).wav', sr=None , mono = True)#ok
y, sr = librosa.load('../../../dataset/dj/DeathNov Feat. Mc LuJian - Yo Yo Yeah Yeah (Original Mix).mp3', sr=None )#/data3/code/github_code/yolov3-to-dj/dataset/dj
# y, sr = librosa.load('./test.wav', sr=None)
# plt.plot(y);
# plt.title('Signal');
# plt.xlabel('Time (samples)');#采样点数，除以采样率才是时间
# plt.ylabel('Amplitude');
# plt.show()

import numpy as np
n_fft = 2048
ft = np.abs(librosa.stft(y[:n_fft], hop_length=n_fft + 1))
plt.plot(ft);
plt.title('Spectrum');
plt.xlabel('Frequency Bin');
plt.ylabel('Amplitude');
plt.show()

spec = np.abs(librosa.stft(y, hop_length=512))
spec = librosa.amplitude_to_db(spec, ref=np.max)
librosa.display.specshow(spec, sr=sr, x_axis='time', y_axis='log');
plt.colorbar(format='%+2.0f dB');
plt.title('Spectrogram');
plt.show()

# Load a wav file
# y, sr = librosa.load('./11A -  128 -  L-Dis - Bumaya (DB2N & WOOK2 Edit)(djoffice.cn).wav', sr=None)
# # extract mel spectrogram feature
# melspec = librosa.feature.melspectrogram(y, sr, n_fft=1024, hop_length=512, n_mels=128)
# # convert to log scale
# logmelspec = librosa.power_to_db(melspec)
# # plot mel spectrogram
# plt.figure()
# librosa.display.specshow(logmelspec, sr=sr, x_axis='time', y_axis='mel')
# plt.title('Beat wavform')
# plt.show()


# #第二种方式
mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512)
mel_spect = librosa.power_to_db(mel_spect, ref=np.max)#float32
np.save( "4A -  129 -  IGNITION X HIPSTA (LOCKETT MASHUP)(djoffice.cn).npy" , mel_spect*(-100000))
librosa.display.specshow(mel_spect, y_axis='mel', fmax=8000, x_axis='time');
plt.figure()
plt.title('Mel Spectrogram');
plt.colorbar(format='%+2.0f dB');
plt.show()


