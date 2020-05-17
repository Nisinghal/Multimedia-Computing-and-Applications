# -*- coding: utf-8 -*-
"""question1

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ReUcyAtNLGHiGD19TlQTWG1cDtsHajpf
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile
import librosa
import os ,pickle

from google.colab import drive
drive.mount('/content/drive')

base_path = '/content/drive/My Drive/assignment2/Dataset/training/'
test_path = '/content/drive/My Drive/assignment2/Dataset/validation/'
pkl_path = '/content/drive/My Drive/assignment2/pickle/spectrogram1/'
classes = ['zero','one','two','three','four','five','six','seven','eight','nine']

def plot(s):
  plt.pcolormesh(s)
  plt.ylabel('Frequency [Hz]')
  plt.xlabel('Time [ms]')
  plt.show()

def spectrogram(path, samplerate, windowlength, overlap):
  sound = librosa.load(path, sr = samplerate)
  beg  = np.arange(0, len(sound[0]), windowlength - overlap, dtype=int)
  beg  = beg[beg + windowlength < len(sound[0])]
  arr = []
  for b in beg:
    sound_window = []
    l = len(sound[0][b:b + windowlength])
    for n in range(int(l/2)): 
      sound_window.append(abs(sum(sound[0][b:b + windowlength] * np.exp((1j * 2 * np.pi *np.array(range(0,l,1)) * n)/l))/l) * 2)
    arr.append(sound_window)
  spectogram = np.array(arr).T
  plot(spectogram)
  return spectogram

def pickleitout(path, obj):
  with open(path, "ab") as f:
    pickle.dump(obj, f)
    f.close()

def main(path, xname, yname):
  # px = open(pkl_path+''+xname,"rb")
  # x= pickle.load(px)
  # px.close()
  # py = open(pkl_path+''+yname,"rb")
  # y = pickle.load(py)
  # py.close()

  x = []
  x = np.array(x)
  y = []
  count = 1
  for i in range(10):
    sound_path = path  + classes[i]+'/'
    for filename in os.listdir(sound_path):
      y.append(i)
      print("processing " + str(count))
      count += 1
      s = spectrogram(sound_path + filename, None, 256, 32)
      np.append(x,s)
    x = 10*np.log10(x)
      # break
    # break

    pickleitout(pkl_path+classes[i]+xname,x)
    pickleitout(pkl_path+classes[i]+yname,y)

  pickleitout(pkl_path+xname,x)
  pickleitout(pkl_path+yname,y)

main(base_path, 'spec.pkl', 'specy.pkl')
main(test_path, 'specval.pkl', 'specyval.pkl')