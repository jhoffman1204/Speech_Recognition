import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import scipy.signal

audio = '.\\Desktop\\speech_test\\clean\\s2_bbbs5p.wav'
data, sampling_rate = librosa.load(audio, sr=4000)
print(data.size)
plt.figure(figsize=(12, 4))
plt.plot(data)
plt.show()
print("y:", data, end='\n\n')
spectral = 10 * np.log10(np.abs(librosa.core.stft(data, n_fft=512, hop_length=80, window=scipy.signal.hanning)))
print(spectral.shape)
plt.plot(spectral[:,50])
plt.show()
print("Spectral Vector:")
print(spectral)


# Notes: 
# kevinsprojects.wordpress.com/2014/12/13
