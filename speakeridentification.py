import numpy as np
import librosa
import librosa.display
import scipy.signal

audio = '.\\Desktop\\speech_test\\clean\\s2_bbbs5p.wav'
data, sampling_rate = librosa.load(audio, sr=4000)
spectral = 10 * np.log10(np.abs(librosa.core.stft(data, n_fft=512, hop_length=80, window=scipy.signal.hanning)))
w, h = spectral.shape
spectral = spectral.reshape(w * h)

#create loop to concatenate all the spectrals for designated speaker
