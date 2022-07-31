import mindspore
import joblib
import librosa
import re
from mindspore import Tensor
import os
import numpy as np
import mindspore.dataset as ds
import mindspore.dataset.transforms as transforms


class Generator(object):
    def __init__(self, sr,
                 n_fft=1024,
                 n_mels=128,
                 win_length=1024,
                 hop_length=512,
                 power=2.0
                 ):
        self.sr=sr
        self.n_fft=n_fft
        self.n_mels=n_mels
        self.win_length=win_length
        self.hop_length=hop_length
        self.power=power
       #sr=np.ndarray(sr)
       # # sr=sr.astype('numpy.ndarray')
       #  n_fft =np.ndarray(n_fft)
       #  n_mels = np.ndarray(n_mels)
       #  win_length = np.ndarray(win_length)
       #  hop_length = np.ndarray(hop_length)
       #  power = np.ndarray(int(power))
      #  x = np.empty([sr,win_length,hop_length, 2], dtype=int)
        # self.mel_transform = transforms.MelSpectrogram
        #self.mel_transform=self.mel_transform.astype('numpy.ndarray')
        # self.mel_transform = librosa.feature.melspectrogram(sample_rate=sr,
        #                              win_length=win_length,
        #                              hop_length=hop_length,
        #                              n_fft=n_fft,
        #                             n_mels=n_mels,
        #                             power=power)

        #self.mel_transform = mindspore.from_numpy(self.mel_transform )

        #self.amplitude_to_db = mindspore.dataset.audio.transforms.AmplitudeToDB(stype='power')
        #self.amplitude_to_db =librosa.amplitude_to_db(stype='power')
    def __call__(self, x):
        # spec =  self.amplitude_to_db(self.mel_transform(x)).squeeze().transpose(-1,-2)
#        x = x.asnumpy()
       # x=np.array(x)
        librosa.feature.melspectrogram(y=x,
                                        sr=self.sr,
                                       win_length=self.win_length,
                                       hop_length=self.hop_length,
                                       n_fft=self.n_fft,
                                       n_mels=self.n_mels,
                                       power=self.power)
       # x=x.asnumpy()
        return librosa.amplitude_to_db(librosa.feature.melspectrogram(x))


class Wav_Mel_ID_Dataset:
    def __init__(self, root_floder, ID_factor, sr,
                 win_length, hop_length, transform=None):
       # with open(root_floder, 'rb') as f:
        with open(root_floder, 'rb') as f:

           self.file_path_list = joblib.load(f)
         #  print(self.file_path_list)
        self.transform = transform
        self.factor = ID_factor
        self.sr = sr
        self.win_len = win_length
        self.hop_len = hop_length
        # print(len(self.file_path_list))

    def getitem(self):
        file_path = self.file_path_list
        label = []
        x_wav = None
        x_mel = None
        for i, file_path in enumerate(self.file_path_list):
            file_path= file_path.replace('\\','/')
            print(i,file_path)
            if i==0:
                machine = file_path.split('/')[-3]
                id_str = re.findall('id_[0-9][0-9]', file_path)
                if machine == 'ToyCar' or machine == 'ToyConveyor':
                    id = int(id_str[0][-1]) - 1
                else:
                    id = int(id_str[0][-1])
                label = [int(self.factor[machine] * 7 + id)]

                (x, _) = librosa.core.load(file_path, sr=self.sr, mono=True)

                x = x[:self.sr * 10]  # (1, audio_length)
                x_wav =x[np.newaxis,:]
                x_mel = self.transform(x_wav)[np.newaxis,np.newaxis,:]
               # x_mel = np.expand_dims(np.expand_dims(self.transform(x_wav),axis=0)
            else:

                machine = file_path.split('/')[-3]
                id_str = re.findall('id_[0-9][0-9]', file_path)
                if machine == 'ToyCar' or machine == 'ToyConveyor':
                    id = int(id_str[0][-1]) - 1
                else:
                    id = int(id_str[0][-1])
                label1 = int(self.factor[machine] * 7 + id)
                label.append(label1)
                (x, _) = librosa.core.load(file_path, sr=self.sr, mono=True)

                x = x[:self.sr * 10]  # (1, audio_length)
                x_wav1 = x[np.newaxis,:]
                x_mel1= self.transform(x_wav1)[np.newaxis,np.newaxis,:]
                x_wav=np.concatenate([x_wav,x_wav1],axis=0)
                x_mel=np.concatenate([x_mel,x_mel1],axis=0)
        # print(x.shape)

            return x_wav, x_mel, label

    def __len__(self):
        return len(self.file_path_list)


class WavMelClassifierDataset:
    def __init__(self, root_folder, sr, ID_factor):
        self.root_folder = root_folder
        self.sr = sr
        self.factor = ID_factor

    def get_dataset(self,
                    n_fft=1024,
                    n_mels=128,
                    win_length=1024,
                    hop_length=512,
                    power=2.0):
        dataset = Wav_Mel_ID_Dataset(self.root_folder,
                                     self.factor,
                                     self.sr,
                                     win_length,
                                     hop_length,
                                     transform=Generator(
                                         self.sr,
                                         n_fft=n_fft,
                                         n_mels=n_mels,
                                         win_length=win_length,
                                         hop_length=hop_length,
                                         power=power,
                                     ))
        dataset1=dataset.getitem()


        return dataset1

if __name__ == '__main__':
    pass
