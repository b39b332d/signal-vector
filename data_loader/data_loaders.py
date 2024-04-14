from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os,torch
import matplotlib.pyplot as plt
class SignalDataset(Dataset):
    def __init__(self, data_dir):
        label_files = os.listdir(data_dir+"gth/")
        train_files = os.listdir(data_dir+"train/")

        label_files_head = [ f.split(".")[0] for f in label_files ]
        train_files_head = [ f.split(".")[0] for f in train_files ]

        self.datas = [x for x in label_files_head if x in train_files_head]

        self.data_path = data_dir

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        train_path = self.data_path +"train/"+self.datas[idx]+".npy"
        label_path = self.data_path +"gth/"+self.datas[idx]+".npy"
        #ref_path = self.data_path +"label/"+self.datas[idx]+".npy"
        train_data = np.load(train_path) 
        # train_data=np.fft.fft(train_data,axis=1)
        # train_data = np.vstack((train_data.real,train_data.imag))
        label_data = np.load(label_path)
        #train_data = (train_data.T- np.mean(train_data,axis=1))/np.std(train_data,axis=1)
        #label_data = label_data/np.sqrt(np.sum(label_data**2))
        #label_data= np.zeros_like(label_data)
        
        # label_data = torch.from_numpy(label_data@train_data).float().cuda()
        label_data = torch.from_numpy(label_data).float().cuda()
        train_data = torch.from_numpy(train_data).float().cuda()

        return train_data, label_data
    def get_preview(self, idx):
        train_path = self.data_path +"train/"+self.datas[idx]+".npy"
        label_path = self.data_path +"gth/"+self.datas[idx]+".npy"
        ref_path = self.data_path +"label/"+self.datas[idx]+".npy"
        train_data = np.load(train_path) 
        # train_data=np.fft.fft(train_data,axis=1)
        # train_data = np.vstack((train_data.real,train_data.imag))
        label_data = np.load(label_path)
        #train_data = (train_data.T- np.mean(train_data,axis=1))/np.std(train_data,axis=1)
        #label_data = label_data/np.max(np.abs(label_data))
        ref_data = np.load(ref_path)

        return train_data, label_data,ref_data


class  SignalDataLoader(BaseDataLoader):
    def __init__(self, data_path, batch_size, shuffle=True, validation_split=0.0, num_workers=1):
        self.dataset = SignalDataset(data_path)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
    def preview(self,model):
        sample_idx = torch.randint(0,len(self.dataset), size=(1,)).item()
        data, label,ref = self.dataset.get_preview(sample_idx)
        plt.close()
        out = model(torch.from_numpy(data[None,:,:]).float().cuda()).detach().cpu().numpy()[0,0]
        out -= np.min(out)
        out /= np.max(out)
        plt.plot(out)
        # plt.plot(model(torch.from_numpy(data[None,:,:]).float().cuda()).detach().cpu()[0]@data)
        plt.plot(label.T)
        plt.plot((ref@data)[0])
        plt.pause(0.01)


