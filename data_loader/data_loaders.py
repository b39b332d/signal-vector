from torchvision import datasets, transforms
from base import BaseDataLoader
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os,torch
import matplotlib.pyplot as plt
class SignalDataset(Dataset):
    def __init__(self, data_dir):
        self.datas = []
        self.datasets_ofs = [0]
        self.data_path = os.path.join(data_dir,"")
        for datasets in os.listdir(data_dir):
            train_files = os.listdir(data_dir + datasets + "/train/")
            self.datas += [ os.path.join(datasets,"{}",f) for f in train_files ]


    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        train_path = self.data_path+self.datas[idx].format("train")
        label_path = self.data_path+self.datas[idx].format("label_gth")
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
        train_path = self.data_path+self.datas[idx].format("train")
        label_path = self.data_path+self.datas[idx].format("label_gth")
        ref_path   = self.data_path+self.datas[idx].format("label_fit")
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
        def norm_01(sig):
            sig-=np.min(sig,axis=0)
            sig/= np.max(sig,axis=0)
            return sig
        # sample_idx = torch.randint(0,len(self.dataset), size=(1,)).item()
        # data,label,ref,cmp = self.dataset.get_preview(sample_idx)
        # plt.close()
        # out = model(torch.from_numpy(data[None,:,:]).float().cuda()).detach().cpu().numpy()[0,0]
        # plt.plot(norm_01(out),label="inf")
        # # plt.plot(model(torch.from_numpy(data[None,:,:]).float().cuda()).detach().cpu()[0]@data)
        # plt.plot(norm_01(label.T),label="ecg")
        # plt.plot(norm_01(cmp.T),label="old")
        # plt.plot(norm_01((ref@data)[0]),label="LS")
        # plt.legend()
        # plt.pause(0.01)


