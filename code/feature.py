import os
import time
import math
from os.path import join

import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import common as com
from common import listfile
from config import Config
from model import MobileFacenet, ArcMarginProduct

info = com.logger.info


def file_to_stft_array(file_name, n_fft=2046, hop_length=512, split_length=32, stride=16):
    try:
        y, sr = librosa.load(file_name, sr=None, mono=False)
    except:
        com.logger.error("file_broken or not exists!! : {}".format(file_name))

    stft_spectrogram = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)).T
    # shape = (313, 1024) or (344, 1024) for TorCar

    split_num = math.ceil(stft_spectrogram.shape[0] / stride) - 1

    arr = []
    for i in range(split_num-1):
        arr.append(stft_spectrogram[i*stride:i*stride + split_length])
    arr.append(stft_spectrogram[-split_length:])
    arr = np.array(arr)
    # shape = (split_num, 32, 1024)
    
    return arr, split_num


def load_data_list(files):
    data = []
    label = []
    for fname in tqdm(files):
    # for fname in files:
        d, split_num = file_to_stft_array(fname)
        data.append(d)

        label.extend([int(os.path.split(fname)[1].split('_')[2])] * split_num)

    label = (np.array(label)).astype(np.int64)

    data = np.array(data)
    data = data.reshape((-1, data.shape[2], data.shape[3]))
    data = data.astype(np.float32)
    data = np.expand_dims(data, axis=1)

    dataloader = DataLoader(
        dataset=TensorDataset(torch.from_numpy(data), torch.from_numpy(label)),
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=0
    )
    return dataloader


id_dict = {
    'fan':          ['00', '01', '02', '03', '04', '05', '06'], 
    'pump':         ['00', '01', '02', '03', '04', '05', '06'], 
    'slider':       ['00', '01', '02', '03', '04', '05', '06'], 
    'valve':        ['00', '01', '02', '03', '04', '05', '06'], 
    'ToyCar':       ['01', '02', '03', '04', '05', '06', '07'], 
    'ToyConveyor':  ['01', '02', '03', '04', '05', '06'], }


def main():
    version_id = Config.version_id
    info("*************** {} feature ***************".format(version_id))

    model_directory = join(Config.model_directory, version_id)
    feature_directory = join(Config.feature_directory, version_id)
    os.makedirs(feature_directory, exist_ok=True)

    device = torch.device(Config.cuda if torch.cuda.is_available() else "cpu")

    types = os.listdir(Config.dev_directory)
    # types = ["ToyCar"]
    for idx, machine_type in enumerate(sorted(types)):
        print("\n===========================")
        info("[{}/{}] {}".format(idx+1, len(types), machine_type))

        model_file_path = "{}/model_{}.pkl".format(model_directory, machine_type)
        if not os.path.exists(model_file_path):
            info("model not exists")
            continue
        model = torch.load(model_file_path).to(device)
        model.eval()

        all_files = listfile(join(Config.dev_directory, machine_type, "train"))
        all_files += listfile(join(Config.eval_directory, machine_type, "train"))

        for machine_id in id_dict[machine_type]:
            info("machine_id : {}".format(machine_id))
            files = []
            for file in all_files:
                if file.split('_')[-2] == machine_id:
                    files.append(file)
            
            dataloader = load_data_list(files)

            features = []
            for data, _ in dataloader:
                data = data.to(device)
                feature = model(data)
                feature = feature.cpu().detach().numpy()
                features.extend(list(feature))
            features = np.array(features)
            info("feature shape : {}".format(features.shape))

            np.savetxt(join(feature_directory, "feature_{}_{}.csv".format(machine_type, machine_id)), features)

    return


if __name__ == "__main__":
    main()