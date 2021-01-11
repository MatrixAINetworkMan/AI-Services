import os
import time
import math
from os.path import join

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection  import train_test_split
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
        arr.append(stft_spectrogram[i*stride: i*stride + split_length])
    arr.append(stft_spectrogram[-split_length:])
    arr = np.array(arr)
    # shape = (split_num, 32, 1024)

    return arr, split_num


def load_data_by_type(machine_type):
    # load both dev_data and eval_data
    files = listfile(join(Config.dev_directory, machine_type, "train"))
    files = files + listfile(join(Config.eval_directory, machine_type, "train"))

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

    if not 0 in label: label = label - 1
    label_num = len(np.unique(label))

    train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=0.1)

    train_dataloader = DataLoader(
        dataset=TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_label)),
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=0
    )
    test_dataloader = DataLoader(
        dataset=TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_label)),
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=0
    )
    return train_dataloader, test_dataloader, label_num


def model_train(model, metric_fc, train_dataloader, test_dataloader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=Config.lr, momentum=0.9, nesterov=True)
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.T_max, eta_min=Config.eta_min)

    lr_strategy = Config.lr_strategy
    num_epochs = Config.num_epochs
    model_save_interval = Config.model_save_interval
    device = torch.device(Config.cuda if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    metric_fc = metric_fc.to(device)

    train_start = time.time()
    for epoch in range(num_epochs):
        epoch_start = time.time()
        info('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 50)

        running_loss = 0
        running_corrects = 0

        model.train()
        for data, label in train_dataloader:
            data = data.to(device)
            label = label.to(device)

            feature = model(data)
            output = metric_fc(feature, label)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.data.item() * data.size(0)
            preds = torch.argmax(output, 1)
            running_corrects += (preds == label.data).sum().float()

        if lr_strategy: scheduler.step()

        train_loss = running_loss / len(train_dataloader.dataset)
        train_acc = running_corrects.double() / len(train_dataloader.dataset)
        info('train Loss: {:.4f}'.format(train_loss))
        info('train acc :{:.4f}'.format(train_acc))
        print("train_time: {}".format(time.time() - epoch_start))

        running_loss = 0
        running_corrects = 0

        model.eval()
        with torch.no_grad():
            for data, label in test_dataloader:
                data = data.to(device)
                label = label.to(device)

                feature = model(data)
                output = metric_fc(feature, label)
                loss = criterion(output, label)

                running_loss += loss.data.item() * data.size(0)
                preds = torch.argmax(output, 1)
                running_corrects += (preds == label.data).sum().float()

        val_loss = running_loss / len(test_dataloader.dataset)
        val_acc = running_corrects.double() / len(test_dataloader.dataset)
        info('val Loss: {:.4f}'.format(val_loss))
        info('val acc : {:.4f}'.format(val_acc))
        print("epoch_time: {}".format(time.time() - epoch_start))

    time_elapsed = time.time() - train_start
    info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return model, metric_fc


def main():
    version_id = Config.version_id
    info("*************** {} train ***************".format(version_id))

    model_directory = join(Config.model_directory, version_id)
    os.makedirs(model_directory, exist_ok=True)

    types = os.listdir(Config.dev_directory)
    for idx, machine_type in enumerate(sorted(types)):
        print("\n===========================")
        info("[{}/{}] {}".format(idx+1, len(types), machine_type))

        model_file_path = "{}/model_{}.pkl".format(model_directory, machine_type)
        metric_file_path = "{}/metric_{}.pkl".format(model_directory, machine_type)
        if os.path.exists(model_file_path):
            info("model exists")
            continue

        print("============== GENERATE DATASET ==============")
        train_dataloader, test_dataloader, label_num = load_data_by_type(machine_type)
        info("num of ids : {}".format(label_num))

        info("============== MODEL TRAINING ==============")
        model = MobileFacenet()
        ArcMargin = ArcMarginProduct(in_features=128, out_features=label_num, s=30, m=0.05, easy_margin=False)

        model, ArcMargin = model_train(model, ArcMargin, train_dataloader, test_dataloader)

        torch.save(model, model_file_path)
        torch.save(ArcMargin, metric_file_path)

        print("save_model -> {}".format(model_file_path))
        info("============== END TRAINING ==============")

    return


if __name__ == "__main__":
    main()