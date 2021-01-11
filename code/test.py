import os
import glob
import csv
import re
import itertools
import math
from os.path import join

import librosa
import numpy as np
import torch
from sklearn import metrics
from tqdm import tqdm

import common as com
from common import listfile
from config import Config

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


def save_csv(save_file_path, save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)


def get_machine_id_list_for_test(target_dir, dir_name="test", ext="wav"):
    # create test files
    dir_path = os.path.abspath("{dir}/{dir_name}/*.{ext}".format(
        dir=target_dir, dir_name=dir_name, ext=ext))
    file_paths = sorted(glob.glob(dir_path))
    # extract id
    machine_id_list = sorted(list(set(itertools.chain.from_iterable(
        [re.findall('id_[0-9][0-9]', ext_id) for ext_id in file_paths]))))
    return machine_id_list


def test_file_list_generator(
    target_dir, id_name, dir_name="test", prefix_normal="normal", 
    prefix_anomaly="anomaly", ext="wav", mode=True):
    info("target_dir : {}".format(target_dir+"_"+id_name))

    # development
    if mode:
        normal_files = sorted(
            glob.glob("{dir}/{dir_name}/{prefix_normal}_{id_name}*.{ext}".format(dir=target_dir,
                                                                                 dir_name=dir_name,
                                                                                 prefix_normal=prefix_normal,
                                                                                 id_name=id_name,
                                                                                 ext=ext)))
        normal_labels = np.zeros(len(normal_files))
        anomaly_files = sorted(
            glob.glob("{dir}/{dir_name}/{prefix_anomaly}_{id_name}*.{ext}".format(dir=target_dir,
                                                                                  dir_name=dir_name,
                                                                                  prefix_anomaly=prefix_anomaly,
                                                                                  id_name=id_name,
                                                                                  ext=ext)))
        anomaly_labels = np.ones(len(anomaly_files))
        files = np.concatenate((normal_files, anomaly_files), axis=0)
        labels = np.concatenate((normal_labels, anomaly_labels), axis=0)
        info("test_file  num : {num}".format(num=len(files)))
        if len(files) == 0:
            com.logger.exception("no_wav_file!!")

    # evaluation
    else:
        files = sorted(
            glob.glob("{dir}/{dir_name}/*{id_name}*.{ext}".format(dir=target_dir,
                                                                  dir_name=dir_name,
                                                                  id_name=id_name,
                                                                  ext=ext)))
        labels = None
        info("test_file  num : {num}".format(num=len(files)))
        if len(files) == 0:
            com.logger.exception("no_wav_file!!")

    return files, labels


def main():
    version_id = Config.version_id
    print("******************** {} test ********************".format(version_id))

    mode = Config.mode
    model_directory = join(Config.model_directory, version_id)
    feature_directory = join(Config.feature_directory, version_id)
    if mode:
        result_directory = join(Config.result_directory, version_id)
        dirs = listfile(Config.dev_directory)
        info("load_directory <- development")
    else:
        result_directory = join(Config.result_directory, "eval_{}".format(version_id))
        dirs = listfile(Config.eval_directory)
        info("load_directory <- evaluation")
    os.makedirs(result_directory, exist_ok=True)

    device = torch.device(Config.cuda if torch.cuda.is_available() else "cpu")

    csv_lines = []
    for idx, target_dir in enumerate(sorted(dirs)):
        print("======= BEGIN TEST FOR A MACHINE TYPE =======[{idx}/{total}]".format(idx=idx+1, total=len(dirs)))  
        print(target_dir)
        machine_type = os.path.split(target_dir)[1]

        print("============== MODEL LOAD ==============")
        model_file = "{}/model_{}.pkl".format(model_directory, machine_type)
        if not os.path.exists(model_file):
            com.logger.error("{} model not found ".format(machine_type))
            continue
        model = torch.load(model_file).to(device)
        model.eval()

        if mode:
            csv_lines.append([machine_type])
            csv_lines.append(["id", "AUC", "pAUC"])
            performance = []

        machine_id_list = get_machine_id_list_for_test(target_dir)

        for id_str in machine_id_list:
            normal_feature = np.loadtxt(join(feature_directory, "feature_{}_{}.csv".format(machine_type, id_str.split('_')[1])))
            
            test_files, y_true = test_file_list_generator(target_dir, id_str, mode=mode)

            # setup anomaly score file path
            anomaly_score_csv = "{result}/anomaly_score_{machine_type}_{id_str}.csv".format(
                result=result_directory,
                machine_type=machine_type,
                id_str=id_str)
            anomaly_score_list = []

            print("====== BEGIN TEST FOR A MACHINE ID ======")
            y_pred = [0. for k in test_files]
            for file_idx, file_path in enumerate(test_files):
                data, _ = file_to_stft_array(file_path)
                data = np.expand_dims(data, axis=1)
                data = data.astype(np.float32)
                data = torch.from_numpy(data)
                data = data.to(device)

                feature = model(data)
                feature = feature.cpu().detach().numpy()
                feature /= np.linalg.norm(feature, axis=1, keepdims=True)
                normal_feature /= np.linalg.norm(normal_feature, axis=1, keepdims=True)

                score = np.dot(feature, normal_feature.T)
                score = (1 - np.sort(score, axis=1)[:, -10:].mean(axis=1))
                score = score.mean()

                y_pred[file_idx] = score
                anomaly_score_list.append([os.path.basename(file_path), score])

            # save anomaly score
            save_csv(save_file_path=anomaly_score_csv, save_data=anomaly_score_list)

            if mode:
                # append AUC and pAUC to lists
                auc = metrics.roc_auc_score(y_true, y_pred)
                p_auc = metrics.roc_auc_score(y_true, y_pred, max_fpr=0.1)
                csv_lines.append([id_str.split("_", 1)[1], auc, p_auc])
                performance.append([auc, p_auc])
                info("AUC : {}".format(auc))
                info("pAUC : {}".format(p_auc))

            print("====== END OF TEST FOR A MACHINE ID ======\n")

        if mode:
            # calculate averages for AUCs and pAUCs
            averaged_performance = np.mean(np.array(performance, dtype=float), axis=0)
            csv_lines.append(["Average"] + list(averaged_performance))
            csv_lines.append([])

    if mode:
        # output results
        result_path = "{result}/{file_name}".format(
            result=result_directory, file_name=Config.result_file)
        info("AUC and pAUC results -> {}".format(result_path))
        save_csv(save_file_path=result_path, save_data=csv_lines)


if __name__ == "__main__":
    main()
