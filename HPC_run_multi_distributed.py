#!/usr/bin/env python
# coding: utf-8

import numpy as np
import tensorflow as tf
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import datasets, layers, models, callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import datetime
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras import backend
from sklearn.manifold import TSNE
import sklearn
import itertools
import pickle as pk
import csv
import sys
sys.path.append('../models')
from dataset import *
from my_models import *
from utils import *
from my_optimizer import *
from subprocess import check_output


# In[2]:


def get_names(filepath):
    parts = tf.strings.split(filepath,os.path.sep)
    name = tf.strings.split(parts[-1],'.')[0]
    name_parts = tf.strings.split(name,'_')
    return name_parts

def decode_image(img):
  # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_png(img, channels=3)
    return img

def process_path(filepath):
    '''input:path
       output: dict object for input'''
    name_parts = get_names(filepath)
    # id, group, sex, age, diagnosis_count, medication count
    img = tf.io.read_file(filepath)
    img = decode_image(img)
    img = img/7
    sex = tf.strings.to_number(name_parts[2])-1
    age = (tf.strings.to_number(name_parts[3])-50)/20
    s_a = tf.stack([sex,age])
    _grp = tf.strings.to_number(name_parts[1],tf.int32)
    grp = tf.one_hot(_grp,2)
    idx = tf.strings.to_number(name_parts[0])
    
    return ({'input1':s_a,'input2':img,'id':idx},grp)
#     return ({'input1':s_a,'input2':img},grp)

def prepare_for_training(ds, cache=True, shuffle = True, repeat = True, shuffle_buffer_size=25000,batch_size=128):
  # This is a small dataset, only load it once, and keep it in memory.
  # use `.cache(filename)` to cache preprocessing work for datasets that don't
  # fit in memory.
    if cache:
        if isinstance(cache, str):
            ds = ds.cache(cache)
        else:
            ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(buffer_size=shuffle_buffer_size)
    ds = ds.batch(batch_size)
    # `prefetch` lets the dataset fetch batches in the background while the model
    # is training.
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return ds

def get_dataset_v2(filelist, cache=True ,shuffle=True,batch_size=128):
    list_ds = tf.data.Dataset.from_tensor_slices(filelist)
    labeled_ds = list_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = prepare_for_training(labeled_ds,shuffle=shuffle,cache=cache,batch_size=batch_size)
    return ds


def get_pts_baseline(ids,folder):
    pts_dict ={}
    for pfn in folder:
        fn = pfn.split('/')[-1]
#         print(fn)
        pts_dict[fn.split('_')[0]] = fn.split('.')[0]
    pt_ls = [[pts_dict[str(x)].split('_')[n] for n in [0,2,3,4,5]] for x in ids] #id,sex,age,diag,drug
    return np.asarray(pt_ls).astype(int)

def nvidia_smi(options=['-q','-d','MEMORY']):
    return check_output(['nvidia-smi'] + options)


# In[3]:


def main(EXP_NAMES):
    set_gpu()
    strategy = tf.distribute.MirroredStrategy()

    # MATCH= True
    PADDING = False # use false now. this feature will be developed in the future
    LR =0.0003* strategy.num_replicas_in_sync
    # BATCH_SIZE = 2048
    BATCH_SIZE_PER_REPLICA = 2048
    BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync
    tf.random.set_seed(42)
    np.random.seed(42)
    OPT = 'adam' 
    BASE_MODEL = 'base'
    CACHE =False

    print('excute: {}'.format(EXP_NAMES))

    for EXP_NAME in EXP_NAMES:

        with open(os.path.join(os.getcwd(),'experiments',EXP_NAME+'.pkl'),'rb') as f:
            file_list_dict = pk.load(f)

        log_dir = os.path.join("logs", "{:s}_{:s}_{:.2e}_{:d}_{:s}_".format(OPT,BASE_MODEL, LR, BATCH_SIZE,EXP_NAME)+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

        checkpoint_path = "./checkpoints/CP_{:s}_{:s}_{:.2e}_{:d}_{:s}/cp.ckpt".format(OPT,BASE_MODEL, LR, BATCH_SIZE,EXP_NAME)
        checkpoint_dir = os.path.dirname(checkpoint_path)

        TRAIN_SIZE = len(file_list_dict['train'])
        VAL_SIZE = len(file_list_dict['val'])
        TEST_SIZE = len(file_list_dict['test'])
        print("train:val:test = {:d}:{:d}:{:d} imgs".format(TRAIN_SIZE,VAL_SIZE,TEST_SIZE))
        print('checkpoint path:{}'.format(checkpoint_dir))

        img_input =[]
        for pn in file_list_dict['train']:
            img_input.append(pn.split('/')[-1])


        # img_input = os.listdir(train_dir)
        count_ag = np.zeros((len(img_input)))
        count_male = np.zeros((len(img_input)))
        count_grp = np.zeros((len(img_input)))

        case_ag =[]
        case_male =[]
        case_diag = []
        case_meds = []

        ctrl_ag =[]
        ctrl_male =[]
        ctrl_diag = []
        ctrl_meds = []

        #     format: id, group, sex, age, diagnosis_count, medication count
        for c, i in enumerate(img_input):
            if i.split('_')[1] == '0':
                ctrl_ag.append(int(i.split('_')[3]))
                ctrl_male.append(int(i.split('_')[2])-1)
                ctrl_diag.append(int(i.split('_')[4]))
                ctrl_meds.append(int(i.split('_')[5].split('.')[0]))
            elif i.split('_')[1] == '1':
                case_ag.append(int(i.split('_')[3]))
                case_male.append(int(i.split('_')[2])-1) 
                case_diag.append(int(i.split('_')[4]))
                case_meds.append(int(i.split('_')[5].split('.')[0]))        

        case_ag = np.asarray(case_ag)
        case_male =np.asarray(case_male)
        case_diag = np.asarray(case_diag)
        case_meds = np.asarray(case_meds)

        ctrl_ag =np.asarray(ctrl_ag)
        ctrl_male =np.asarray(ctrl_male)
        ctrl_diag = np.asarray(ctrl_diag)
        ctrl_meds = np.asarray(ctrl_meds)
        print('CANCER: mean age:{:4.2f}+-{:4.2f}, male:{:4.2f}, diagnosis:{:4.2f}+-{:4.2f},medication:{:4.2f}+-{:4.2f}, training tot:{}'.format(
        case_ag.mean(),
        case_ag.std(),
        case_male.sum()/len(case_ag),
        case_diag.mean(),
        case_diag.std(),
        case_meds.mean(),
        case_meds.std(),
        len(case_ag)))
        print('CONTROL: mean age:{:4.2f}+-{:4.2f}, male:{:4.2f}, diagnosis:{:4.2f}+-{:4.2f},medication:{:4.2f}+-{:4.2f}, training tot:{}'.format(
        ctrl_ag.mean(),
        ctrl_ag.std(),
        ctrl_male.sum()/len(ctrl_ag),
        ctrl_diag.mean(),
        ctrl_diag.std(),
        ctrl_meds.mean(),
        ctrl_meds.std(),
        len(ctrl_ag)))


        tb = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                            embeddings_freq=1)

        # set training weight

        # Create a callback that saves the model's weights
        cp = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                monitor='val_loss',
                                                save_weights_only=True,
                                                save_best_only=True,
                                                mode='min',
                                                verbose=0)
        es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        with strategy.scope():
            if CACHE:
                # use memory cache
                train_ds = get_dataset_v2(file_list_dict['train'], batch_size=BATCH_SIZE)
                val_ds = get_dataset_v2(file_list_dict['val'], shuffle=False, batch_size=BATCH_SIZE)
                test_ds = get_dataset_v2(file_list_dict['test'], shuffle=False, batch_size=BATCH_SIZE)
            else:
                # do not use cache
                train_ds = get_dataset_v2(file_list_dict['train'], cache=False, batch_size=BATCH_SIZE)
                val_ds = get_dataset_v2(file_list_dict['val'], shuffle=False, cache=False, batch_size=BATCH_SIZE)
                test_ds = get_dataset_v2(file_list_dict['test'], shuffle=False, cache=False, batch_size=BATCH_SIZE)
                # val_ds_l = get_dataset_v2(file_list_dict['val'],shuffle=False,cache=False,batch_size=BATCH_SIZE_PER_REPLICA)
                # test_ds_l = get_dataset_v2(file_list_dict['test'],shuffle=False,cache=False,batch_size=BATCH_SIZE_PER_REPLICA)
            weight_0 = 0.5 * (len(ctrl_ag) + len(case_ag)) / len(ctrl_ag)
            weight_1 = 0.5 * (len(ctrl_ag) + len(case_ag)) / len(case_ag)
            print('weight0:{} weight1:{}'.format(weight_0, weight_1))
            class_weight = {0: weight_0, 1: weight_1}

            model = make_model_test_ranger(opt=OPT, padding=PADDING, lr=LR,lr_decay=False )
            if os.path.exists(os.path.dirname(checkpoint_path)):
                model.load_weights(checkpoint_path)
                print('model loaded')
            print('training started for...{}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
            zero_bias_history = model.fit(
                train_ds,
                initial_epoch=0,
                epochs=1,
                validation_data=val_ds, verbose=0, callbacks=[tb, cp, es], class_weight=class_weight)
            w = nvidia_smi()
            print('after gpu:{}'.format(w))
            print('training ended for...{}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

            model = make_model_test_ranger(opt=OPT, padding=PADDING, lr=LR,lr_decay=False )
            model.load_weights(checkpoint_path)
            # val_set
            roc_val, roc_test, sen, spe, t = evaluate_result(model, val_ds, test_ds, plot=True)

            x_test = get_pts_baseline(x_test_id,file_list_dict['test'])
            # x_val = get_pts_baseline(x_val_id,file_list_dict['val'])
            result_test = pd.DataFrame(data=np.concatenate((x_test,y_test[:,1][:,np.newaxis],test_pred_raw[:,1][:,np.newaxis]),axis=1),
                                 index=[i for i in range(x_test.shape[0])],
                                 columns=['id','sex','age', 'diag_count', 'med_count', 'grp','pred'])

            # result_val = pd.DataFrame(data=np.concatenate((x_val,y_val[:,1][:,np.newaxis],val_pred_raw[:,1][:,np.newaxis]),axis=1),
            #                      index=[i for i in range(x_val.shape[0])],
            #                      columns=['id','sex','age', 'diag_count', 'med_count', 'grp','pred'])
            # age_group=[18,35,45,55,65,75,85,95]
            age_group=[18,55,95]
            if EXP_NAME.split('_')[-1] == 'above55' :
                sen_a,spe_a,roc_a=subgroup_analysis(result_test,'age',[55,95],plot=False)
                sen_b,spe_b,roc_b =0,0,0
            elif EXP_NAME.split('_')[-1] == 'below55' :
                sen_b,spe_b,roc_b=subgroup_analysis(result_test,'age',[18,55],plot=False)
                sen_a,spe_a,roc_a = 0,0,0
            else:
                sen_a,spe_a,roc_a=subgroup_analysis(result_test,'age',[55,95],plot=False)
                sen_b,spe_b,roc_b=subgroup_analysis(result_test,'age',[18,55],plot=False)
            # record results
            with open(os.path.join(os.getcwd(),'experiments','RESULTS.csv'),'a',newline='') as f:
                writer = csv.writer(f)
                writer.writerow([EXP_NAME,BASE_MODEL,OPT,TRAIN_SIZE,TEST_SIZE,roc_val,roc_test,sen,spe,sen+spe,roc_a,roc_b])
                print('writing to csv...done')
    return


if __name__== "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-e','--experiment',nargs='+')
    args = parser.parse_args()
    main(args.experiment)




