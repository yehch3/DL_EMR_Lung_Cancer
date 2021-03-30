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
from my_models import MyLayer

def get_names(filepath):
    parts = tf.strings.split(filepath,os.path.sep)
    name = tf.strings.split(parts[-1],'.')[0]
    name_parts = tf.strings.split(name,'_')
    return name_parts

def decode_image(img):
  # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_png(img, channels=3)
    return img

def get_dataset_v2(filelist, cache=True ,shuffle=True,batch_size=128):
    list_ds = tf.data.Dataset.from_tensor_slices(filelist)
    labeled_ds = list_ds.map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = prepare_for_training(labeled_ds,shuffle=shuffle,cache=cache,batch_size=batch_size)
    return ds

def process_path(filepath):
    '''input:path
       output: dict object for input'''
    name_parts = get_names(filepath)
    # id, group, sex, age, diagnosis_count, medication count
    img = tf.io.read_file(filepath)
    img = decode_image(img)
    img = img/7
    sex = tf.strings.to_number(name_parts[2])-1
    age = (tf.strings.to_number(name_parts[3])-50)/40
    s_a = tf.stack([sex,age])
    _grp = tf.strings.to_number(name_parts[1],tf.int32)
    grp = tf.one_hot(_grp,2)
    idx = tf.strings.to_number(name_parts[0])

    return ({'input1':s_a,'input2':img,'id':idx},grp)
#     return ({'input1':s_a,'input2':img},grp)

def prepare_for_training(ds, cache=True, shuffle = True, shuffle_buffer_size=5000,batch_size=128):
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


def get_pts_baseline(ids,folder):
    pts_dict ={}
    for pfn in folder:
        fn = pfn.split(os.path.sep)[-1]
        pts_dict[fn.split('_')[0]] = fn.split('.')[0]
    pt_ls = [[pts_dict[str(x)].split('_')[n] for n in [0,2,3,4,5]] for x in ids] #id,sex,age,diag,drug
    return np.asarray(pt_ls).astype(int)


def get_dataset_3layer(filelist, cache=True ,shuffle=True,batch_size=128):
    list_ds = tf.data.Dataset.from_tensor_slices(filelist)
    labeled_ds = list_ds.map(process_path_3layer, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = prepare_for_training(labeled_ds,shuffle=shuffle,cache=cache,batch_size=batch_size)
    return ds

def process_path_3layer(filepath):
    '''input:path
       output: dict object for input
       stack three years data to cut down sample size'''
    name_parts = get_names(filepath)
    # id, group, sex, age, diagnosis_count, medication count
    img = tf.io.read_file(filepath)
    img = decode_image(img)
    img = img/28
    img = tf.stack([img[0:643,:,1],img[643:1286,:,1],img[1286:,:,1]],2)
    img = tf.cast(img,tf.float32)
    sex = tf.strings.to_number(name_parts[2])-1
    age = (tf.strings.to_number(name_parts[3])-50)/40
    s_a = tf.stack([sex,age])
    _grp = tf.strings.to_number(name_parts[1],tf.int32)
    grp = tf.one_hot(_grp,2)
    idx = tf.strings.to_number(name_parts[0])
    return ({'input1':s_a,'input2':img,'id':idx},grp)


def make_dataset(pn, img_stack):
    fn = pn.split(os.path.sep)[-1]
    name_parts = fn.split('_')
    sex = float(name_parts[2]) - 1
    age = (float(name_parts[3]) - 50) / 40
    s_a = np.tile([sex, age], (len(img_stack), 1))
    imgs = np.asarray(img_stack) / 7

    #     _s_a = tf.cast(s_a,tf.float32)
    #     _imgs = tf.cast(imgs,tf.float32)
    ds = tf.data.Dataset.from_tensor_slices((s_a, imgs))
    ds = ds.map(mk_tensor)
    ds = prepare_for_training(ds, shuffle=False, cache=False, batch_size=16)
    return ds


def mk_tensor(sa, im):
    _s_a = tf.cast(sa, tf.float32)
    _imgs = tf.cast(im, tf.float32)
    return {'input1': _s_a, 'input2': _imgs}


# In[3]:


def main(args):
    set_gpu()
    # strategy = tf.distribute.MirroredStrategy()

    # MATCH= True
    LR = args.learning_rate
    BATCH_SIZE = 64
    tf.random.set_seed(42)
    np.random.seed(42)
    OPT = args.opt  # adam: AUC: 0.909, 0917 Loss:0.28313
    BASE_MODEL = args.model
    CACHE = False
    FINAL_NODE = args.fn
    EXP_NAMES =args.experiment
    BATCH_NAME = 256
    ACT = args.activation
    print('excute: {}'.format(EXP_NAMES))

    for EXP_NAME in EXP_NAMES:

        with open(os.path.join(os.getcwd(),'experiments',EXP_NAME+'.pkl'),'rb') as f:
            file_list_dict = pk.load(f)

        checkpoint_path = "./checkpoints/CP_{:s}_{:s}_{:.2e}_{:d}_{:s}_{:d}_{:s}/cp.ckpt".format(str(OPT),str(BASE_MODEL), float(LR), int(BATCH_NAME),str(EXP_NAME),int(FINAL_NODE),ACT)
        # checkpoint_path = "./checkpoints/CP_adam_Xception_zero_1.00e-03_256_HPC_nfd2012_dd_match10_0_30/cp.ckpt".format(str(OPT),str(BASE_MODEL), float(LR), int(BATCH_SIZE),str(EXP_NAME),int(FINAL_NODE))
        checkpoint_dir = os.path.dirname(checkpoint_path)

        EXP_NAME = EXP_NAME+'_relu'


        TRAIN_SIZE = len(file_list_dict['train'])
        VAL_SIZE = len(file_list_dict['val'])
        TEST_SIZE = len(file_list_dict['test'])
        print("train:val:test = {:d}:{:d}:{:d} imgs".format(TRAIN_SIZE,VAL_SIZE,TEST_SIZE))
        print('checkpoint path:{}'.format(checkpoint_dir))

        # set training weight

        # Create a callback that saves the model's weights

        if CACHE:
            # use memory cache
            val_ds = get_dataset_v2(file_list_dict['val'], shuffle=False, batch_size=BATCH_SIZE)
            test_ds = get_dataset_v2(file_list_dict['test'], shuffle=False, batch_size=BATCH_SIZE)
        else:
            # do not use cache
            val_ds = get_dataset_v2(file_list_dict['val'], shuffle=False, cache=False, batch_size=BATCH_SIZE)
            test_ds = get_dataset_v2(file_list_dict['test'], shuffle=False, cache=False, batch_size=BATCH_SIZE)


        model = make_commercial(BASE_MODEL, OPT, final= FINAL_NODE,lr=LR, trainable=False, train_layer=0, no_train = True,act=ACT)
        # if os.path.exists(os.path.dirname(checkpoint_path)):
        model.load_weights(checkpoint_path)
        print('model loaded')
        w = nvidia_smi()
        print('after gpu:{}'.format(w))
        print('training ended for...{}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))


        logits = model.get_layer('logits').output
        model_calib = models.Model(inputs=model.input, outputs=logits)

        # =======================Calibration=====================
        roc_val, roc_test, sen, spe, t, x_val_id, val_pred_raw, y_val, x_test_id, test_pred_raw, y_test = evaluate_result_and_id(
            model, val_ds, test_ds, plot=False)
        y_val_c = y_val
        y_val_c = np.asarray(y_val_c)
        val_logit_raw_c = model_calib.predict(val_ds)
        print('val_done')

        y_test_c = y_test
        y_test_c = np.asarray(y_test_c)
        test_logits_raw_c = model_calib.predict(test_ds)
        (test_result_calib, y_test_calib), model_top = model_calibration_with_temp(val_logit_raw_c, y_val_c, test_logits_raw_c,
                                                                        y_test_c)
        save_dict = {
            'EXP_NAME': EXP_NAME,
            'CP': checkpoint_path,
            'X_VAL_ID': x_val_id,
            'VAL_PRED': val_pred_raw,
            'VAL_LOGITS':val_logit_raw_c,
            'Y_VAL': y_val,
            'X_TEST_ID': x_test_id,
            'TEST_PRED': test_pred_raw,
            'TEST_LOGITS':test_logits_raw_c,
            'Y_TEST': y_test,
            'TEST_CALIB': test_result_calib,
            'Y_TEST_CALIB': y_test_calib
        }
        if not os.path.exists('predictions/' + EXP_NAME+'_.pkl'):
            with open('predictions/' + EXP_NAME + '_.pkl', 'wb') as f:
                pk.dump(save_dict, f)
        model_top.save('predictions/'+EXP_NAME+'_top.h5')


        # =====================Occlusion=========================
        with open('predictions/' + EXP_NAME+'_.pkl', 'rb') as f:
            r = pk.load(f)

        # connect the models to get calibration
        logits = model.get_layer('logits').output
        model_calib = models.Model(inputs=model.input, outputs=logits)
        model_top = tf.keras.models.load_model('predictions/'+EXP_NAME+'_top.h5', compile=False,
                                               custom_objects={'MyLayer': MyLayer()})
        model_top.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), loss='categorical_crossentropy')
        model_final_out = model_top(model_calib.outputs)
        model_final = models.Model(inputs=model_calib.inputs, outputs=model_final_out)
        model_final.trainable = False
        model_final.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), loss='categorical_crossentropy')
        model_final.save('predictions/'+EXP_NAME+'_final.h5')
        os.system('echo "model saved"')
        from PIL import Image
        # create records on the way
        if not os.path.exists('predictions/rec_'+EXP_NAME+'.pkl'):
            rec = -1
            summary = dict()
            for n in range(1929):
                summary[n] = []
            print('creating new')
        else:
            with open('predictions/rec_'+EXP_NAME+'.pkl','rb') as f:
                a = pk.load(f)
                rec = a[0]
                summary = a[1]

        for c, pn in enumerate(file_list_dict['test']):
            if rec > c: continue
            img = np.array(Image.open(pn))
            rows = identify_row(img)
            img_oc = apply_mask(img, rows)
            input_ds = make_dataset(pn, img_oc)
            after = model_final.predict(input_ds)
            input_before = make_dataset(pn, img[np.newaxis, ...])
            # before = r['TEST_PRED'][c, 1]
            before = model_final.predict(input_before)[:, 1]
            idx = pn.split(os.path.sep)[-1].split('_')[0]

            for cr, (row_n, rate) in enumerate(zip(rows, after[:, 1] - before)):
                summary[row_n].append((idx, r['Y_TEST'][c, 1], rate, after[cr, 1]))
            rec = c

            if rec % 1000 == 1:
                with open('predictions/rec_'+EXP_NAME+'.pkl', 'wb') as f:
                    pk.dump([rec, summary], f)
                print('{}_{}'.format(rec, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))


        print('test_done')
        # with open(os.path.join(os.getcwd(),'experiments','RESULTS_all_data.csv'),'a',newline='') as f:
        #     writer = csv.writer(f)
        #     # writer.writerow([EXP_NAME,BASE_MODEL,OPT,LR,TRAIN_SIZE,TEST_SIZE,roc_val,roc_test,sen,spe,sen+spe,roc_a,roc_b])
        #     writer.writerow([EXP_NAME,BASE_MODEL,OPT,LR,TRAIN_SIZE,TEST_SIZE,roc_val,roc_test,sen,spe,sen+spe,t])
        #     print('writing to csv...done')
        print('program ended for...{}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))

    return


if __name__== "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='Xception')
    parser.add_argument('-b', '--batch', default=128, type=int)
    parser.add_argument('-e', '--experiment', nargs='+')
    parser.add_argument('-lr', '--learning_rate',type=float,default=0.003)
    parser.add_argument('-o','--opt', default='sgd')
    parser.add_argument('-n', '--fn', default=30)
    parser.add_argument('-a','---activation', default='linear')
    args = parser.parse_args()
    main(args)




