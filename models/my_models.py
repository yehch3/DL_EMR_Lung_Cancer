import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import datasets, layers, models, callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, LSTM,Bidirectional, Activation
import datetime
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras import backend
from sklearn.manifold import TSNE
import sklearn
import itertools
import tempfile
from my_optimizer import *
# import keras.backend as K
import tensorflow.keras.backend as K
# from keras_radam.training import RAdamOptimizer





# def MyAUC(y_true, y_pred):
#     return sklearn.metrics.roc_auc_score(y_true[:,1], y_pred[:,1])


def make_model_agesex_only(opt='adam',lr=0.001,lr_decay=False ):
    '''test input with age and sex only, no EMR'''

    metrics = [tf.keras.metrics.AUC(name='auc')]
    # metrics = [MyAUC]
    input1 = tf.keras.layers.Input(shape=(2,),name='input1')

    x = tf.keras.layers.Dense(30)(input1)
    drop = tf.keras.layers.Dropout(0.1)(x)
    logits = tf.keras.layers.Dense(2,name='logits')(drop)
    pred = tf.keras.layers.Activation('softmax',name='predictions')(logits)
    model = models.Model(inputs=input1, outputs=pred)
    model = compile_with_opt(model, metrics, opt, lr, lr_decay)

    return model



def make_model_1d(opt='adam',lr=0.001,lr_decay=False):
    '''model for input 1d binarized EMR'''

    metrics = [tf.keras.metrics.AUC(name='auc')]
    # metrics = [MyAUC]
    input1 = tf.keras.layers.Input(shape=(2,), name='input1')
    input2 = tf.keras.layers.Input(shape=(1929, 3), name='input2')
    x1 = tf.keras.layers.Conv1D(8, 3, padding='valid', activation='relu')(input2)
    x1 = tf.keras.layers.MaxPooling1D(2)(x1)
    x1 = tf.keras.layers.Conv1D(16, 3, padding='valid', activation='relu')(x1)
    x1 = tf.keras.layers.MaxPooling1D(2)(x1)
    x1 = tf.keras.layers.Conv1D(16, 3, padding='valid', activation='relu')(x1)
    x1 = tf.keras.layers.MaxPooling1D(2)(x1)
    x1 = tf.keras.layers.Conv1D(8, 3, padding='valid', activation='relu', name='heatmap')(x1)
    x1 = tf.keras.layers.MaxPooling1D(2)(x1)
    x1 = tf.keras.layers.Flatten()(x1)
    x1 = tf.keras.layers.concatenate([x1, input1])

    x1 = tf.keras.layers.Dense(30)(x1)
    #     emb = tf.keras.layers.GlobalAveragePooling2D()(x1)
    drop = tf.keras.layers.Dropout(0.1)(x1)
    logits = tf.keras.layers.Dense(2, name='logits')(drop)
    pred = tf.keras.layers.Activation('softmax', name='predictions')(logits)
    model = models.Model(inputs=[input1, input2], outputs=pred)
    model = compile_with_opt(model, metrics, opt, lr, lr_decay)

    return model


def make_commercial(name, opt, lr=0.001,lr_decay=False,trainable=False,train_layer=-18,final=30,act='linear',no_train=False, dropout=False, fcloss=False):
    '''testing all on-shelf models of tensorflow
    main model training
    name : model name, can be 'base', or other commercial model name
    opt : optimizer name
    '''

    metrics = [tf.keras.metrics.CategoricalAccuracy(name='CatAccuracy'), tf.keras.metrics.AUC(name='auc')]
    input1 = tf.keras.layers.Input(shape=(2,), name='input1')
    input2 = tf.keras.layers.Input(shape=(1929, 157, 3), name='input2')
    if name == 'base':
        x1 = tf.keras.layers.Conv2D(16, 3, padding='valid', activation='relu')(input2)
        x1 = tf.keras.layers.MaxPooling2D(2)(x1)
        x1 = tf.keras.layers.Conv2D(32, 3, padding='valid', activation='relu')(x1)
        x1 = tf.keras.layers.MaxPooling2D(2)(x1)
        x1 = tf.keras.layers.Conv2D(32, 3, padding='valid', activation='relu')(x1)
        x1 = tf.keras.layers.MaxPooling2D(2)(x1)
        x1 = tf.keras.layers.Conv2D(16, 3, padding='valid', activation='relu', name='heatmap')(x1)
        x1 = tf.keras.layers.MaxPooling2D(2)(x1)
        x1 = tf.keras.layers.Flatten()(x1)
        x1 = tf.keras.layers.concatenate([x1, input1])

        x1 = tf.keras.layers.Dense(30)(x1)
        #     emb = tf.keras.layers.GlobalAveragePooling2D()(x1)
        drop = tf.keras.layers.Dropout(0.1)(x1)
        logits = tf.keras.layers.Dense(2, name='logits')(drop)
        pred = tf.keras.layers.Activation('softmax', name='predictions')(logits)
        model = models.Model(inputs=[input1, input2], outputs=pred)
    else:
        # input2 = tf.keras.layers.ZeroPadding2D(((0,7),(0,3)))(input2)
        if name=='DenseNet121':
            base_model= tf.keras.applications.DenseNet121(input_shape=(1929, 157, 3), include_top=False, pooling='avg',input_tensor=input2)

        elif name == 'InceptionV3':
            base_model = tf.keras.applications.InceptionV3(input_shape=(1929, 157, 3), include_top=False,
                                                                input_tensor=input2)
        elif name=='NASNetLarge':
            base_model = tf.keras.applications.NASNetLarge(input_shape=(1929, 157, 3), pooling='avg',include_top=False, input_tensor=input2)
        elif name == 'InceptionResNetV2':
            base_model = tf.keras.applications.InceptionResNetV2(input_shape=(1929, 157, 3), include_top=False, input_tensor=input2
                                                               )
        elif name=='ResNet50V2':
            base_model = tf.keras.applications.ResNet50V2(input_shape=(1929, 157, 3), include_top=False, input_tensor=input2)
        elif name == 'Xception':
            base_model = tf.keras.applications.Xception(input_shape=(1929, 157, 3), include_top=False,
                                                               pooling='avg',input_tensor=input2)
        elif name == 'Xception_zero':
            base_model = tf.keras.applications.Xception(input_shape=(1929, 157, 3), include_top=False,
                                                               pooling='avg',input_tensor=input2)
        elif name == 'ResNet101V2':
            base_model = tf.keras.applications.ResNet101V2(input_shape=(1929, 157, 3), include_top=False,
                                                           pooling='avg' ,input_tensor=input2)
        elif name== 'EfficientB7':
            import efficientnet.tfkeras as efn
            base_model =efn.EfficientNetB7(input_shape=(1929, 157, 3), include_top=False, pooling='avg',input_tensor=input2)
        elif name== 'EfficientB4':
            import efficientnet.tfkeras as efn
            base_model =efn.EfficientNetB4(input_shape=(1929, 157, 3), include_top=False, pooling='avg',input_tensor=input2)
        elif name== 'EfficientB5':
            import efficientnet.tfkeras as efn
            base_model =efn.EfficientNetB5(input_shape=(1929, 157, 3), include_top=False, pooling='avg',input_tensor=input2)
        elif name== 'EfficientB6':
            import efficientnet.tfkeras as efn
            base_model =efn.EfficientNetB6(input_shape=(1929, 157, 3), include_top=False, pooling='avg',input_tensor=input2)

        for layer in base_model.layers[:train_layer]:
            layer.trainable = False #lock baselayers

        for layer in base_model.layers[train_layer:]:
            layer.trainable = trainable

        x1 = base_model.output
        x1 = tf.keras.layers.Flatten()(x1)
        x1 = tf.keras.layers.concatenate([x1, input1])


        x1 = tf.keras.layers.Dense(final,activation=act)(x1)
        # prediction = tf.keras.layers.Dense(2, activation='softmax', name='predictions')(x1)
        # model = models.Model(inputs=[input1,input2], outputs=prediction)
        if dropout:
            x1 = tf.keras.layers.Dropout(0.2)(x1)

        if fcloss:
            logits = tf.keras.layers.Dense(2, name='logits',
                                           kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.0001),
                                           bias_initializer=tf.keras.initializers.Constant(value=-2))(x1)
            print('set biased output at last layer')
        else:
            logits = tf.keras.layers.Dense(2, name='logits')(x1)
        pred= tf.keras.layers.Activation('softmax', name='predictions')(logits)
        model = models.Model(inputs=[input1, input2], outputs=pred)
        if no_train:
            model.trainable = False
    model = compile_with_opt(model, metrics, opt, lr, lr_decay,fcloss)
    return model



def compile_with_opt(model, metrics, opt, lr, lr_decay, fcloss=False):
    '''choose different optimizer setting'''


    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        lr,
        decay_steps=1000,
        decay_rate=0.90,
        staircase=True)
    if opt=='adam':
        if lr_decay:
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    elif opt =='ranger':
        optimizer = RAdamOptimizer(learning_rate=lr,beta1=0.95 ,warmup_proportion=0.1, min_lr=1e-5)
        optimizer = Lookahead(optimizer, la_steps=5, la_alpha=0.5)
    elif opt=='sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)


    if fcloss:
        model.compile(optimizer=optimizer, loss=FocalLoss(alpha=0.5,gamma=2.5), metrics=metrics)
        print('use focal loss')
    else:
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=metrics)
        print('use XE')
    return model


class MyLayer(tf.keras.layers.Layer):
    '''function for generating your our layer
    this is used to do model calibration'''
    def __init__(self,**kwargs):
        super(MyLayer, self).__init__()

        #your variable goes here
        self.variable = tf.Variable(0.01, trainable=True, dtype=tf.float32)

    def call(self, inputs, **kwargs):

        # your mul operation goes here
        x = inputs / self.variable

        # x = tf.divide(inputs,self.variable)
        return x

def model_calibration_with_temp(input_val_logits,val_label,input_test_logits,test_label):
    '''temperature calibration, this is one of the most common ways to calibrate the model
    https://arxiv.org/abs/1706.04599'''
    ds_val_input = tf.data.Dataset.from_tensor_slices((input_val_logits,val_label)).shuffle(buffer_size=10000).batch(512)
    ds_val_input = ds_val_input.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    ds_test_input = tf.data.Dataset.from_tensor_slices((input_test_logits,test_label)).batch(512)
    ds_test_input = ds_test_input.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    input_x = tf.keras.layers.Input(shape=(2,),name='input_x')
    logits_temp = MyLayer()(input_x)
    pred = tf.keras.layers.Activation('softmax',name='prediction')(logits_temp)
    model = models.Model(inputs=input_x,outputs=pred)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer,loss='categorical_crossentropy')
    print(model.summary())
    _ = model.fit(ds_val_input,epochs=50,verbose=0)
    temperature = model.layers[1].get_weights()
    print('temp: {}'.format(temperature))
    # test_result = []
    # y_test = []
    # for i in ds_test_input:
    #     y_test.extend(i[1])
    #     pred = model.predict_on_batch(i[0])
    #     test_result.extend(pred)
    test_result = model.predict(ds_test_input)
    y_test = np.asarray(test_label)
    return (test_result,y_test), model




