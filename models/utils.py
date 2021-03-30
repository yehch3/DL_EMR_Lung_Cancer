import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import numpy as np
from tensorboard.plugins.hparams import api as hp
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from subprocess import check_output
import os
import pickle as pk
import pandas as pd

def set_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print('allowing memory growth of GPU')
        except RuntimeError as e:
            print(e)

def nvidia_smi(options=['-q','-d','MEMORY']):
    return check_output(['nvidia-smi'] + options)


def plot_cm(labels, predictions, p=0.5, plot=True, print_out =True):
    '''this is used to plot confusion matrix, modified from tensorflow website'''
    cm = confusion_matrix(labels, predictions > p)
    #     plt.figure(figsize=(8,8))
    #     akws = {"ha": 'center',"va": 'center'}
    if plot:
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title('Confusion matrix @{:.2f}'.format(p))
        #     plt.ylim([-0.5,1.5])
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.show()
    sen = cm[1][1]/(cm[1][1]+ cm[1][0])
    spe = cm[0][0] / (cm[0][0] + cm[0][1])
    if print_out:
        print('True Non-cancer Detected (True Negatives): ', cm[0][0])
        print('False Non-cancer Detected (False Positives): ', cm[0][1])
        print('False Cancer Missed (False Negatives): ', cm[1][0])
        print('True cancer Detected (True Positives): ', cm[1][1])
        print('Total Cancer Transactions: ', np.sum(cm[1]))
        print('----')
        print('sensitivity: {:4.2f}'.format((cm[1][1]/( cm[1][1]+ cm[1][0]))))
        print('specificity: {:4.2f}'.format((cm[0][0] / (cm[0][0] + cm[0][1]))))
    return sen, spe

def plot_roc(name, labels, predictions, adjust_sen=True, plot=True, print_out=True, **kwargs):
    '''this is used to plot roc curve, modified from tensorflow website'''
    fp, tp, t = sklearn.metrics.roc_curve(labels, predictions)
    sesp = tp - (fp - 1)
    if adjust_sen:
        sesp[tp < 0.8] = 0
    i = sesp.argmax()
    sen = tp[i]
    spe = 1 - fp[i]
    roc = sklearn.metrics.roc_auc_score(labels, predictions)
    if print_out:
        print('sensitivity:{:2.3f}'.format(tp[i]))
        print('specificity:{:2.3f}'.format(1 - fp[i]))
        print('overall:{:5.3f}'.format(tp[i] + 1 - fp[i]))
        print('ROC:{:2.3f}'.format(sklearn.metrics.roc_auc_score(labels, predictions)))
    if plot:
        plt.figure()
        plt.plot(100 * fp, 100 * tp, label=name, linewidth=2, **kwargs)
        plt.xlabel('False positives [%]')
        plt.ylabel('True positives [%]')
        plt.xlim([-0.5, 100])
        plt.ylim([-0.5, 100.5])
        plt.grid(True)
        ax = plt.gca()
        ax.set_aspect('equal')
        plt.legend(loc='lower right')
        plt.show()

    return t[i], sen,spe,roc


# subgroup analysis
def subgroup_analysis(df, group_name, group_list, adj=False, plot=True):
    '''run subgroup analysis'''
    print('case:{}, control:{}, total:{}, ratio:{:2.2f}'.format(
        df[df.grp == 1].shape[0],
        df[df.grp == 0].shape[0],
        df.shape[0],
        df[df.grp == 1].shape[0] / df.shape[0]))
    plt.figure(figsize=(20, 20))
    for c, a in enumerate(group_list):
        if c == 0: continue
        subgroup = df[(df[group_name] >= group_list[c - 1]) & (df[group_name] < group_list[c])]
        if subgroup.empty: continue
        ratio = subgroup[subgroup.grp == 1].shape[0] / subgroup.shape[0]
        sub_case = subgroup[subgroup.grp == 1]
        sub_ctl = subgroup[subgroup.grp == 0]
        print(
            group_name + ' [case] {}-{} total:{}/{} age:{:2.2f}+-{:4.2f} dx:{:4.2f}+-{:4.2f} meds:{:4.2f}+-{:4.2f} ratio:{:2.2f}'.format(
                group_list[c - 1],
                group_list[c],
                sub_case.shape[0],
                subgroup.shape[0],
                sub_case.age.mean(),
                sub_case.age.std(),
                sub_case.diag_count.mean(),
                sub_case.diag_count.std(),
                sub_case.med_count.mean(),
                sub_case.med_count.std(),
                ratio))

        print(
            group_name + ' [control] {}-{} total:{}/{} age:{:2.2f}+-{:4.2f} dx:{:4.2f}+-{:4.2f} meds:{:4.2f}+-{:4.2f} ratio:{:2.2f}'.format(
                group_list[c - 1],
                group_list[c],
                sub_ctl.shape[0],
                subgroup.shape[0],
                sub_ctl.age.mean(),
                sub_ctl.age.std(),
                sub_ctl.diag_count.mean(),
                sub_ctl.diag_count.std(),
                sub_ctl.med_count.mean(),
                sub_ctl.med_count.std(),
                1-ratio))

        if ratio == 0:
            print(subgroup)
            continue
        s1 = [5 for i in range(sub_case.shape[0])]
        s2 = [5 for i in range(sub_ctl.shape[0])]
        if plot:
            plt.subplot(len(group_list), 4, c * 4 + 1)
            plt.scatter(sub_case.age, sub_case.diag_count, s1, alpha=0.3)
            plt.scatter(sub_ctl.age, sub_ctl.diag_count, s2, alpha=0.3)
            plt.subplot(len(group_list), 4, c * 4 + 2)
            plt.scatter(sub_case.age, sub_case.med_count, s1, alpha=0.3)
            plt.scatter(sub_ctl.age, sub_ctl.med_count, s2, alpha=0.3)
            plt.subplot(len(group_list), 4, c * 4 + 3)
            plt.scatter(sub_case.diag_count, sub_case.med_count, s1, alpha=0.3)
            plt.scatter(sub_ctl.diag_count, sub_ctl.med_count, s2, alpha=0.3)
            plt.subplot(len(group_list), 4, c * 4 + 4)
            plt.show()
        t, spe, sen, roc = plot_roc(group_list[c], subgroup.grp.to_numpy(), subgroup.pred.to_numpy(), adjust_sen=False)
        return spe,sen,roc
    return




def occlusion_sensitivity_by_row(df, img_stack, model):
    '''df: data_frame including 'id','sex','age', 'diag_count', 'med_count', 'grp','pred';img_stack: shape(N,H,W,C)'''
    summary = dict()
    for n in range(1929):
        summary[n] = []

    for c, img in enumerate(img_stack):
        r = identify_row(img)
        img_oc = apply_mask(img, r)  # for each img, generate n occluded ones
        after = model.predict(img_oc)
        before = model.predict(img[np.newaxis, ...])
        for row_n, rate in zip(r, after[:, 1] - before[:, 1]):
            summary[row_n].append((df.id[c], df.grp[c], row_n, before[:, 1], rate))
    return summary


def occlusion_sensitivity_by_row_HPC(model, list_of_img_path ):
    '''df: data_frame including 'id','sex','age', 'diag_count', 'med_count', 'grp','pred';img_stack: shape(N,H,W,C)'''
    from PIL import Image
    summary = dict()
    for n in range(1929):
        summary[n] = []
    img_stack = list(map(lambda x: np.array(Image.open(x)),list_of_imgs))

    for c, img in enumerate(img_stack):
        r = identify_row(img)
        img_oc = apply_mask(img, r)  # for each img, generate n occluded ones
        after = model.predict(img_oc)
        before = model.predict(img[np.newaxis, ...])
        for row_n, rate in zip(r, after[:, 1] - before[:, 1]):
            summary[row_n].append((df.id[c], df.grp[c], row_n, before[:, 1], rate))
    return summary


def identify_row(img):
    ret = []
    for rc, r in enumerate(img):
        if r.sum() != 0:
            ret.append(rc)
    return ret


def apply_mask(img, row_list):
    ret = np.zeros((len(row_list), img.shape[0], img.shape[1], img.shape[2]))
    for c, row in enumerate(row_list):
        ret[c, ...] = img
        ret[c, row, ...] = ret[c, row, ...] * 0
    return ret


def evaluate_result(model,val_ds,test_ds,plot=True):
#     val
    ADJ_SEN=False
    val_pred_raw = []
    y_val = []
    x_val_id = []
    for i in val_ds:
        y_val.extend(i[1])
        x_val_id.extend(i[0]['id'])
    pred = model.predict(val_ds)
    val_pred_raw.extend(pred)
    y_val =np.asarray(y_val)
    x_val_id = np.asarray(x_val_id).astype(int)
    val_pred_raw = np.asarray(val_pred_raw)
    val_pred = val_pred_raw.argmax(axis=1)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    t,_,_,roc_val = plot_roc("Test", y_val[:,1], val_pred_raw[:, 1], adjust_sen = ADJ_SEN, color=colors[0], linestyle='-',plot=plot)
    plot_cm(y_val[:,1], val_pred_raw[:, 1], t,plot=plot)
#     test
    # test_set
    test_pred_raw = []
    y_test = []
    x_test_id = []
    for i in test_ds:
        y_test.extend(i[1])
        x_test_id.extend(i[0]['id'])
    pred = model.predict(test_ds)
    test_pred_raw.extend(pred)
    y_test =np.asarray(y_test)
    x_test_id = np.asarray(x_test_id).astype(int)
    test_pred_raw = np.asarray(test_pred_raw)
    test_pred = test_pred_raw.argmax(axis=1)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    _,sen,spe,roc_test = plot_roc("Test", y_test[:,1], test_pred_raw[:, 1], adjust_sen = ADJ_SEN, color=colors[0], linestyle='-',plot=plot)
    plot_cm(y_test[:,1], test_pred_raw[:, 1], t,plot=plot)
    return roc_val, roc_test, sen, spe, t, test_pred_raw


def evaluate_result_and_id(model,val_ds,test_ds,plot=True):
#     val
    ADJ_SEN=False
    val_pred_raw = []
    y_val = []
    x_val_id = []
    for i in val_ds:
        y_val.extend(i[1])
        x_val_id.extend(i[0]['id'])
    pred = model.predict(val_ds)
    val_pred_raw.extend(pred)
    y_val =np.asarray(y_val)
    x_val_id = np.asarray(x_val_id).astype(int)
    val_pred_raw = np.asarray(val_pred_raw)
    val_pred = val_pred_raw.argmax(axis=1)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    t,_,_,roc_val = plot_roc("Test", y_val[:,1], val_pred_raw[:, 1], adjust_sen = ADJ_SEN, color=colors[0], linestyle='-',plot=plot)
    plot_cm(y_val[:,1], val_pred_raw[:, 1], t,plot=plot)
#     test
    # test_set
    test_pred_raw = []
    y_test = []
    x_test_id = []
    for i in test_ds:
        y_test.extend(i[1])
        x_test_id.extend(i[0]['id'])
    pred = model.predict(test_ds)
    test_pred_raw.extend(pred)
    y_test =np.asarray(y_test)
    x_test_id = np.asarray(x_test_id).astype(int)
    test_pred_raw = np.asarray(test_pred_raw)
    test_pred = test_pred_raw.argmax(axis=1)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    _,sen,spe,roc_test = plot_roc("Test", y_test[:,1], test_pred_raw[:, 1], adjust_sen = ADJ_SEN, color=colors[0], linestyle='-',plot=plot)
    plot_cm(y_test[:,1], test_pred_raw[:, 1], t,plot=plot)
    return roc_val, roc_test, sen, spe, t, x_val_id,val_pred_raw, y_val,x_test_id, test_pred_raw, y_test


def evaluate_result_and_id_v2(model,val_ds,test_ds, val_file_path, test_file_path, plot=True):
#     val
    ADJ_SEN=False
    a = [int(x.split(os.path.sep)[-1].split('_')[1]) for x in val_file_path] # original y_val
    a = np.array(a)
    a = np.clip(a,0,1)
    x_val_id = [int(x.split(os.path.sep)[-1].split('_')[0]) for x in val_file_path]
    # broadcast y_val to one hot
    y_val = np.zeros((a.shape[0], 2))
    y_val[np.arange(a.shape[0]), a] = 1
    # for i in val_ds:
    #     y_val.extend(i[1])
    #     x_val_id.extend(i[0]['id'])
    print('running val')
    pred = model.predict(val_ds)
    x_val_id = np.asarray(x_val_id).astype(int)
    val_pred_raw = np.asarray(pred)
    # val_pred = val_pred_raw.argmax(axis=1)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    t,_,_,roc_val = plot_roc("Test", y_val[:,1], val_pred_raw[:, 1], adjust_sen = ADJ_SEN, color=colors[0], linestyle='-',plot=plot)
    plot_cm(y_val[:,1], val_pred_raw[:, 1], t,plot=plot)
#     test
    # test_set
    b = [int(x.split(os.path.sep)[-1].split('_')[1]) for x in test_file_path] # original y_val
    b = np.array(b)
    b = np.clip(b,0,1)
    x_test_id = [int(x.split(os.path.sep)[-1].split('_')[0]) for x in test_file_path]
    y_test = np.zeros((b.shape[0], 2))
    y_test[np.arange(b.shape[0]), b] = 1
    # for i in test_ds:
    #     y_test.extend(i[1])
    #     x_test_id.extend(i[0]['id'])
    print('running test')
    pred = model.predict(test_ds)
    y_test =np.asarray(y_test)
    x_test_id = np.asarray(x_test_id).astype(int)
    test_pred_raw = np.asarray(pred)
    # test_pred = test_pred_raw.argmax(axis=1)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    _,sen,spe,roc_test = plot_roc("Test", y_test[:,1], test_pred_raw[:, 1], adjust_sen = ADJ_SEN, color=colors[0], linestyle='-',plot=plot)
    plot_cm(y_test[:,1], test_pred_raw[:, 1], t, plot=plot)
    return roc_val, roc_test, sen, spe, t, x_val_id,val_pred_raw, y_val,x_test_id, test_pred_raw, y_test

def HPC2local(dict_file):
    '''version used in lung cancer'''
    # CONTROL ='input_img/new_control_nfd_dd/'
    # CANCER = 'input_img/new_cancer_dd/'
    ret={'train':[],'val':[],'test':[]}
    for k in dict_file.keys():
        for i in dict_file[k]:
            fn = i.split('/')[-1]
            grp = int(fn.split('_')[1])
            if grp==1:
                ret[k].append(os.path.join('input_img', 'new_cancer_dd', fn))
            else:
                ret[k].append(os.path.join('input_img','new_control_nfd_dd',fn))
    return ret

def HPC2local_v2(dict_file):
    '''version used in cancer_prediction_raw'''

    ret={'train':[],'val':[],'test':[]}
    for k in dict_file.keys():
        for i in dict_file[k]:
            fn = i.split('/')[-1]
            grp = fn.split('_')[1]
            # if grp==cancer_folder.split('_')[-1]:
            if grp != '0':
                cancer_folder = 'cancer_'+grp
                ret[k].append(os.path.join('input_img', cancer_folder, fn))
            else:
                ret[k].append(os.path.join('input_img','cancer_0',fn))
    return ret


def get_subgroup_results_by_id(test_ids,test_preds, test_y):
    sub_list = ['HPC_nfd2012_dd_match10_0',
                'HPC_nfd2012_dd_above55_0',
                'HPC_nfd2012_dd_below55_0',
                'HPC_nfd2012_dd_ld_0',
                'HPC_nfd2012_dd_nld_0',
                'HPC_nfd2012_dd_above55_ld_0',
                'HPC_nfd2012_dd_above55_nld_0',
               'HPC_nfd2012_dd_below55_ld_0',
               'HPC_nfd2012_dd_below55_nld_0']
    pred_dict ={}
    for c, i in enumerate(test_ids):
        pred_dict[i] = (test_preds[c],test_y[c])
#         print(pred_dict[i])
    for sub in sub_list:
        with open('experiments/'+sub+'.pkl','rb') as f:
            d = pk.load(f)
        t_ids = list(map(lambda x:int(x.split('/')[-1].split('_')[0]),d['test']))
        y_pred = list(map(lambda x: pred_dict[x][0],t_ids))
        y_true = list(map(lambda x: pred_dict[x][1],t_ids))
        print(sub)
        plot_roc(sub, y_true, y_pred, adjust_sen=False, plot=False)
    return





def get_age_by_id(ids, name='unknown', print_out=True):
    id2fn = dict()
    with open(os.path.join(os.getcwd(), 'experiments', 'HPC_nfd2012_dd' + '.pkl'), 'rb') as f:
        file_list_dict = pk.load(f)
    for k in file_list_dict.keys():
        for p in file_list_dict[k]:
            #             print(p)
            fn = p.split('/')[-1]
            idx = fn.split('_')[0]
            grp = fn.split('_')[1]
            sex = fn.split('_')[2]
            age = fn.split('_')[3]
            diag = fn.split('_')[4]
            med = fn.split('_')[5].split('.')[0]
            id2fn[int(idx)] = (int(idx), int(grp), int(sex) - 1, int(age), int(diag), int(med))
    ret_ls = [id2fn[x] for x in ids]
    df = pd.DataFrame(ret_ls, columns=['idx', 'grp', 'sex', 'age', 'diag', 'meds'])

    #     df= df[df.age>=55]
    if print_out:
        print(
            '{}: mean age:{:4.2f}+-{:4.2f}, male:{:4.2f}, diagnosis:{:4.2f}+-{:4.2f}, medication:{:4.2f}+-{:4.2f}, tot:{}'.format(
                name,

                df.age.mean(),
                df.age.std(),
                df.sex.sum() / df.shape[0],
                df.diag.mean(),
                df.diag.std(),
                df.meds.mean(),
                df.meds.std(),
                df.shape[0]))
    return df


def get_description(df):
    for p, t, name in [(1, 1, 'TP'), (1, 0, 'FP'), (0, 0, 'TN'), (0, 1, 'FN')]:
        dfa = df[(df.yp_class == p) & (df.y_true == t)]
        print(
            '{}: mean age:{:4.2f}+-{:4.2f}, male:{:4.3f}, diagnosis:{:4.2f}+-{:4.2f}, medication:{:4.2f}+-{:4.2f}, tot:{}'.format(
                name,
                dfa.age.mean(),
                dfa.age.std(),
                dfa.sex.sum() / dfa.shape[0],
                dfa.diag.mean(),
                dfa.diag.std(),
                dfa.meds.mean(),
                dfa.meds.std(),
                dfa.shape[0]))
    return


def get_age_by_testid(test_ids, y_pred, y_true, name='unknown'):

    id2fn = dict()
    with open(os.path.join(os.getcwd(), 'experiments', 'HPC_nfd2012_dd' + '.pkl'), 'rb') as f:
        file_list_dict = pk.load(f)
    for k in file_list_dict.keys():
        for p in file_list_dict[k]:
            #             print(p)
            fn = p.split('/')[-1]
            idx = fn.split('_')[0]
            grp = fn.split('_')[1]
            sex = fn.split('_')[2]
            age = fn.split('_')[3]
            diag = fn.split('_')[4]
            med = fn.split('_')[5].split('.')[0]
            id2fn[int(idx)] = [int(idx), int(grp), int(sex) - 1, int(age), int(diag), int(med)]
    ret_ls = [id2fn[x] + [y_pred[c]] + [int(y_true[c])] for c, x in enumerate(test_ids)]
    df = pd.DataFrame(ret_ls, columns=['idx', 'grp', 'sex', 'age', 'diag', 'meds', 'y_pred', 'y_true'])
    #     all age
    df1 = df.copy()
    t, _, _, _ = plot_roc('all', df1.y_true.tolist(), df1.y_pred.tolist(), plot=False)
    print('all age:t ={:4.2f}'.format(t))
    df1['yp_class'] = np.asarray([1 if x >= t else 0 for x in df1.y_pred.tolist()])
    get_description(df1)
    #     >=55
    df1 = df[df.age >= 55].copy()
    t, _, _, _ = plot_roc('above55', df1.y_true.tolist(), df1.y_pred.tolist(), plot=False)
    print('age>=55:t ={:4.2f}'.format(t))
    df1.loc[:, 'yp_class'] = np.asarray([1 if x >= t else 0 for x in df1.y_pred.tolist()])
    get_description(df1)
    #     <55
    df1 = df[df.age < 55].copy()
    t, _, _, _ = plot_roc('below55', df1.y_true.tolist(), df1.y_pred.tolist(), plot=False)
    print('age<55:t ={:4.2f}'.format(t))
    df1.loc[:, 'yp_class'] = np.asarray([1 if x >= t else 0 for x in df1.y_pred.tolist()])
    get_description(df1)
    return


def get_age_by_testid_ret(test_ids, y_pred, y_true, name='unknown'):

    id2fn = dict()
    with open(os.path.join(os.getcwd(), 'experiments', 'HPC_nfd2012_dd' + '.pkl'), 'rb') as f:
        file_list_dict = pk.load(f)
    for k in file_list_dict.keys():
        for p in file_list_dict[k]:
            #             print(p)
            fn = p.split('/')[-1]
            idx = fn.split('_')[0]
            grp = fn.split('_')[1]
            sex = fn.split('_')[2]
            age = fn.split('_')[3]
            diag = fn.split('_')[4]
            med = fn.split('_')[5].split('.')[0]
            id2fn[int(idx)] = [int(idx), int(grp), int(sex) - 1, int(age), int(diag), int(med)]
    ret_ls = [id2fn[x] + [y_pred[c]] + [int(y_true[c])] for c, x in enumerate(test_ids)]
    df = pd.DataFrame(ret_ls, columns=['idx', 'grp', 'sex', 'age', 'diag', 'meds', 'y_pred', 'y_true'])
    #     all age
    df1 = df.copy()
    t, _, _, _ = plot_roc('all', df1.y_true.tolist(), df1.y_pred.tolist(), plot=False, print_out=False)
    print('all age:t ={:4.2f}'.format(t))
    df1['yp_class'] = np.asarray([1 if x >= t else 0 for x in df1.y_pred.tolist()])
    l1 = get_description_ret(df1)
    #     >=55
    df1 = df[df.age >= 55].copy()
    t, _, _, _ = plot_roc('above55', df1.y_true.tolist(), df1.y_pred.tolist(), plot=False, print_out=False)
    print('age>=55:t ={:4.2f}'.format(t))
    df1.loc[:, 'yp_class'] = np.asarray([1 if x >= t else 0 for x in df1.y_pred.tolist()])
    l2 = get_description_ret(df1)
    #     <55
    df1 = df[df.age < 55].copy()
    t, _, _, _ = plot_roc('below55', df1.y_true.tolist(), df1.y_pred.tolist(), plot=False, print_out=False)
    print('age<55:t ={:4.2f}'.format(t))
    df1.loc[:, 'yp_class'] = np.asarray([1 if x >= t else 0 for x in df1.y_pred.tolist()])
    l3 = get_description_ret(df1)
    return {'all':l1,'above':l2,'below':l3}

def get_description_ret(df):
    l = {'TP':[],'FP':[],'TN':[],'FN':[]}
    for p, t, name in [(1, 1, 'TP'), (1, 0, 'FP'), (0, 0, 'TN'), (0, 1, 'FN')]:
        dfa = df[(df.yp_class == p) & (df.y_true == t)]
        l[name]=dfa.age.tolist()
        print(
            '{}: mean age:{:4.2f}+-{:4.2f}, male:{:4.2f}, diagnosis:{:4.2f}+-{:4.2f}, medication:{:4.2f}+-{:4.2f}, tot:{}'.format(
                name,
                dfa.age.mean(),
                dfa.age.std(),
                dfa.sex.sum() / dfa.shape[0],
                dfa.diag.mean(),
                dfa.diag.std(),
                dfa.meds.mean(),
                dfa.meds.std(),
                dfa.shape[0]))
    return l


def get_age_by_testid_all_cancer(test_ids, y_pred, y_true, cancer_id, cutoff=55, cutoff_b=90):

    id2fn = dict()
    with open(os.path.join(os.getcwd(), 'experiments', 'HPC_nfd2012_'+str(cancer_id) + '_0.pkl'), 'rb') as f:
        file_list_dict = pk.load(f)
    for k in file_list_dict.keys():
        for p in file_list_dict[k]:
            #             print(p)
            fn = p.split('/')[-1]
            idx = fn.split('_')[0]
            grp = fn.split('_')[1]
            sex = fn.split('_')[2]
            age = fn.split('_')[3]
            diag = fn.split('_')[4]
            med = fn.split('_')[5].split('.')[0]
            id2fn[int(idx)] = [int(idx), int(grp), int(sex) - 1, int(age), int(diag), int(med)]
    ret_ls = [id2fn[x] + [y_pred[c]] + [int(y_true[c])] for c, x in enumerate(test_ids)]
    df = pd.DataFrame(ret_ls, columns=['idx', 'grp', 'sex', 'age', 'diag', 'meds', 'y_pred', 'y_true'])
    #     all age
    df1 = df.copy()
    t, _, _, _ = plot_roc('all', df1.y_true.tolist(), df1.y_pred.tolist(), plot=False)
    print('all age:t ={:4.2f}'.format(t))
    df1['yp_class'] = np.asarray([1 if x >= t else 0 for x in df1.y_pred.tolist()])
    l1 = get_description_ret(df1)
    #     >=55
    df1 = df[(df.age >= cutoff)&(df.age <= cutoff_b)].copy()
    t, _, _, _ = plot_roc('above55', df1.y_true.tolist(), df1.y_pred.tolist(), plot=False)
    print('age:{:d}-{:d}:t ={:4.2f}'.format(cutoff,cutoff_b,t))
    df1.loc[:, 'yp_class'] = np.asarray([1 if x >= t else 0 for x in df1.y_pred.tolist()])
    l2 = get_description_ret(df1)
    #     <55
    df1 = df[df.age < cutoff].copy()
    t, _, _, _ = plot_roc('below55', df1.y_true.tolist(), df1.y_pred.tolist(), plot=False)
    print('age<{:d}:t ={:4.2f}'.format(cutoff,t))
    df1.loc[:, 'yp_class'] = np.asarray([1 if x >= t else 0 for x in df1.y_pred.tolist()])
    l3 = get_description_ret(df1)
    return {'all':l1,'above':l2,'below':l3}