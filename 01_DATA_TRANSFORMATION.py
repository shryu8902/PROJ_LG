#%%
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import random
import lightgbm as lgb
import re
from sklearn.metrics import *
from sklearn.model_selection import KFold,StratifiedKFold
import warnings
import pickle
import os

warnings.filterwarnings(action='ignore')

# 필요한 함수 정의
def make_datetime(x):
    # string 타입의 Time column을 datetime 타입으로 변경
    x     = str(x)
    year  = int(x[:4])
    month = int(x[4:6])
    day   = int(x[6:8])
    hour  = int(x[8:10])
    #mim  = int(x[10:12])
    #sec  = int(x[12:])
    return dt.datetime(year, month, day, hour)

def string2num(x):
    # (,)( )과 같은 불필요한 데이터 정제
    x = re.sub(r"[^0-9]+", '', str(x))
    if x =='':
        return 0
    else:
        return int(x)


PATH = './DATA'

#%%
train_err  = pd.read_csv(PATH+'/train_err_data.csv')
train_err['time'] = pd.to_datetime(train_err.time,format= '%Y%m%d%H%M%S')
test_err = pd.read_csv(PATH+'/test_err_data.csv')
test_err['time'] = pd.to_datetime(test_err.time,format= '%Y%m%d%H%M%S')

display(train_err.head())
train_user_id_max = 24999
train_user_id_min = 10000
train_user_number = 15000

print(np.sort(train_err.errtype.unique()))

# #%% ID 별 개별 에러 정보 저장 / 트레이닝
# id_list = train_err.user_id.unique()

# for person_idx in tqdm(id_list):
#     temp = train_err[train_err.user_id==person_idx]
#     with open('./DATA/train_err/{}.pkl'.format(person_idx),'wb') as f:
#         pickle.dump(temp,f)

# #%% ID 별 개별 에러 정보 저장 / 트레이닝
# test_err  = pd.read_csv(PATH+'/test_err_data.csv')
# test_err['time'] = pd.to_datetime(test_err.time,format= '%Y%m%d%H%M%S')
# id_list2 = test_err.user_id.unique()

# for person_idx in tqdm(id_list2):
#     temp = test_err[test_err.user_id==person_idx]
#     with open('./DATA/test_err/{}.pkl'.format(person_idx),'wb') as f:
#         pickle.dump(temp,f)


#%%
import glob
        
train_file_list = glob.glob('./DATA/train_err/*.pkl')
train_file_list.sort()

err_counts_list = []
err_info_list = []
daily_error=[]
hourly_error=[]
minute_error=[]
error_diff=[]
for file_name in tqdm(train_file_list):
    with open(file_name,'rb') as f:
        temp_df = pickle.load(f)
        temp_df['time_diff']=temp_df.time.diff().apply(lambda x: x.total_seconds())
        t_df = temp_df.set_index('time')
        daily_error.append(t_df.groupby(t_df.index.floor('D')).user_id.count().describe().to_dict())
        hourly_error.append(t_df.groupby(t_df.index.floor('H')).user_id.count().describe().to_dict())
        minute_error.append(t_df.groupby(t_df.index.floor('T')).user_id.count().describe().to_dict())

        d0 = np.sum(temp_df.time_diff==0)
        d1 = np.sum(temp_df.time_diff<=1)
        d5 = np.sum(temp_df.time_diff<=5)
        error_diff.append({'diff0':d0,'diff1':d1-d0,'diff5':d5-d1})

        err_counts = np.zeros(42)
        err_counts_df = temp_df.errtype.value_counts()
        for i in err_counts_df.index:
            err_counts[i-1]=err_counts_df.loc[i]
        err_counts_list.append([str(temp_df.user_id.iloc[0])]+err_counts.tolist())
        err_info = ['NOVAL']*12
        err_modelnm = temp_df.model_nm.mode().iloc[0]
        err_fwver = temp_df.fwver.mode().iloc[0]
        err_info[10]=err_modelnm
        err_info[11]=err_fwver

        for i in range(min(5,len(err_counts_df))):
            temp_err_type = err_counts_df.index[i]
            temp_err_info = temp_df[temp_df.errtype==temp_err_type]
            err_info[2*i]= str(temp_err_info.errtype.mode().iloc[0])
            err_info[2*i+1] = temp_err_info.errcode.mode().iloc[0]
        err_info_list.append(err_info)
#%%
train_counts_df = pd.DataFrame.from_records(err_counts_list)
train_counts_df=train_counts_df.add_prefix('X')
train_counts_df.rename(columns = {'X0' : 'uid'}, inplace = True)

info_columns = ['type1','code1','type2','code2','type3','code3','type4','code4','type5','code5','model_nm','fwver']
train_info_df =pd.DataFrame.from_records(err_info_list,columns=info_columns)
train_info_df.fillna('',inplace=True)
for col in info_columns:
    train_info_df[col]=train_info_df[col].astype('category')

train_d_error_df = pd.DataFrame(daily_error)
train_d_error_df.fillna(0,inplace=True)
train_h_error_df = pd.DataFrame(hourly_error)
train_h_error_df.fillna(0,inplace=True)
train_h_error_df = train_h_error_df.add_prefix('H')
train_m_error_df = pd.DataFrame(minute_error)
train_m_error_df.fillna(0,inplace=True)
train_m_error_df = train_m_error_df.add_prefix('M')

train_error_diff_df = pd.DataFrame(error_diff)

train_x=pd.concat([train_counts_df,train_d_error_df,train_h_error_df,train_m_error_df,train_info_df,train_error_diff_df],axis=1)

# %%
PATH = './DATA'
train_prob = pd.read_csv(PATH+'/train_problem_data.csv')
train_prob['time'] = pd.to_datetime(train_prob.time,format= '%Y%m%d%H%M%S')
problem = np.zeros(15000)
problem[train_prob.user_id.unique()-10000] = 1 
train_y = pd.DataFrame({'uid':range(10000,25000),'problem':problem})
train_y['smooth']=train_y['problem']*0.6+0.2

#%%
test_file_list = glob.glob('./DATA/test_err/*.pkl')
test_file_list.sort()

err_counts_list = []
err_info_list = []
daily_error=[]
hourly_error=[]
minute_error=[]
error_diff=[]

for file_name in tqdm(test_file_list):
    with open(file_name,'rb') as f:
        temp_df = pickle.load(f)
        temp_df['time_diff']=temp_df.time.diff().apply(lambda x: x.total_seconds())
        t_df = temp_df.set_index('time')
        daily_error.append(t_df.groupby(t_df.index.floor('D')).user_id.count().describe().to_dict())
        hourly_error.append(t_df.groupby(t_df.index.floor('H')).user_id.count().describe().to_dict())
        minute_error.append(t_df.groupby(t_df.index.floor('T')).user_id.count().describe().to_dict())

        d0 = np.sum(temp_df.time_diff==0)
        d1 = np.sum(temp_df.time_diff<=1)
        d5 = np.sum(temp_df.time_diff<=5)
        error_diff.append({'diff0':d0,'diff1':d1-d0,'diff5':d5-d1})

        err_counts = np.zeros(42)
        err_counts_df = temp_df.errtype.value_counts()
        for i in err_counts_df.index:
            err_counts[i-1]=err_counts_df.loc[i]
        err_counts_list.append([str(temp_df.user_id.iloc[0])]+err_counts.tolist())
        err_info = ['None']*12
        err_modelnm = temp_df.model_nm.mode().iloc[0]
        err_fwver = temp_df.fwver.mode().iloc[0]
        err_info[10]=err_modelnm
        err_info[11]=err_fwver

        for i in range(min(5,len(err_counts_df))):
            temp_err_type = err_counts_df.index[i]
            temp_err_info = temp_df[temp_df.errtype==temp_err_type]
            err_info[2*i]= str(temp_err_info.errtype.mode().iloc[0])
            err_info[2*i+1] = temp_err_info.errcode.mode().iloc[0]
        err_info_list.append(err_info)
#%%
test_counts_df = pd.DataFrame.from_records(err_counts_list)
test_counts_df=test_counts_df.add_prefix('X')
test_counts_df.rename(columns = {'X0' : 'uid'}, inplace = True)

info_columns = ['type1','code1','type2','code2','type3','code3','type4','code4','type5','code5','model_nm','fwver']
test_info_df =pd.DataFrame.from_records(err_info_list,columns=info_columns)
test_info_df.fillna('',inplace=True)
for col in info_columns:
    test_info_df[col]=test_info_df[col].astype('category')

test_d_error_df = pd.DataFrame(daily_error)
test_d_error_df.fillna(0,inplace=True)
test_error_diff_df = pd.DataFrame(error_diff)
test_h_error_df = pd.DataFrame(hourly_error)
test_h_error_df.fillna(0,inplace=True)
test_h_error_df = test_h_error_df.add_prefix('H')
test_m_error_df = pd.DataFrame(minute_error)
test_m_error_df.fillna(0,inplace=True)
test_m_error_df = test_m_error_df.add_prefix('M')

test_error_diff_df = pd.DataFrame(error_diff)

test_x=pd.concat([test_counts_df,test_d_error_df,test_h_error_df,test_m_error_df,test_info_df,test_error_diff_df],axis=1)
#%% load and save
if os.path.isfile('./DATA/train_x_ver3.pkl'):
    with open('./DATA/train_x_ver3.pkl','rb') as f:
        train_x = pickle.load(f)
else :
    with open('./DATA/train_x_ver3.pkl','wb') as f:
        pickle.dump(train_x,f)

if os.path.isfile('./DATA/test_x_ver3.pkl'):
    with open('./DATA/test_x_ver3.pkl','rb') as f:
        
        test_x = pickle.load(f)
else :
    with open('./DATA/test_x_ver3.pkl','wb') as f:
        pickle.dump(test_x,f)


#%%
# Train
#-------------------------------------------------------------------------------------
# validation auc score를 확인하기 위해 정의
def f_pr_auc(probas_pred, y_true):
    labels=y_true.get_label()
    p, r, _ = precision_recall_curve(labels, probas_pred)
    score=auc(r,p) 
    return "pr_auc", score, True
#-------------------------------------------------------------------------------------
models     = []
recalls    = []
precisions = []
auc_scores   = []
threshold = 0.5
# 파라미터 설정
params =      {
                'boosting_type' : 'dart',
                'objective'     : 'binary',
                'metric'        : 'auc',
                'seed': 1015,
                'num_leaves' : 80,
                'max_depth':-1,
                'scale_pos_weight':3
                }
#%%
#-------------------------------------------------------------------------------------
# 5 Kfold cross validation
col_list=['uid', 'min', '25%', '50%','75%', 
    'Hmin', 'H25%', 'H50%','H75%',
    'Mmin', 'M25%', 'M50%', 'M75%',
    'type1','code1','type2', 'code2', 'type3', 'code3', 'type4', 'code4',
       'type5', 'code5', 'diff0','diff1','diff5']+train_ecumsum.columns.to_list()

use_col_list=['X11', 'X14', 'X15', 'X16', 'X17', 'X18', 'X20', 'X30', 'X31', 'X34',
       'X35', 'X42', 'count', 'mean', 'std', 'max', 'Hcount', 'Hmean', 'Hstd',
       'Hmax', 'Mcount', 'model_nm', 'diff5']
train_x_t = train_x2.drop(columns=col_list)
test_x_t = test_x2.drop(columns=col_list)
#%%
# train_x_t = train_x[['X'+str(x) for x in range(1,43)]+['model_nm','fwver','count','mean','std','min','25%','50%','75%','max']]
k_fold = KFold(n_splits=5, shuffle=True, random_state=0)
for train_idx, val_idx in k_fold.split(train_x):
    print(train_idx.shape)
    # split train, validation set
    X = train_x_t.iloc[train_idx]
    y = train_y.problem.iloc[train_idx]
    valid_x = train_x_t.iloc[val_idx]
    valid_y = train_y.problem.iloc[val_idx]

    d_train= lgb.Dataset(X, y)
    d_val  = lgb.Dataset(valid_x, valid_y)
    
    #run traning
    model = lgb.train(
                        params,
                        train_set       = d_train,
                        num_boost_round = 500,
                        valid_sets      = d_val,
                        feval           = f_pr_auc,
                        verbose_eval    = 20, 
                        early_stopping_rounds = 20
                        
                       )
    
    # cal valid prediction
    valid_prob = model.predict(valid_x)
    valid_pred = np.where(valid_prob > threshold, 1, 0)
    
    # cal scores
    recall    = recall_score(    valid_y, valid_pred)
    precision = precision_score( valid_y, valid_pred)
    auc_score = roc_auc_score(   valid_y, valid_prob)

    # append scores
    models.append(model)
    recalls.append(recall)
    precisions.append(precision)
    auc_scores.append(auc_score)
    print('==========================================================')
#%%
    print(np.mean(auc_scores))

#%%
pred_y_list = []
for model in models:
    pred_y = model.predict(test_x)
    pred_y_list.append(pred_y.reshape(-1,1))
    
pred_ensemble = np.mean(pred_y_list, axis = 0)
pred = pd.DataFrame({'user_id':test_counts_df[0],'prob':pred_ensemble.reshape(-1)})
pred['user_id']=pred['user_id'].astype('int64')
sample_submission = pd.read_csv(PATH+'/sample_submission.csv').set_index('user_id',drop=False)
for i in tqdm(range(len(pred))):
    uid,prob = pred.iloc[i]
    sample_submission.set_value(uid,'problem',prob)

sample_submission.to_csv("./dacon_baseline3.csv", index = False)
#%% catboost grid search

model = CatBoostClassifier(loss_function='CrossEntropy',eval_metric='AUC')
train_data = Pool(data=train_x_t,label=train_y.smooth,
                cat_features=['model_nm','fwver'])
grid = {'learning_rate': [0.1],
        'depth': [6, 8, 10],
        'l2_leaf_reg': [7, 9, 12, 15]}
grid_search_result = model.grid_search(grid, 
                                       X=train_data)

#%% catboost
from catboost import CatBoostClassifier, Pool

models     = []
recalls    = []
precisions = []
auc_scores   = []
threshold = 0.5

k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
for train_idx, val_idx in k_fold.split(train_x_t,train_y.problem):
    print(train_idx.shape)
    # split train, validation set
    X = train_x_t.iloc[train_idx]
    y = train_y.smooth[train_idx]
    valid_x = train_x_t.iloc[val_idx]
    valid_y = train_y.problem[val_idx]
   
    #run traning
    for seed in range(3):
        model = CatBoostClassifier(iterations=1500,
                            loss_function='CrossEntropy',
                            random_seed=seed,
                            eval_metric='AUC',
                            depth=10,
                            l2_leaf_reg=20,
                            learning_rate=0.03,
                            # auto_class_weights='Balanced',
                            verbose=True)

        # train the model   ## cat_features=info_columns
        model.fit(X, y,cat_features =['model_nm','fwver'],eval_set=(valid_x,valid_y),early_stopping_rounds=100)
        # model.fit(X, y,eval_set=(valid_x,valid_y),early_stopping_rounds=50)

        # cal valid prediction
        valid_prob = model.predict_proba(valid_x)
        valid_pred = model.predict(valid_x)

        # valid_pred = np.where(valid_prob > threshold, 1, 0)
        # valid_pred=[0 if x[0]>x[1] else 1 for x in valid_prob]
        valid_prob_1 = [x[1] for x in valid_prob]
        # cal scores
        # recall    = recall_score(    valid_y, valid_pred)
        # precision = precision_score( valid_y, valid_pred)
        # auc_score = roc_auc_score(   valid_y, valid_prob_1)

        recall = recall_score(train_y.problem[val_idx], valid_pred)
        precision = precision_score(train_y.problem[val_idx], valid_pred)
        auc_score = roc_auc_score(   train_y.problem[val_idx], valid_prob_1)

        # append scores
        models.append(model)
        recalls.append(recall)
        precisions.append(precision)
        auc_scores.append(auc_score)
        print('==========================================================')
#%%
select_model=[]
for i in range(len(models)):
    print(models[i].best_score_['validation'])
    if models[i].best_score_['validation']['AUC']>=0.81:
        select_model.append(i)
pred_y_list = []
for i in select_model:
    model=models[i]
    pred_y = model.predict_proba(test_x_t)
    pred_y_= np.array([x[1] for x in pred_y])
    pred_y_list.append(pred_y_.reshape(-1,1))
    
pred_ensemble = np.mean(pred_y_list, axis = 0)
pred = pd.DataFrame({'user_id':test_x.uid,'prob':pred_ensemble.reshape(-1)})
pred['user_id']=pred['user_id'].astype('int64')
sample_submission = pd.read_csv(PATH+'/sample_submission.csv').set_index('user_id',drop=False)
for i in tqdm(range(len(pred))):
    uid,prob = pred.iloc[i]
    # sample_submission.loc.set_value(uid,'problem',prob)
    sample_submission.loc[uid,'problem']=prob
sample_submission.to_csv("./dacon_baseline_cat_cross_en2.csv", index = False)
#%%

model = CatBoostClassifier(iterations=1000,
                           depth=10,
                           learning_rate=1,
                           loss_function='Logloss',
                           verbose=True)
# train the model
model.fit(train_x, train_y,cat_features =info_columns)
# make the prediction using the resulting model
preds_class = model.predict(test_data)
preds_proba = model.predict_proba(test_data)
print("class = ", preds_class)
print("proba = ", preds_proba)