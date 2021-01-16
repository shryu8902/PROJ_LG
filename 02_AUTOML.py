#%%
import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from tqdm import tqdm
import gc
import random
import pickle
import h2o
from h2o.automl import H2OAutoML

#%%
h2o.init(nthreads=8)

#%%
if os.path.isfile('./DATA/train_x_ver2.pkl'):
    with open('./DATA/train_x_ver2.pkl','rb') as f:
        train_x = pickle.load(f)
else :
    with open('./DATA/train_x_ver2.pkl','wb') as f:
        pickle.dump(train_x,f)

if os.path.isfile('./DATA/test_x_ver2.pkl'):
    with open('./DATA/test_x_ver2.pkl','rb') as f:
        
        test_x = pickle.load(f)
else :
    with open('./DATA/test_x_ver2.pkl','wb') as f:
        pickle.dump(test_x,f)
#%%
PATH = './DATA'
train_prob = pd.read_csv(PATH+'/train_problem_data.csv')
problem = np.zeros(15000)
problem[train_prob.user_id.unique()-10000] = 1 


train_y = problem
        
#%%
info_columns = ['type1','code1','type2','code2','type3','code3','type4','code4','type5','code5','model_nm','fwver']

# train_y_pd = pd.Series(train_y,name='Y')
train = pd.concat([train_x, train_y.problem],axis=1)
h2o_train = h2o.H2OFrame(train)
h2o_train['problem']=h2o_train['problem'].asfactor()
for i in info_columns:
    h2o_train[i]=h2o_train[i].asfactor()

test_y_pd = pd.Series(np.zeros(len(test_x)),name='Y')
test = pd.concat([test_x,test_y_pd],axis=1)
h2o_test = h2o.H2OFrame(test)
h2o_test['problem']=h2o_test['problem'].asfactor()
for i in info_columns:
    h2o_test[i]=h2o_test[i].asfactor()

predictors = list(h2o_train.columns)
predictors.remove('problem')
predictors.remove('uid')
estimator='problem'
#%%
aml = H2OAutoML(max_models = 5, max_runtime_secs=10, seed=1,nfolds=5,balance_classes=True)
aml.train(x=predictors, y=estimator, training_frame=h2o_train)

lb = h2o.automl.get_leaderboard(aml, extra_columns = 'ALL')
preds = aml.leader.predict(h2o_test)
#%%
pred = pd.DataFrame({'user_id':test_x.uid,'prob':np.array(preds.as_data_frame().p1)})
pred['user_id']=pred['user_id'].astype('int64')
sample_submission = pd.read_csv(PATH+'/sample_submission.csv').set_index('user_id',drop=False)
for i in tqdm(range(len(pred))):
    uid,prob = pred.iloc[i]
    sample_submission.at[uid,'problem']= prob

sample_submission.to_csv("./dacon_baseline_h2o.csv", index = False)


#%%