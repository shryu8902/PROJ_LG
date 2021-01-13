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
h2o.init(nthreads=4,max_mem_size=4)

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
train_y_pd = pd.Series(train_y,name='Y')
train = pd.concat([train_x, train_y_pd],axis=1)
h2o_train = h2o.H2OFrame(train)
h2o_train['Y']=h2o_train['Y'].asfactor()

test_y_pd = pd.Series(np.zeros(len(test_x)),name='Y')
test = pd.concat([test_x,test_y_pd],axis=1)
h2o_test = h2o.H2OFrame(test)
h2o_test['Y']=h2o_test['Y'].asfactor()

predictors = list(h2o_train.columns)
predictors.remove('Y')
predictors.remove('uid')
estimator='Y'
#%%
aml = H2OAutoML(seed=1,nfolds=10)
aml.train(x=predictors, y=estimator, training_frame=h2o_train)

lb = h2o.automl.get_leaderboard(aml, extra_columns = 'ALL')
preds = aml.leader.predict(h2o_test)
#%%
pred = pd.DataFrame({'user_id':test_counts_df[0],'prob':pred_ensemble.reshape(-1)})
pred['user_id']=pred['user_id'].astype('int64')
sample_submission = pd.read_csv(PATH+'/sample_submission.csv').set_index('user_id',drop=False)
for i in tqdm(range(len(pred))):
    uid,prob = pred.iloc[i]
    sample_submission.set_value(uid,'problem',prob)

sample_submission.to_csv("./dacon_baseline_cat2.csv", index = False)


#%%