#%%
import glob
import pandas as ps
import numpy as np
from tqdm import tqdm

class data_gen():
    def __init__(self,train_file_list,test_file_list):
        train_file_list.sort()
        test_file_list.sort()
        self.train_file_list = train_file_list
        self.test_file_list = test_file_list
        self.label_data = pd.read_csv('./DATA/train_problem_data.csv')
        self.label_data['time'] = pd.to_datetime(label_data.time,format= '%Y%m%d%H%M%S') 
        
        self.train_X = [] #self
        self.train_Y = [] #self
        self.test_X = []#self
        print('start reading train data')
        for file_name in tqdm(self.train_file_list): #self
            with open(file_name,'rb') as f:
                temp_df = pickle.load(f)
            current_uid = temp_df.user_id.iloc[0]
            t_df = temp_df.set_index('time')
            grouped_t_df=t_df.groupby([t_df.index.floor('D'),t_df.errtype]).user_id.count()

            first_date = t_df.index.min().floor('D')
            last_date = t_df.index.max().floor('D')
            date_seq = pd.date_range(first_date,last_date,freq='D')

            temp_seq = np.zeros((len(date_seq),42))
            for i, id_date in enumerate(date_seq):
                if id_date in t_df.index.floor('D').to_list():
                    temp_seq[i,grouped_t_df[id_date].index-1]=grouped_t_df[id_date]

            ans_seq = np.zeros((len(date_seq)))
            if current_uid in label_data.user_id.to_list():
                report_date = label_data[label_data.user_id==current_uid].time.min().floor('D')
                ans_seq[date_seq>=report_date]=1    

            self.train_X.append(temp_seq)#self
            self.train_Y.append(ans_seq)#self
        print('start reading test data')

        for file_name in tqdm(self.test_file_list):
            with open(file_name,'rb') as f:
                temp_df = pickle.load(f)
            current_uid = temp_df.user_id.iloc[0]
            t_df = temp_df.set_index('time')
            grouped_t_df=t_df.groupby([t_df.index.floor('D'),t_df.errtype]).user_id.count()

            first_date = t_df.index.min().floor('D')
            last_date = t_df.index.max().floor('D')
            date_seq = pd.date_range(first_date,last_date,freq='D')

            temp_seq = np.zeros((len(date_seq),42))
            for i, id_date in enumerate(date_seq):
                temp_seq[i,grouped_t_df[id_date].index-1]=grouped_t_df[id_date]

            self.test_X.append(temp_seq)
    def train_pad(self,target_len):
        for i in range(len(self.train_X)):
            self.train_X[i]=np.pad(self.train_X[i],(0,target_len-len(self.train_X[i]),(0,0)),'edge')    
            self.train_Y[i]=np.pad(self.train_Y[i],(0,target_len-len(self.train_Y[i]),(0,0)),'edge')

    def test_pad(self,target_len):
        for i in range(len(self.test_X)):
            self.test_X[i]=np.pad(self.test_X[i],(0,target_len-len(self.test_X[i]),(0,0)),'edge')

#%%
train_file_list = glob.glob('./DATA/train_err/*.pkl')
test_file_list = glob.glob('./DATA/test_err/*.pkl')
DATA = data_gen(train_file_list,test_file_list)
#%%
varlist = ['X'+str(x) for x in range(1,43)] 

train_file_list = glob.glob('./DATA/train_err/*.pkl')
train_file_list.sort()
# var18_info=[]
ecumsum = []
e18cumsum = []
for file_name in tqdm(train_file_list):
    with open(file_name,'rb') as f:
        temp_df = pickle.load(f)
