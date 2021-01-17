#%%
import glob
varlist = ['X'+str(x) for x in range(1,43)] 

train_file_list = glob.glob('./DATA/train_err/*.pkl')
train_file_list.sort()
# var18_info=[]
ecumsum = []
e18cumsum = []
for file_name in tqdm(train_file_list):
    with open(file_name,'rb') as f:
        temp_df = pickle.load(f)
        # temp_df['time_diff']=temp_df.time.diff().apply(lambda x: x.total_seconds())
        day = pd.Series(pd.date_range("2020/11/2", freq="D", periods=45),name='Day')
        temp_cumsum =[]
        temp_18cumsum=[]
        day_ind = ['d'+str(j) for j in range(45)]
        day_ind2 = ['xd'+str(j) for j in range(45)]
        for i in day:
            temp_cumsum.append(np.sum(temp_df.time<=i))
            temp_18cumsum.append(np.sum((temp_df.errtype==18)&(temp_df.time<=i)))
        cumsum=pd.DataFrame({'cumsum':temp_cumsum},index=day_ind)['cumsum']to_dict()
        x18cumsum=pd.DataFrame({'x18cumsum':temp_18cumsum},index=day_ind2)['x18cumsum'].to_dict()
        ecumsum.append(cumsum)
        e18cumsum.append(x18cumsum)
        # t_df = temp_df.set_index('time')
        # t_df2 = t_df[t_df.errtype==18]
        # var18_info.append(t_df2.groupby(t_df2.index.floor('D')).user_id.count().describe().to_dict())

#%%
ecumsum=[x['cumsum'] for x in ecumsum]
e18cumsum=[x['x18cumsum'] for x in e18cumsum]

train_ecumsum = pd.DataFrame(ecumsum)
train_e18cumsum = pd.DataFrame(e18cumsum)

train_var18_df = pd.DataFrame(var18_info)
train_var18_df.fillna(0,inplace=True)
train_var18_df = train_var18_df.add_prefix('X18')

train_var_ratio_df = train_x[varlist].divide(train_x[varlist].sum(axis=1),axis=0)
train_var_ratio_df.fillna(0,inplace=True)
train_var_ratio_df = train_var_ratio_df.add_prefix('R_')
# train_x2 = pd.concat([train_x,train_var18_df,train_var_ratio_df],axis=1)
train_x2 = pd.concat([train_x,train_ecumsum,train_e18cumsum],axis=1)

#%%
test_file_list = glob.glob('./DATA/test_err/*.pkl')
test_file_list.sort()
# var18_info=[]
ecumsum = []
e18cumsum = []

for file_name in tqdm(test_file_list):
    with open(file_name,'rb') as f:
        temp_df = pickle.load(f)
        # temp_df['time_diff']=temp_df.time.diff().apply(lambda x: x.total_seconds())
        day = pd.Series(pd.date_range("2020/11/2", freq="D", periods=45),name='Day')
        temp_cumsum =[]
        temp_18cumsum=[]
        day_ind = ['d'+str(j) for j in range(45)]
        day_ind2 = ['xd'+str(j) for j in range(45)]
        for i in day:
            temp_cumsum.append(np.sum(temp_df.time<=i))
            temp_18cumsum.append(np.sum((temp_df.errtype==18)&(temp_df.time<=i)))
        cumsum=pd.DataFrame({'cumsum':temp_cumsum},index=day_ind)['cumsum'].to_dict()
        x18cumsum=pd.DataFrame({'x18cumsum':temp_18cumsum},index=day_ind2)['x18cumsum'].to_dict()
        ecumsum.append(cumsum)
        e18cumsum.append(x18cumsum)
        
        # t_df = temp_df.set_index('time')
        # t_df2 = t_df[t_df.errtype==18]
        # var18_info.append(t_df2.groupby(t_df2.index.floor('D')).user_id.count().describe().to_dict())
#%%

test_ecumsum = pd.DataFrame(ecumsum)
test_e18cumsum = pd.DataFrame(e18cumsum)

test_var18_df = pd.DataFrame(var18_info)
test_var18_df.fillna(0,inplace=True)
test_var18_df = test_var18_df.add_prefix('X18')

test_var_ratio_df = test_x[varlist].divide(test_x[varlist].sum(axis=1),axis=0)
test_var_ratio_df.fillna(0,inplace=True)
test_var_ratio_df = test_var_ratio_df.add_prefix('R_')
# test_x2 = pd.concat([test_x,test_var18_df,test_var_ratio_df],axis=1)
test_x2 = pd.concat([test_x,test_ecumsum,test_e18cumsum],axis=1)
