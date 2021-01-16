#%%
PATH = './DATA'
train_prob = pd.read_csv(PATH+'/train_problem_data.csv')
train_prob['time'] = pd.to_datetime(train_prob.time,format= '%Y%m%d%H%M%S')
problem = np.zeros(15000)
problem[train_prob.user_id.unique()-10000] = 1 
train_y = pd.DataFrame({'uid':range(10000,25000),'problem':problem})

diff0=[]
diff1=[]
diff5=[]
daily_error=[]
for file_name in tqdm(train_file_list):
    i=10015
    file_name=train_file_list[i-10000]
    with open(file_name,'rb') as f:
        temp_df = pickle.load(f)
        temp_df['time_diff']=temp_df.time.diff().apply(lambda x: x.total_seconds())
        t_df = temp_df.set_index('time')
        daily_error.append(t_df.groupby(t_df.index.floor('D')).user_id.count().describe().to_dict())
        d0 = np.sum(temp_df.time_diff==0)
        d1 = np.sum(temp_df.time_diff<=1)
        d5 = np.sum(temp_df.time_diff<=5)
        diff0.append(d0)
        diff1.append(d1-d0)
        diff5.append(d5-d1)
        
train_y['diff0'] = diff0
train_y['diff1'] = diff1
train_y['diff5'] = diff5
train_y['t_diff0'] = train_y.diff0>=20
train_y['t_diff1'] = train_y.diff1>=20
train_y['t_diff5'] = train_y.diff5>=20

for i in [100,200,300,400,500]:
    train_y['t_nfd'] = train_y.num_fast_diff>=i
    train_y['t_diff0'] = train_y.diff0>=i
    train_y['t_diff1'] = train_y.diff1>=i
    train_y['t_diff5'] = train_y.diff5>=i
    print(i,train_y.corr().loc['problem'])
train_y.describe()
train_y['diff_over_50']
# %%
