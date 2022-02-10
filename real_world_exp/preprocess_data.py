import csv
import pandas as pd
from sklearn.model_selection import train_test_split
import pyreadr
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
import numpy as np

def read_START_data(q_prime=0.1):
    result = pyreadr.read_r('STAR_Students.rdata')
    
    dat = result['x']
    df = dat
    
    #Drop nans
    treatment_indicator = df['g1classtype'].notna()
    df = df[treatment_indicator]
    
    #Only consider regular and small classes
    reg_size = df['g1classtype']=='REGULAR CLASS'
    small_size = df['g1classtype']=='SMALL CLASS'
    size_indicator = reg_size|small_size
    df = df[size_indicator]
     
    #Remove students with missing outcome 
    df = df[df['g1tlistss'].notna()]
    df.shape
    
    df = df[df['g1treadss'].notna()]
    df.shape
    
    df = df[df['g1tmathss'].notna()]
    df.shape
    
    #Treatment and outcome variables
    df['treatment'] = (df['g1classtype']=='SMALL CLASS').astype(int)
    df['outcome'] = df['g1tlistss'] + df['g1treadss'] + df['g1tmathss']
    df['rural'] = (df['g1surban'] == 'RURAL') | (df['g1surban'] == 'INNER CITY')
    
    #Only use covariates: gender, race, birth month, birthday, birth year, free lunch given or not, teacher id
    df_all = df[['outcome', 'treatment', 'gender', 'race', 'birthmonth', 'birthday', 'birthyear', 'g1freelunch', 'g1tchid', 'rural']]
    
    #Remove students with missing covariates
    for i in ['gender', 'race', 'birthmonth', 'birthday', 'birthyear', 'g1freelunch', 'g1tchid', 'rural']:
        df_all = df_all[df_all[i].notna()]
        df = df[df[i].notna()]
    
    #Ordinal transformer
    for i in ['gender', 'race', 'birthmonth', 'g1freelunch', 'rural']:
        enc = OrdinalEncoder(dtype=int)
        x = np.array(df_all[i].values).reshape(-1,1)
        enc.fit(x)
        x_transform = enc.transform(x)
        df_all[i] = x_transform
    
    #Center/scale data
    scaler = StandardScaler()
    scaler.fit(df_all[['birthday', 'birthyear', 'g1tchid', 'outcome']].values)
    df_all[['birthday', 'birthyear', 'g1tchid', 'outcome']] = scaler.transform(df_all[['birthday', 'birthyear', 'g1tchid', 'outcome']])

    rct_indicator1 = (df_all['birthday'] < 0)*np.random.binomial(1, 0.5, df_all['birthday'].shape[0]) + (df_all['birthday'] >= 0)*np.random.binomial(1, 0.1, df_all['birthday'].shape[0])
    
    rct_indicator = (rct_indicator1)>0
    
    df_RCT = df_all[rct_indicator==1]
    df_OS = df_all[rct_indicator==0]

    indicator = df_OS['treatment'] == 0 
    df_OS_control = df_OS[indicator]
    
    #Introduce confounding
    indicator = df_OS['treatment'] == 1
    df_OS_treated = df_OS[indicator]
    mean = df_OS_treated['outcome'].mean() 
    std = df_OS_treated['outcome'].std() 
    indicator_treat =  df_OS_treated['outcome'] > mean+std
    

    indicator_control =  df_OS_control['outcome'] < mean-std
    
    df_OS_treated_upper_half = df_OS_treated[indicator_treat]
    
    df_OS_control_lower_half = df_OS_control[indicator_control]    
    
    df_unc = df_RCT
    df_conf = pd.concat((df_OS_control_lower_half,
                         df_OS_treated_upper_half))

    #Unconfouded data
    x_unc = np.array(df_unc.values[:, 2:], dtype = 'float64')
    t_unc = np.array(df_unc['treatment'].values.reshape(-1,1), dtype = 'float64')
    y_unc = np.array(df_unc['outcome'].values.reshape(-1,1), dtype = 'float64')
    
    #Confounded data   
    x_conf = np.array(df_conf.values[:, 2:], dtype = 'float64')
    t_conf = np.array(df_conf['treatment'].values.reshape(-1,1), dtype = 'float64')
    y_conf = np.array(df_conf['outcome'].values.reshape(-1,1), dtype = 'float64')
    
    #Test on ALL\Unc  
    df_test = df_all[~df_all.isin(df_unc)].dropna()
    
    x_test = np.array(df_test.values[:, 2:], dtype = 'float64')
    t_test = np.array(df_test['treatment'].values.reshape(-1,1), dtype = 'float64')
    y_test = np.array(df_test['outcome'].values.reshape(-1,1), dtype = 'float64')
    
    #all
    x_all = np.array(df_all.values[:, 2:], dtype = 'float64')
    t_all = np.array(df_all['treatment'].values.reshape(-1,1), dtype = 'float64')
    y_all = np.array(df_all['outcome'].values.reshape(-1,1), dtype = 'float64')
    
    
    return {'x_unc': x_unc, 't_unc': t_unc, 'y_unc': y_unc, 'x_conf': x_conf, 't_conf': t_conf, 'y_conf': y_conf, 'x_test': x_test, 't_test': t_test, 'y_test': y_test, 'x_all': x_all, 't_all': t_all, 'y_all': y_all}
        
