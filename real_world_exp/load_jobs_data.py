import pandas as pd
from sklearn import preprocessing
import numpy as np
import random
from sklearn.preprocessing import StandardScaler

def sample_jobs(n_unc):
    df = pd.read_csv('Jobs_Lalonde_Data.csv.gz')
    
    df['outcome'] = df['RE78']
    
    #Center/scale data
    scaler = StandardScaler()
    scaler.fit(df[['Age', 'Education', 'outcome']].values)
    df[['Age', 'Education', 'outcome']] = scaler.transform(df[['Age', 'Education', 'outcome']])

    
    #Among the ones in the original experiment, sample more that are older -> covariate shift in age    
    n_rct = 297 + 425
    
    median_age = df['Age'].median() 
    ind_l = df[:n_rct]['Age'] < median_age
    ind_u = df[:n_rct]['Age'] >= median_age
    df_1 = df[:n_rct][ind_l].sample(int(95/100*n_unc))
    df_2 = df[:n_rct][ind_u].sample(int(5/100*n_unc))
    df_unc = pd.concat((df_1, df_2))
    
    df_not_unc = df[~df.isin(df_unc)].dropna()
    df_not_unc.shape
 
    #Only little confounding for the treated - we don't have many samples 
    ind_t = df_not_unc['Treatment'] == 1
    df_not_unc_t = df_not_unc[ind_t] 
    mean = df_not_unc_t['outcome'].mean() 
    std = df_not_unc_t['outcome'].std() 
    ind_t =  df_not_unc_t['outcome'] > mean+0.25*std
    
    df_not_unc_t_upper_half = df_not_unc_t[ind_t]  
    
    #Confounding for the treated - we have many samples
    ind_c = df_not_unc['Treatment'] == 0
    df_not_unc_c = df_not_unc[ind_c] 
    mean = df_not_unc_c['outcome'].mean() 
    std = df_not_unc_c['outcome'].std() 
    ind_c =  df_not_unc_c['outcome'] < mean-1.2*std
    
    df_not_unc_c_lower_half = df_not_unc_c[ind_c]  
    
    df_conf = pd.concat((df_not_unc_t_upper_half, df_not_unc_c_lower_half))
    
    #Unconfouded data
    x_unc = np.array(df_unc.values[:, :6], dtype = 'float64')
    t_unc = np.array(df_unc['Treatment'].values.reshape(-1,1), dtype = 'float64')
    y_unc = np.array(df_unc['outcome'].values.reshape(-1,1), dtype = 'float64')
    
    #Confounded data
    x_conf = np.array(df_conf.values[:, :6], dtype = 'float64')
    t_conf = np.array(df_conf['Treatment'].values.reshape(-1,1), dtype = 'float64')
    y_conf = np.array(df_conf['outcome'].values.reshape(-1,1), dtype = 'float64')
    
    #Test on ALL\Unc   
    df_test = df_not_unc
    
    x_test = np.array(df_test.values[:, :6], dtype = 'float64')
    t_test = np.array(df_test['Treatment'].values.reshape(-1,1), dtype = 'float64')
    y_test = np.array(df_test['outcome'].values.reshape(-1,1), dtype = 'float64')
    
    #all
    x_all = np.array(df[:n_rct].values[:, :6], dtype = 'float64')
    t_all = np.array(df[:n_rct]['Treatment'].values.reshape(-1,1), dtype = 'float64')
    y_all = np.array(df[:n_rct]['outcome'].values.reshape(-1,1), dtype = 'float64')

    return {'x_unc': x_unc, 't_unc': t_unc, 'y_unc': y_unc, 'x_conf': x_conf, 't_conf': t_conf, 'y_conf': y_conf, 'x_test': x_test, 't_test': t_test, 'y_test': y_test, 'x_all': x_all, 't_all': t_all, 'y_all': y_all}

