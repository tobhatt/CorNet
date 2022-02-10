import pandas as pd
from sklearn import preprocessing
import numpy as np
import random

def sample_rct(n_unc):
    df = pd.read_csv("actg175.csv", header=0, index_col=0)
    
    cd4_baseline = df['cd40']
    cd4_20 = df['cd420']
    
    outcome = (cd4_20 - cd4_baseline).values
    outcome_norm = preprocessing.scale(outcome)
    
    cov_cont = df[['age', 'wtkg', 'cd40', 'karnof', 'cd80']]
    cov_cont_norm = preprocessing.scale(cov_cont)
    
    cov_bin = df[['gender', 'homo', 'race', 'drugs', 'symptom', 'str2', 'hemo']]
    cov_bin_val = cov_bin.values 
    t = df[['arms']].values
    
    data = np.concatenate((cov_cont_norm, cov_bin_val, t.reshape(-1,1), outcome_norm.reshape(-1,1)), axis=1)
    data.shape
    
    #Only focus on one arm (0=zidovudine, 1=zidovudine and didanosine, 2=zidovudine and zalcitabine,3=didanosine)
    t_1 = 2
    t_0 = 0
    t_ind = (t == t_0) + (t == t_1)
    
    data_rct = data[t_ind.flatten()]
    #change treatment sign to 1
    data_rct[:,-2] = np.where(data_rct[:,-2] == 2, 1, 0)
    
    #All data
    x_all = data_rct[:,:-2]
    t_all = data_rct[:,-2].reshape(-1,1)
    y_all = data_rct[:,-1].reshape(-1,1)

    #UNC selection
    ind_unc = random.sample(range(x_all.shape[0]), n_unc)
    x_unc = x_all[ind_unc, ]
    t_unc = t_all[ind_unc, ].reshape(-1,1)
    y_unc = y_all[ind_unc, ].reshape(-1,1)
    
    x_not_unc = np.delete(x_all, ind_unc, axis = 0)
    t_not_unc = np.delete(t_all, ind_unc)
    y_not_unc = np.delete(y_all, ind_unc)
    
    #CONF selection - balanced gender - take all females and sample male s.t. ~ balanced
    #Among males, introduce confounding
    ind_f = (x_not_unc[:, 5] == 0)
    ind_m = (x_not_unc[:, 5] == 1)
    
    ind_m_t = (t_not_unc == 1) * ind_m
    mean = y_not_unc[ind_m_t].mean()
    std = y_not_unc[ind_m_t].std()
    ind_m_t_upper = y_not_unc[ind_m_t] > mean
    
    x_m_t_upper = x_not_unc[ind_m_t,:][ind_m_t_upper,:]
    t_m_t_upper = t_not_unc[ind_m_t][ind_m_t_upper]
    y_m_t_upper = y_not_unc[ind_m_t][ind_m_t_upper]

    ind_m_c = (t_not_unc == 0) * ind_m
    mean = y_not_unc[ind_m_c].mean()
    std = y_not_unc[ind_m_c].std()
    ind_m_c_lower = y_not_unc[ind_m_c] < mean

    x_m_c_lower = x_not_unc[ind_m_c,:][ind_m_c_lower,:]
    t_m_c_lower = t_not_unc[ind_m_c][ind_m_c_lower]
    y_m_c_lower = y_not_unc[ind_m_c][ind_m_c_lower]

    x_f = x_not_unc[ind_f,:]
    t_f = t_not_unc[ind_f]
    y_f = y_not_unc[ind_f]
    
    x_conf = np.concatenate((x_m_t_upper, x_m_c_lower, x_f))
    t_conf = np.concatenate((t_m_t_upper, t_m_c_lower, t_f)).reshape(-1,1)
    y_conf = np.concatenate((y_m_t_upper, y_m_c_lower, y_f)).reshape(-1,1)

    
    x_test = x_not_unc
    t_test = t_not_unc.reshape(-1,1)
    y_test = y_not_unc.reshape(-1,1)

    return {'x_unc': x_unc, 't_unc': t_unc, 'y_unc': y_unc, 'x_conf': x_conf, 't_conf': t_conf, 'y_conf': y_conf, 'x_test': x_test, 't_test': t_test, 'y_test': y_test, 'x_all': x_all, 't_all': t_all, 'y_all': y_all}

