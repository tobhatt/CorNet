from IPython import get_ipython
get_ipython().magic('reset -sf')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path
import random

import matplotlib.pyplot as plt
from models import CorNet, CorNet_ADA, MTL, WeightNet
from TwoStepBaseline import TwoStepRF, TwoStepRidge, TwoStepTarNet, TwoStepRF_modi, TwoStepRidge_modi, TwoStepTarNet_modi
from sklearn.linear_model import LogisticRegression
import torch
from copy import deepcopy
import pickle


n_run=10
n_conf_size = [300, 400, 500, 600, 700]

cate_trans_reg_l, cate_trans_l, cate_conf_l, cate_unc_l = [], [], [], []
cate_mtl_l, cate_mtl_delta_1_l, cate_mtl_bal_l, cate_mtl_bal_delta_1_l = [], [], [], []
cate_base_ridge_l, cate_base_rf_l, cate_base_nn_l = [], [], []
cate_base_ridge_prop_l, cate_base_rf_prop_l, cate_base_nn_prop_l = [], [], []
cate_base_ridge_modi_l, cate_base_rf_modi_l, cate_base_nn_modi_l = [], [], []
cate_trans_delta_1_l, cate_trans_delta_2_l, cate_trans_out_l, cate_trans_out_1_l, cate_trans_out_2_l = [], [], [], [], []
cate_trans_bal_l, cate_trans_bal_out_1_l, cate_trans_bal_out_2_l = [], [], []
cate_avg_l = []
cate_weight_l = []
cate_test = []

params_l = []
params_avg_l = []
params_weight_l = []

#-----------------------------
# Load data
#-----------------------------

a_file = open("sim_dict_star.pkl", "rb")
sim_dict = pickle.load(a_file)

n_conf = sim_dict['x_conf'].shape[0]
ind_ = random.sample(range(n_conf), n_conf) 


for i in range(n_run):
    print('###################     Running algorithm for the ', i + 1, 'time.     ####################') 
    
    n_unc = 2*sim_dict['x_unc'].shape[1]
    
    #Unconfounded data
    x_unc = sim_dict['x_unc'][:n_unc,]
    t_unc = sim_dict['t_unc'][:n_unc,]
    y_unc = sim_dict['y_unc'][:n_unc,]
    
    #Confounded data
    x_conf_ = sim_dict['x_conf']
    t_conf_ = sim_dict['t_conf']
    y_conf_ = sim_dict['y_conf']
    
    #Mix conf data (not sorted treatment and control group)  
    x_conf, t_conf, y_conf = x_conf_[ind_,:], t_conf_[ind_,:], y_conf_[ind_,:]
    
    #Test data
    x_test = sim_dict['x_test']
    t_test = sim_dict['t_test']
    y_test = sim_dict['y_test']
    
    #All data
    x_all = sim_dict['x_all']
    t_all = sim_dict['t_all']
    y_all = sim_dict['y_all']    

    n_cov = x_conf.shape[1]
    n_hidden = 3
    d_hidden = 100
    n_out = 0
    d_out = d_hidden
    
    lr_train = 0.1
    lr_ret = 0.1
    n_epochs_feature = 500
    n_epochs_retrain = 200
    batch_size = x_test.shape[0]

    #Estimate propoensity score
    clf = LogisticRegression(random_state=0).fit(x_unc, t_unc.flatten())
    prop_score = clf.predict_proba(x_unc)[:,1].reshape(-1,1)

    #-----------------------------
    # Groud truth CATE
    #-----------------------------
    nn_rand = CorNet(n_cov, n_hidden, d_hidden, n_out, d_out)
    #Concatenate treatment and control
    x_a, y_a, t_a = torch.Tensor(x_all.copy()), torch.Tensor(y_all.copy()), torch.Tensor(t_all.copy())
    x_t = torch.Tensor(x_test)
    #Train both networks on confounded data (Step 1)
    nn_rand._train(x_a, y_a, t_a, x_a, t_a, y_a, 0, x_a.size(0), lr_train, n_epochs_feature)

    #Groud truth
    cate_test.append(nn_rand.predict_naive(x_t))
    
    
    cate_trans_reg, cate_trans, cate_conf, cate_unc = [], [], [], []
    cate_mtl, cate_mtl_delta_1, cate_mtl_bal, cate_mtl_bal_delta_1 = [], [], [], []
    cate_base_ridge, cate_base_rf, cate_base_nn = [], [], []
    cate_base_ridge_prop, cate_base_rf_prop, cate_base_nn_prop = [], [], []
    cate_base_ridge_modi, cate_base_rf_modi, cate_base_nn_modi = [], [], []
    cate_trans_delta_1, cate_trans_delta_2, cate_trans_out, cate_trans_out_1, cate_trans_out_2 = [], [], [], [], []
    cate_trans_bal, cate_trans_bal_out_1, cate_trans_bal_out_2 = [], [], []
    cate_avg = []
    cate_weight = []

    x_uc, y_uc, t_uc = torch.Tensor(x_unc.copy()), torch.Tensor(y_unc.copy()), torch.Tensor(t_unc.copy())
    #Validation UNC set
    x_u_t, x_val, t_u_t, t_val, y_u_t, y_val = train_test_split(x_uc, t_uc, y_uc, test_size = 0.2)


    for j_ind, j in enumerate(n_conf_size):
        print('###################     Training CorNet on ', j,' samples.     ####################') 
        x_c, t_c, y_c = x_conf[:j,:], t_conf[:j,:], y_conf[:j,:]
    
        #-----------------------------
        # Our algorithm
        #-----------------------------
        #Convert to tensors        
        x_c, y_c, t_c = torch.Tensor(x_c.copy()), torch.Tensor(y_c.copy()), torch.Tensor(t_c.copy()) 
        x_c, x_v, t_c, t_v, y_c, y_v = train_test_split(x_c, t_c, y_c, test_size = 0.2)
        x_t = torch.Tensor(x_test.copy())
    
        #Hyperparameters
        batch_size = x_c.size(0)
        lambd_h_c_l = [0.01, 0.1, 1.]
        lambd_ada_l = [np.sqrt((1/d_hidden)*np.log(d_hidden)/n_unc)]
        alpha_l = [1.]
        lambd_avg_l = [0., 0.25, 0.5, 0.75, 1]
        lambd_weight_l = [0., 0.5, 1., 2., 5.]
    
        
        #--------------------------------------------------------
        # Baselines CATE from "Removing hidden confounding ..."
        #--------------------------------------------------------
        q = (t_unc == 1).sum()/t_unc.shape[0]
        #Train 2StepRidge
        baseline_ridge = TwoStepRidge(t_unc, t_c, x_unc, x_c, y_unc, y_c, x_test, q)
        #Train 2StepRF
        baseline_rf = TwoStepRF(t_unc, t_c, x_unc, x_c, y_unc, y_c, x_test, q)
        
        cate_base_rf.append(baseline_rf[0])
        cate_base_ridge.append(baseline_ridge[0])
        
        
        #Train 2StepRidge
        baseline_ridge = TwoStepRidge(t_unc, t_conf, x_unc, x_conf, y_unc, y_conf, x_test, q)
        cate_base_ridge.append(baseline_ridge[0])
        #Train 2StepRF
        baseline_rf = TwoStepRF(t_unc, t_conf, x_unc, x_conf, y_unc, y_conf, x_test, q)
        cate_base_rf.append(baseline_rf[0])

        #Train 2StepRidge - prop
        baseline_ridge_prop = TwoStepRidge(t_unc, t_conf, x_unc, x_conf, y_unc, y_conf, x_test, q=prop_score)
        cate_base_ridge_prop.append(baseline_ridge_prop[0])
        #Train 2StepRF - prop
        baseline_rf_prop = TwoStepRF(t_unc, t_conf, x_unc, x_conf, y_unc, y_conf, x_test, q=prop_score)
        cate_base_rf_prop.append(baseline_rf_prop[0])

        #Train 2StepRidge - modi
        baseline_ridge_modi = TwoStepRidge_modi(t_unc, t_conf, x_unc, x_conf, y_unc, y_conf, x_test)
        cate_base_ridge_modi.append(baseline_ridge_modi[0])
        #Train 2StepRF - modi
        baseline_rf_modi = TwoStepRF_modi(t_unc, t_conf, x_unc, x_conf, y_unc, y_conf, x_test)
        cate_base_rf_modi.append(baseline_rf_modi[0])

        
        #-----------------------------
        # Our algorithm
        #-----------------------------  
        #\tau_cor^+
        best_loss = 1e100
        best_params_tn_reg = {'lambd_delta': 0, 'lambda_h_c': 0, 'lambda_ada': 0, 'alpha': 0}
        
        for lambd_ada in lambd_ada_l:
            for alpha in alpha_l:
                for lambd_h_c in lambd_h_c_l:
                        #Train feature map
                        tn_reg = CorNet_ADA(n_cov, n_hidden, d_hidden, n_out, d_out)
                        #Train both networks on confounded data (Step 1)
                        tn_reg._train(x_c, y_c, t_c, x_uc, t_uc, x_v, y_v, t_v, alpha, lambd_ada, lambd_h_c, batch_size, lr_train, n_epochs_feature, balancing=True)
                        
                        rep, o1, o0, d = tn_reg(x_v)
                        if (((y_v - (t_v*o1+(1-t_v)*o0))**2).mean()).detach().numpy() < best_loss:
                            best_tn_reg = deepcopy(tn_reg)
                            best_params_tn_reg['lambda_h_c'] = lambd_h_c
                            best_params_tn_reg['lambda_ada'] = lambd_ada
                            best_params_tn_reg['alpha'] = alpha
                            best_loss = (((y_v - (t_v*o1+(1-t_v)*o0))**2).mean()).detach().numpy()
        
        #For ablation: only balancing, but no bias regularization
        best_tn_bal = deepcopy(best_tn_reg)
        best_tn_bal_out_2 = deepcopy(best_tn_reg)
        best_tn_bal_out_1 = deepcopy(best_tn_reg)
        
        best_params_tn_reg['lambd_delta'] = np.sqrt((1/d_hidden)*np.log(d_hidden)/n_unc)#best_params_tn_reg['lambda_h_c']*np.sqrt(n_conf/n_unc)##np.sqrt(np.log(d_hidden)/n_unc)#np.sqrt(2*np.log(d_hidden)/n_unc)
        best_tn_reg._retrain_delta(x_uc, y_uc, t_uc, best_params_tn_reg['lambd_delta'], lr_ret, n_epochs_retrain)
        best_tn_bal._retrain_delta(x_uc, y_uc, t_uc, 0, lr_ret, n_epochs_retrain)
        best_tn_bal_out_2._retrain_delta(x_uc, y_uc, t_uc, best_params_tn_reg['lambd_delta'], lr_ret, n_epochs_retrain, bias_norm=2, bias_learning=False)
        best_tn_bal_out_1._retrain_delta(x_uc, y_uc, t_uc, best_params_tn_reg['lambd_delta'], lr_ret, n_epochs_retrain, bias_norm=1, bias_learning=False)
                    
             
        #\tau_cor
        best_loss = 1e100
        best_params_tn = {'lambd_delta': 0, 'lambda_h_c': 0, 'lambda_ada': 0, 'alpha': 0}
        for lambd_h_c in lambd_h_c_l:
            #Copy \hat{\phi}
            tn = CorNet(n_cov, n_hidden, d_hidden, n_out, d_out)
            tn._train(x_c, y_c, t_c, x_v, y_v, t_v, lambd_h_c, batch_size, lr_train, n_epochs_feature)                        
            
            rep, o1, o0 = tn(x_v)
            if (((y_v - (t_v*o1+(1-t_v)*o0))**2).mean()).detach().numpy() < best_loss:
                best_tn = deepcopy(tn)
                best_params_tn['lambda_h_c'] = lambd_h_c    
                best_loss = (((y_v - (t_v*o1+(1-t_v)*o0))**2).mean()).detach().numpy()
          
        #For the different ablations
        best_tn_delta_1 = deepcopy(best_tn)
        best_tn_delta_2 = deepcopy(best_tn)
        best_tn_out = deepcopy(best_tn)
        best_tn_out_1 = deepcopy(best_tn)
        best_tn_out_2 = deepcopy(best_tn)

    
        #Train head regulaized on uncofounded data (Step 2)
        best_params_tn['lambd_delta'] = np.sqrt((1/d_hidden)*np.log(d_hidden)/n_unc)#best_params_tn['lambda_h_c']*np.sqrt(n_conf/n_unc)#np.sqrt(0.1*np.log(d_hidden)/n_unc)##np.sqrt(np.log(d_hidden)/n_unc)#np.sqrt(2*np.log(d_hidden)/n_unc)
        best_tn._retrain_delta(x_uc, y_uc, t_uc, 0, lr_ret, n_epochs_retrain)
        best_tn_delta_1._retrain_delta(x_uc, y_uc, t_uc, best_params_tn['lambd_delta'], lr_ret, n_epochs_retrain)
        best_tn_delta_2._retrain_delta(x_uc, y_uc, t_uc, best_params_tn['lambd_delta'], lr_ret, n_epochs_retrain, bias_norm=2)
        best_tn_out._retrain_delta(x_uc, y_uc, t_uc, 0, lr_ret, n_epochs_retrain, bias_learning=False)
        best_tn_out_1._retrain_delta(x_uc, y_uc, t_uc, best_params_tn['lambd_delta'], lr_ret, n_epochs_retrain, bias_norm=1, bias_learning=False)
        best_tn_out_2._retrain_delta(x_uc, y_uc, t_uc, best_params_tn['lambd_delta'], lr_ret, n_epochs_retrain, bias_norm=2, bias_learning=False)
    
             
        #-----------------------------
        # Our variants
        #-----------------------------  
        #\tau_mtl
        best_loss = 1e100
        best_params_mtl = {'lambd_delta': 0, 'lambda_h_c': 0, 'lambda_ada': 0, 'alpha': 0}
        for lambd_h_c in lambd_h_c_l:
            #Train feature map
            mtl = MTL(n_cov, n_hidden, d_hidden, n_out, d_out)
            #Train both networks on confounded data (Step 1)
            mtl._train(x_c, y_c, t_c, x_uc, t_uc, y_uc, x_v, y_v, t_v, 0, 0, lambd_h_c, batch_size, lr_train, n_epochs_feature, balancing=False)
            
            rep, o1, o0, o1_u, o0_u, d = mtl(x_v)
            if (((y_v - (t_v*o1+(1-t_v)*o0))**2).mean()).detach().numpy() < best_loss:
                best_mtl = deepcopy(mtl)
                best_params_mtl['lambda_h_c'] = lambd_h_c
                best_loss = (((y_v - (t_v*o1+(1-t_v)*o0))**2).mean()).detach().numpy()
                
        #Train head regulaized on uncofounded data (Step 2)
        best_params_mtl['lambd_delta'] = np.sqrt((1/n_hidden)*np.log(d_hidden)/n_unc)#best_params_mtl['lambda_h_c']*np.sqrt(n_conf/n_unc)#np.sqrt(0.1*np.log(d_hidden)/n_unc)#best_params_tn_reg['lambda_h_c']*np.sqrt(n_conf/n_unc)#np.sqrt(np.log(d_hidden)/n_unc)#np.sqrt(2*np.log(d_hidden)/n_unc)
        
        #For variants    
        best_mtl._retrain_delta(x_uc, y_uc, t_uc, best_params_mtl['lambd_delta'], lr_ret, n_epochs_retrain)
                                       
        
        #\tau_mtl
        best_loss = 1e100
        best_params_mtl_bal = {'lambd_delta': 0, 'lambda_h_c': 0, 'lambda_ada': 0, 'alpha': 0}
        for lambd_ada in lambd_ada_l:
            for alpha in alpha_l:
                for lambd_h_c in lambd_h_c_l:
                    #Train feature map
                    mtl = MTL(n_cov, n_hidden, d_hidden, n_out, d_out)
                    #Train both networks on confounded data (Step 1)
                    mtl._train(x_c, y_c, t_c, x_uc, t_uc, y_uc, x_v, y_v, t_v, alpha, lambd_ada, lambd_h_c, batch_size, lr_train, n_epochs_feature, balancing=True)
                    
                    rep, o1, o0, o1_u, o0_u, d = mtl(x_v)
                    if (((y_v - (t_v*o1+(1-t_v)*o0))**2).mean()).detach().numpy() < best_loss:
                        best_mtl_bal = deepcopy(mtl)
                        best_params_mtl_bal['lambda_h_c'] = lambd_h_c
                        best_params_mtl_bal['lambda_ada'] = lambd_ada
                        best_params_mtl_bal['alpha'] = alpha
                        best_loss = (((y_v - (t_v*o1+(1-t_v)*o0))**2).mean()).detach().numpy()
        
        #Train head regulaized on uncofounded data (Step 2)
        best_params_mtl_bal['lambd_delta'] = np.sqrt((1/d_hidden)*np.log(d_hidden)/n_unc)#best_params_mtl_bal['lambda_h_c']*np.sqrt(n_conf/n_unc)#np.sqrt(0.1*np.log(d_hidden)/n_unc)#np.sqrt(np.log(d_hidden)/n_unc)#np.sqrt(2*np.log(d_hidden)/n_unc)
        best_mtl_bal._retrain_delta(x_uc, y_uc, t_uc, best_params_mtl_bal['lambd_delta'], lr_ret, n_epochs_retrain)
          

        #-----------------------------
        # Remaining baselines
        #-----------------------------  
                                 
        #Train 2StepNN
        baseline_tn = TwoStepTarNet(best_tn, t_unc, x_unc, y_unc, x_test, q)
        cate_base_nn.append(baseline_tn)
    
        #Train 2StepNN - prop
        baseline_tn_prop = TwoStepTarNet(best_tn, t_unc, x_unc, y_unc, x_test, q=prop_score)
        cate_base_nn_prop.append(baseline_tn_prop)
    
        #Train 2StepNN - modi
        baseline_tn_modi = TwoStepTarNet_modi(best_tn, t_unc, x_unc, y_unc, x_test)
        cate_base_nn_modi.append(baseline_tn_modi)
                     
        
        #\tau_avg
        best_loss = 1e100
        best_params_avg = {'lambda_h_u': 0,'lambda_h_c': 0, 'lambda_avg': 0}
        best_params_avg['lambda_h_c'] = best_params_tn['lambda_h_c']
        
        #On unc data
        tn_unc = CorNet(n_cov, n_hidden, d_hidden, n_out, d_out)
        tn_unc._train(x_u_t, y_u_t, t_u_t, x_v, y_v, t_v, best_params_avg['lambda_h_c'], x_u_t.shape[0], lr_train, n_epochs_feature)  
        
        rep, o1_u, o0_u = tn_unc(x_val)
        rep, o1_c, o0_c = best_tn(x_val)
    
        for lambd_avg in lambd_avg_l:
            o1 = lambd_avg*o1_u + (1-lambd_avg)*o1_c
            o0 = lambd_avg*o0_u + (1-lambd_avg)*o0_c
            if (((y_val - (t_val*o1+(1-t_val)*o0))**2).mean()).detach().numpy() < best_loss:
                best_tn_avg_u = deepcopy(tn_unc) 
                best_params_avg['lambda_avg'] = lambd_avg    
                best_loss = (((y_val - (t_val*o1+(1-t_val)*o0))**2).mean()).detach().numpy()
        params_avg_l.append(best_params_avg)

        #\tau_weight
        #Use same validation UNC set as above
        best_loss = 1e100
        best_params_weight = {'lambda_h_c': 0, 'lambda_weight': 0}
        for lambd_weight in lambd_weight_l:
            for lambd_h_c in lambd_h_c_l:
                wn = WeightNet(n_cov, n_hidden, d_hidden, n_out, d_out)
                wn._train(x_c, y_c, t_c, x_u_t, y_u_t, t_u_t, x_v, y_v, t_v, lambd_h_c, lambd_weight, batch_size, lr_train, n_epochs_feature)                        
                
                rep, o1, o0 = wn(x_val)
                if (((y_val - (t_val*o1+(1-t_val)*o0))**2).mean()).detach().numpy() < best_loss:
                    best_wn = deepcopy(wn)
                    best_params_weight['lambda_h_c'] = lambd_h_c    
                    best_params_weight['lambda_weight'] = lambd_weight    
                    best_loss = (((y_val - (t_val*o1+(1-t_val)*o0))**2).mean()).detach().numpy()        
        params_weight_l.append(best_params_weight)


        #Prediction of CATE on test data
        best_tn_reg.predict_delta(x_t).numpy()
        cate_trans_reg.append(best_tn_reg.predict_delta(x_t).numpy())
        cate_trans.append(best_tn.predict_delta(x_t).numpy())
        cate_conf.append(best_tn.predict_naive(x_t).numpy())
        cate_mtl.append(best_mtl.predict_naive(x_t).numpy())
        cate_mtl_delta_1.append(best_mtl.predict_delta(x_t).numpy())
        cate_mtl_bal.append(best_mtl_bal.predict_naive(x_t).numpy())
        cate_mtl_bal_delta_1.append(best_mtl_bal.predict_delta(x_t).numpy())
        cate_trans_delta_1.append(best_tn_delta_1.predict_delta(x_t).numpy())
        cate_trans_delta_2.append(best_tn_delta_2.predict_delta(x_t).numpy())
        cate_trans_out.append(best_tn_out.predict_out(x_t).numpy())
        cate_trans_out_1.append(best_tn_out_1.predict_out(x_t).numpy())
        cate_trans_out_2.append(best_tn_out_2.predict_out(x_t).numpy())
        cate_trans_bal.append(best_tn_bal.predict_delta(x_t).numpy())
        cate_trans_bal_out_1.append(best_tn_bal_out_1.predict_out(x_t).numpy())
        cate_trans_bal_out_2.append(best_tn_bal_out_2.predict_out(x_t).numpy())
        cate_avg.append(best_params_avg['lambda_avg']*best_tn_avg_u.predict_naive(x_t).numpy()+(1-best_params_avg['lambda_avg'])*best_tn.predict_naive(x_t).numpy())
        cate_weight.append(best_wn.predict(x_t).numpy())
        
        
    #\tau_unc - does not depend on conf data
    best_params_unc = best_params_tn
    #On unc data
    best_tn_unc = CorNet(n_cov, n_hidden, d_hidden, n_out, d_out)
    best_tn_unc._train(x_uc, y_uc, t_uc, x_v, y_v, t_v, best_params_unc['lambda_h_c'], x_uc.shape[0], lr_train, n_epochs_feature)   
    for j_ind, j in enumerate(n_conf_size):
        cate_unc.append(best_tn_unc.predict_naive(x_t).numpy())

    
    #Prediction of CATE on test data
    cate_trans_reg_l.append(cate_trans_reg)
    cate_trans_l.append(cate_trans)
    cate_conf_l.append(cate_conf)
    cate_unc_l.append(cate_unc)
    cate_mtl_l.append(cate_mtl)
    cate_mtl_delta_1_l.append(cate_mtl_delta_1)
    cate_mtl_bal_l.append(cate_mtl_bal)
    cate_mtl_bal_delta_1_l.append(cate_mtl_bal_delta_1)
    cate_trans_delta_1_l.append(cate_trans_delta_1)
    cate_trans_delta_2_l.append(cate_trans_delta_2)
    cate_trans_out_l.append(cate_trans_out)
    cate_trans_out_1_l.append(cate_trans_out_1)
    cate_trans_out_2_l.append(cate_trans_out_2)
    cate_trans_bal_l.append(cate_trans_bal)
    cate_trans_bal_out_1_l.append(cate_trans_bal_out_1)
    cate_trans_bal_out_2_l.append(cate_trans_bal_out_2)
    cate_avg_l.append(cate_avg)
    cate_weight_l.append(cate_weight)
    cate_base_rf_l.append(cate_base_rf)
    cate_base_ridge_l.append(cate_base_ridge)
    cate_base_nn_l.append(cate_base_nn)
    cate_base_rf_prop_l.append(cate_base_rf_prop)
    cate_base_ridge_prop_l.append(cate_base_ridge_prop)
    cate_base_nn_prop_l.append(cate_base_nn_prop)    
    cate_base_rf_modi_l.append(cate_base_rf_modi)
    cate_base_ridge_modi_l.append(cate_base_ridge_modi)
    cate_base_nn_modi_l.append(cate_base_nn_modi)



#%%


#Save data
setting = 'STAR'
folder = 'Experiments/Real_World/n_conf/'+ setting +'/'
n_u = '16'

np.save(folder+'exp_'+setting+'_trans_reg_l_'+n_u, cate_trans_reg_l)
np.save(folder+'exp_'+setting+'_trans_l_'+n_u, cate_trans_l)
np.save(folder+'exp_'+setting+'_conf_l_'+n_u, cate_conf_l)
np.save(folder+'exp_'+setting+'_mtl_l_'+n_u, cate_mtl_l)
np.save(folder+'exp_'+setting+'_mtl_delta_1_l_'+n_u, cate_mtl_delta_1_l)
np.save(folder+'exp_'+setting+'_mtl_bal_l_'+n_u, cate_mtl_bal_l)
np.save(folder+'exp_'+setting+'_mtl_bal_delta_1_l_'+n_u, cate_mtl_bal_delta_1_l)
np.save(folder+'exp_'+setting+'_trans_delta_1_l_'+n_u, cate_trans_delta_1_l)
np.save(folder+'exp_'+setting+'_trans_delta_2_l_'+n_u, cate_trans_delta_2_l)
np.save(folder+'exp_'+setting+'_trans_out_l_'+n_u, cate_trans_out_l)
np.save(folder+'exp_'+setting+'_trans_out_1_l_'+n_u, cate_trans_out_1_l)
np.save(folder+'exp_'+setting+'_trans_out_2_l_'+n_u, cate_trans_out_2_l)
np.save(folder+'exp_'+setting+'_trans_bal_l_'+n_u, cate_trans_bal_l)
np.save(folder+'exp_'+setting+'_trans_bal_out_1_l_'+n_u, cate_trans_bal_out_1_l)
np.save(folder+'exp_'+setting+'_trans_bal_out_2_l_'+n_u, cate_trans_bal_out_2_l)
np.save(folder+'exp_'+setting+'_unc_l_'+n_u, cate_unc_l) 
np.save(folder+'exp_'+setting+'_avg_l_'+n_u, cate_avg_l) 
np.save(folder+'exp_'+setting+'_weight_l_'+n_u, cate_weight_l) 
np.save(folder+'exp_'+setting+'_base_rf_l_'+n_u, cate_base_rf_l) 
np.save(folder+'exp_'+setting+'_base_ridge_l_'+n_u, cate_base_ridge_l)
np.save(folder+'exp_'+setting+'_base_nn_l_'+n_u, cate_base_nn_l)
np.save(folder+'exp_'+setting+'_base_rf_prop_l_'+n_u, cate_base_rf_prop_l) 
np.save(folder+'exp_'+setting+'_base_ridge_prop_l_'+n_u, cate_base_ridge_prop_l)
np.save(folder+'exp_'+setting+'_base_nn_prop_l_'+n_u, cate_base_nn_prop_l)
np.save(folder+'exp_'+setting+'_base_rf_modi_l_'+n_u, cate_base_rf_modi_l) 
np.save(folder+'exp_'+setting+'_base_ridge_modi_l_'+n_u, cate_base_ridge_modi_l)
np.save(folder+'exp_'+setting+'_base_nn_modi_l_'+n_u, cate_base_nn_modi_l)
np.save(folder+'exp_'+setting+'_cate_test_'+n_u, cate_test)

#Load data
setting = 'STAR'
folder = 'Experiments/Real_World/n_conf/'+ setting +'/'
n_u = '16'

cate_trans_reg_l = np.load(folder+'exp_'+setting+'_trans_reg_l_'+n_u+'.npy')
cate_trans_l = np.load(folder+'exp_'+setting+'_trans_l_'+n_u+'.npy')
cate_conf_l = np.load(folder+'exp_'+setting+'_conf_l_'+n_u+'.npy')
cate_mtl_l = np.load(folder+'exp_'+setting+'_mtl_l_'+n_u+'.npy')
cate_mtl_delta_1_l = np.load(folder+'exp_'+setting+'_mtl_delta_1_l_'+n_u+'.npy')
cate_mtl_bal_l = np.load(folder+'exp_'+setting+'_mtl_bal_l_'+n_u+'.npy')
cate_mtl_bal_delta_1_l = np.load(folder+'exp_'+setting+'_mtl_bal_delta_1_l_'+n_u+'.npy')
cate_trans_delta_1_l = np.load(folder+'exp_'+setting+'_trans_delta_1_l_'+n_u+'.npy')
cate_trans_delta_2_l = np.load(folder+'exp_'+setting+'_trans_delta_2_l_'+n_u+'.npy')
cate_trans_out_l = np.load(folder+'exp_'+setting+'_trans_out_l_'+n_u+'.npy')
cate_trans_out_1_l = np.load(folder+'exp_'+setting+'_trans_out_1_l_'+n_u+'.npy')
cate_trans_out_2_l = np.load(folder+'exp_'+setting+'_trans_out_2_l_'+n_u+'.npy')
cate_trans_bal_l = np.load(folder+'exp_'+setting+'_trans_bal_l_'+n_u+'.npy')
cate_trans_bal_out_1_l = np.load(folder+'exp_'+setting+'_trans_bal_out_1_l_'+n_u+'.npy')
cate_trans_bal_out_2_l = np.load(folder+'exp_'+setting+'_trans_bal_out_2_l_'+n_u+'.npy')
cate_unc_l = np.load(folder+'exp_'+setting+'_unc_l_'+n_u+'.npy') 
cate_avg_l = np.load(folder+'exp_'+setting+'_avg_l_'+n_u+'.npy') 
cate_weight_l = np.load(folder+'exp_'+setting+'_weight_l_'+n_u+'.npy') 
cate_base_rf_l = np.load(folder+'exp_'+setting+'_base_rf_l_'+n_u+'.npy') 
cate_base_ridge_l = np.load(folder+'exp_'+setting+'_base_ridge_l_'+n_u+'.npy')
cate_base_nn_l = np.load(folder+'exp_'+setting+'_base_nn_l_'+n_u+'.npy')
cate_base_rf_prop_l = np.load(folder+'exp_'+setting+'_base_rf_prop_l_'+n_u+'.npy') 
cate_base_ridge_prop_l = np.load(folder+'exp_'+setting+'_base_ridge_prop_l_'+n_u+'.npy')
cate_base_nn_prop_l = np.load(folder+'exp_'+setting+'_base_nn_prop_l_'+n_u+'.npy')
cate_base_rf_modi_l = np.load(folder+'exp_'+setting+'_base_rf_modi_l_'+n_u+'.npy') 
cate_base_ridge_modi_l = np.load(folder+'exp_'+setting+'_base_ridge_modi_l_'+n_u+'.npy')
cate_base_nn_modi_l = np.load(folder+'exp_'+setting+'_base_nn_modi_l_'+n_u+'.npy')
cate_test = np.load(folder+'exp_'+setting+'_cate_test_'+n_u+'.npy', allow_pickle = True)






k=4

print(np.sqrt(np.array([((cate_test[i].numpy() - cate_trans_l[i][k])**2).mean() for i in range(n_run)])).mean())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_trans_reg_l[i][k])**2).mean() for i in range(n_run)])).mean())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_conf_l[i][k])**2).mean() for i in range(n_run)])).mean())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_unc_l[i][k])**2).mean() for i in range(n_run)])).mean())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_base_rf_l[i][k])**2).mean() for i in range(n_run)])).mean())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_base_ridge_l[i][k])**2).mean() for i in range(n_run)])).mean())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_base_nn_l[i][k])**2).mean() for i in range(n_run)])).mean())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_avg_l[i][k])**2).mean() for i in range(n_run)])).mean())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_weight_l[i][k])**2).mean() for i in range(n_run)])).mean())

print(np.sqrt(np.array([((cate_test[i].numpy() - cate_trans_l[i][k])**2).mean() for i in range(n_run)])).std())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_trans_reg_l[i][k])**2).mean() for i in range(n_run)])).std())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_conf_l[i][k])**2).mean() for i in range(n_run)])).std())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_unc_l[i][k])**2).mean() for i in range(n_run)])).std())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_base_rf_l[i][k])**2).mean() for i in range(n_run)])).std())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_base_ridge_l[i][k])**2).mean() for i in range(n_run)])).std())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_base_nn_l[i][k])**2).mean() for i in range(n_run)])).std())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_avg_l[i][k])**2).mean() for i in range(n_run)])).std())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_weight_l[i][k])**2).mean() for i in range(n_run)])).std())


#For ablation study
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_trans_l[i][k])**2).mean() for i in range(n_run)])).mean())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_trans_delta_1_l[i][k])**2).mean() for i in range(n_run)])).mean())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_trans_delta_2_l[i][k])**2).mean() for i in range(n_run)])).mean())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_trans_out_l[i][k])**2).mean() for i in range(n_run)])).mean())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_trans_out_1_l[i][k])**2).mean() for i in range(n_run)])).mean())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_trans_out_2_l[i][k])**2).mean() for i in range(n_run)])).mean())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_trans_bal_l[i][k])**2).mean() for i in range(n_run)])).mean())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_trans_bal_out_1_l[i][k])**2).mean() for i in range(n_run)])).mean())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_trans_bal_out_2_l[i][k])**2).mean() for i in range(n_run)])).mean())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_trans_reg_l[i][k])**2).mean() for i in range(n_run)])).mean())


print(np.sqrt(np.array([((cate_test[i].numpy() - cate_trans_l[i][k])**2).mean() for i in range(n_run)])).std())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_trans_delta_1_l[i][k])**2).mean() for i in range(n_run)])).std())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_trans_delta_2_l[i][k])**2).mean() for i in range(n_run)])).std())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_trans_out_l[i][k])**2).mean() for i in range(n_run)])).std())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_trans_out_1_l[i][k])**2).mean() for i in range(n_run)])).std())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_trans_out_2_l[i][k])**2).mean() for i in range(n_run)])).std())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_trans_bal_l[i][k])**2).mean() for i in range(n_run)])).std())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_trans_bal_out_1_l[i][k])**2).mean() for i in range(n_run)])).std())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_trans_bal_out_2_l[i][k])**2).mean() for i in range(n_run)])).std())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_trans_reg_l[i][k])**2).mean() for i in range(n_run)])).std())




#For variants
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_mtl_l[i][k])**2).mean() for i in range(n_run)])).mean())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_mtl_delta_1_l[i][k])**2).mean() for i in range(n_run)])).mean())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_mtl_bal_l[i][k])**2).mean() for i in range(n_run)])).mean())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_mtl_bal_delta_1_l[i][k])**2).mean() for i in range(n_run)])).mean())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_trans_l[i][k])**2).mean() for i in range(n_run)])).mean())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_trans_reg_l[i][k])**2).mean() for i in range(n_run)])).mean())

print(np.sqrt(np.array([((cate_test[i].numpy() - cate_mtl_l[i][k])**2).mean() for i in range(n_run)])).std())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_mtl_delta_1_l[i][k])**2).mean() for i in range(n_run)])).std())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_mtl_bal_l[i][k])**2).mean() for i in range(n_run)])).std())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_mtl_bal_delta_1_l[i][k])**2).mean() for i in range(n_run)])).std())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_trans_l[i][k])**2).mean() for i in range(n_run)])).std())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_trans_reg_l[i][k])**2).mean() for i in range(n_run)])).std())



#For baseline modification
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_base_rf_l[i][k])**2).mean() for i in range(n_run)])).mean())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_base_ridge_l[i][k])**2).mean() for i in range(n_run)])).mean())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_base_nn_l[i][k])**2).mean() for i in range(n_run)])).mean())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_base_rf_prop_l[i][k])**2).mean() for i in range(n_run)])).mean())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_base_ridge_prop_l[i][k])**2).mean() for i in range(n_run)])).mean())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_base_nn_prop_l[i][k])**2).mean() for i in range(n_run)])).mean())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_base_rf_modi_l[i][k])**2).mean() for i in range(n_run)])).mean())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_base_ridge_modi_l[i][k])**2).mean() for i in range(n_run)])).mean())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_base_nn_modi_l[i][k])**2).mean() for i in range(n_run)])).mean())


print(np.sqrt(np.array([((cate_test[i].numpy() - cate_base_rf_l[i][k])**2).mean() for i in range(n_run)])).std())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_base_ridge_l[i][k])**2).mean() for i in range(n_run)])).std())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_base_nn_l[i][k])**2).mean() for i in range(n_run)])).std())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_base_rf_prop_l[i][k])**2).mean() for i in range(n_run)])).std())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_base_ridge_prop_l[i][k])**2).mean() for i in range(n_run)])).std())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_base_nn_prop_l[i][k])**2).mean() for i in range(n_run)])).std())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_base_rf_modi_l[i][k])**2).mean() for i in range(n_run)])).std())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_base_ridge_modi_l[i][k])**2).mean() for i in range(n_run)])).std())
print(np.sqrt(np.array([((cate_test[i].numpy() - cate_base_nn_modi_l[i][k])**2).mean() for i in range(n_run)])).std())

