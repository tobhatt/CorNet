from IPython import get_ipython
get_ipython().magic('reset -sf')
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path
from simdata import sim_data_assum_shared_rep
from scipy.sparse import random
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib
from models import CorNet

#-------------------------------------------
# Study F1: Shared Representation
#-------------------------------------------
'''
Set d_\infty = 1 (no covariate shift) and fix n_unc, increase n_conf.
Compare our algorithm on n_unc + n_conf (2-step) with different representation functions
but the same 
'''

# Define \phi and h^u, h^c
n_cov = 25
n_hidden_cov = 25
sparse_ratio = 0.6666
alpha = 10
beta = 2
infty_norm_1 = 1.5
infty_norm_2 = 2.0
n_run=10

#Net with weights
net = CorNet(n_cov, 3, n_hidden_cov, 0, n_hidden_cov)

#normalize outcome weights to || ...|| = alpha
l2_1 = torch.norm(torch.Tensor(net.linear_out_1.weight.detach()), p=2)
l2_0 = torch.norm(torch.Tensor(net.linear_out_0.weight.detach()), p=2)
net.linear_out_1.weight = torch.nn.Parameter(alpha * net.linear_out_1.weight.detach()/(l2_1))
net.linear_out_0.weight = torch.nn.Parameter(alpha * net.linear_out_0.weight.detach()/(l2_0))

#Sparse \delta 
#net.delta_1.weight = torch.nn.parameter.Parameter(torch.randn_like(net.delta_1.weight)+0.1)
#net.delta_0.weight = torch.nn.parameter.Parameter(torch.randn_like(net.delta_0.weight)-0.1)
delta1 = net.delta_1.weight.detach()
delta0 = net.delta_0.weight.detach()
sparse_index = np.random.choice(range(1,n_hidden_cov), size=int(sparse_ratio * n_hidden_cov), replace = False)
delta1.detach().numpy()[:,sparse_index] = 0
sparse_index = np.random.choice(range(1,n_hidden_cov), size=int(sparse_ratio * n_hidden_cov), replace = False)
delta0.detach().numpy()[:,sparse_index] = 0
net.delta_1.weight = torch.nn.Parameter(delta1)
net.delta_0.weight = torch.nn.Parameter(delta0)


#normalize to beta
sim_dict = sim_data_assum_shared_rep(net, infty_norm_1, infty_norm_2, 1000, 50, n_cov)
x_test = sim_dict['x_test']
print(((net.predict_delta(x_test) - net.predict_naive(x_test))**2).mean())
bias = ((net.predict_delta(x_test) - net.predict_naive(x_test))**2).mean()

net.delta_1.weight = torch.nn.Parameter(beta * net.delta_1.weight/torch.sqrt(bias))
net.delta_1.bias = torch.nn.Parameter(beta * net.delta_1.bias/torch.sqrt(bias))

net.delta_0.weight = torch.nn.Parameter(beta * net.delta_0.weight/torch.sqrt(bias))
net.delta_0.bias = torch.nn.Parameter(beta * net.delta_0.bias/torch.sqrt(bias))

print(((net.predict_delta(x_test) - net.predict_naive(x_test))**2).mean())


cate_trans_l, cate_trans_reg_l, cate_unc_l, cate_conf_l = [], [], [], []
cate_trans_shared_l, cate_trans_reg_shared_l, cate_unc_shared_l, cate_conf_shared_l = [], [], [], []
cate_trans_shared_2_l, cate_trans_reg_shared_2_l, cate_unc_shared_2_l, cate_conf_shared_2_l = [], [], [], []

for i in range(10):
    
    # Simulate data
    n_conf = 500
    n_unc = 50
    sim_dict = sim_data_assum_shared_rep(net, infty_norm_1, infty_norm_2, n_conf, n_unc, n_cov)
    
    #Unconfounded data
    x_unc = sim_dict['x_unc']
    t_unc = sim_dict['t_unc']
    y_unc = sim_dict['y_unc']
    
    #Confounded data
    x_conf = sim_dict['x_conf']
    t_conf = sim_dict['t_conf']
    y_conf = sim_dict['y_conf']
    
    #Confounded data
    x_conf_shared = sim_dict['x_conf_shared']
    t_conf_shared = sim_dict['t_conf_shared']
    y_conf_shared = sim_dict['y_conf_shared']
    
    #Confounded data
    x_conf_shared_2 = sim_dict['x_conf_shared_2']
    t_conf_shared_2 = sim_dict['t_conf_shared_2']
    y_conf_shared_2 = sim_dict['y_conf_shared_2']
    
    #Validation set
    x_val, t_val, y_val = [], [], []
    
    #Test data
    x_test = sim_dict['x_test']
    cate_test = sim_dict['cate_test']


    n_cov = x_conf.shape[1]
    n_hidden = 3
    d_hidden = n_hidden_cov
    n_out = 0
    d_out = d_hidden
    
    lr_train = 0.1
    lr_ret = 0.1
    n_epochs_feature = 500
    n_epochs_retrain = 200
    batch_size = x_conf.size(0)
    lambd_h_c = 0.1
    lambd_delta = 0


    cate_trans = []
    #cate_trans_reg = []
    cate_conf = []
    cate_unc = []

    cate_trans_shared = []
    #cate_trans_reg_shared = []
    cate_conf_shared = []
    cate_unc_shared = []
    
    cate_trans_shared_2 = []
    #cate_trans_reg_shared = []
    cate_conf_shared_2 = []
    cate_unc_shared_2 = []
    
    m = n_unc
    x_uc, t_uc, y_uc = x_unc[:m,:], t_unc[:m,:], y_unc[:m,:]
    ####\tau_unc
    tn_unc = CorNet(n_cov, n_hidden, d_hidden, n_out, d_out)
    tn_unc._train(x_uc, y_uc, t_uc, x_val, t_val, y_val, lambd_h_c, batch_size, lr_train, n_epochs_feature)
    for i in range(n_run):
        cate_unc.append(((cate_test - tn_unc.predict_naive(x_test).numpy())**2).mean())
        cate_unc_shared.append(((cate_test - tn_unc.predict_naive(x_test).numpy())**2).mean())
        cate_unc_shared_2.append(((cate_test - tn_unc.predict_naive(x_test).numpy())**2).mean())
    
    n_conf_sizes = np.linspace(25, n_conf, n_run)
    for i in range(n_run):
        n = int(n_conf_sizes[i])
        print('###################     Training CorNet on ', n,' samples.     ####################')
        x_c, t_c, y_c = x_conf[:n,:], t_conf[:n,:], y_conf[:n,:]
        x_c_s, t_c_s, y_c_s = x_conf_shared[:n,:], t_conf_shared[:n,:], y_conf_shared[:n,:]
        x_c_s_2, t_c_s_2, y_c_s_2 = x_conf_shared_2[:n,:], t_conf_shared_2[:n,:], y_conf_shared_2[:n,:]
        
        ####\tau_cor
        #Train both networks on confounded data (Step 1)
        tn = CorNet(n_cov, n_hidden, d_hidden, n_out, d_out)
        tn._train(x_c, y_c, t_c, x_val, y_val, t_val, lambd_h_c, batch_size, lr_train, n_epochs_feature)
        
        #Train head regulaized on uncofounded data (Step 2) WITHOUT covariate shift
        tn._retrain_delta(x_uc, y_uc, t_uc, 0, lr_ret, n_epochs_retrain)
       
        #Prediction of CATE on test data
        cate_trans.append(((cate_test - tn.predict_delta(x_test).numpy())**2).mean())
        cate_conf.append(((cate_test - tn.predict_naive(x_test).numpy())**2).mean())
    
        ####\tau_cor w/ shared rep
        #Train both networks on confounded data (Step 1)
        tn = CorNet(n_cov, n_hidden, d_hidden, n_out, d_out)
        tn._train(x_c_s, y_c_s, t_c_s, x_val, y_val, t_val, lambd_h_c, batch_size, lr_train, n_epochs_feature)
        
        #Train head regulaized on uncofounded data (Step 2) WITHOUT covariate shift
        tn._retrain_delta(x_uc, y_uc, t_uc, 0, lr_ret, n_epochs_retrain)
       
        #Prediction of CATE on test data
        cate_trans_shared.append(((cate_test - tn.predict_delta(x_test).numpy())**2).mean())
        cate_conf_shared.append(((cate_test - tn.predict_naive(x_test).numpy())**2).mean())


        ####\tau_cor w/ shared rep
        #Train both networks on confounded data (Step 1)
        tn = CorNet(n_cov, n_hidden, d_hidden, n_out, d_out)
        tn._train(x_c_s_2, y_c_s_2, t_c_s_2, x_val, y_val, t_val, lambd_h_c, batch_size, lr_train, n_epochs_feature)
        
        #Train head regulaized on uncofounded data (Step 2) WITHOUT covariate shift
        tn._retrain_delta(x_uc, y_uc, t_uc, 0, lr_ret, n_epochs_retrain)
       
        #Prediction of CATE on test data
        cate_trans_shared_2.append(((cate_test - tn.predict_delta(x_test).numpy())**2).mean())
        cate_conf_shared_2.append(((cate_test - tn.predict_naive(x_test).numpy())**2).mean())


    cate_trans_l.append(cate_trans) 
    cate_conf_l.append(cate_conf)
    cate_unc_l.append(cate_unc)

    cate_trans_shared_l.append(cate_trans_shared) 
    cate_conf_shared_l.append(cate_conf_shared)
    cate_unc_shared_l.append(cate_unc_shared)

    cate_trans_shared_2_l.append(cate_trans_shared_2) 
    cate_conf_shared_2_l.append(cate_conf_shared_2)
    cate_unc_shared_2_l.append(cate_unc_shared_2)


#%%


# =============================================================================
# 
# #Save data
# np.save('sim_fail_shared_trans1', cate_t)
# np.save('sim_fail_shared_unc1', cate_u)
# np.save('sim_fail_shared_conf1', cate_c)
# np.save('sim_fail_shared_trans2', cate_t_s)
# np.save('sim_fail_shared_unc2', cate_u_s)
# np.save('sim_fail_shared_conf2', cate_c_s)
# np.save('sim_fail_shared_trans3', cate_t_s_2)
# np.save('sim_fail_shared_unc3', cate_u_s_2)
# np.save('sim_fail_shared_conf3', cate_c_s_2)
# 
# 
# =============================================================================

# =============================================================================
# #Load data
# cate_t = np.load('sim_fail_shared_trans1.npy')
# cate_u = np.load('sim_fail_shared_unc1.npy')
# cate_c = np.load('sim_fail_shared_conf1.npy')
# cate_t_s = np.load('sim_fail_shared_trans2.npy')
# cate_u_s = np.load('sim_fail_shared_unc2.npy')
# cate_c_s = np.load('sim_fail_shared_conf2.npy')
# cate_t_s_2 = np.load('sim_fail_shared_trans3.npy')
# cate_u_s_2 = np.load('sim_fail_shared_unc3.npy')
# cate_c_s_2 = np.load('sim_fail_shared_conf3.npy')
# =============================================================================

n_conf = 500
t = n_conf_sizes = np.linspace(25, n_conf, n_run)
ms = 3
lw=1
elw=0.5
cs=2

fig, ax = plt.subplots(1,3)
fig.set_figwidth(12.5)
fig.set_figheight(5)

ax[0].errorbar(t, cate_t.mean(0), cate_t.std(0), mfc='black', color = 'black', ecolor = 'black', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)
ax[0].errorbar(t, cate_c.mean(0), cate_c.std(0), mfc='b', color = 'b', ecolor = 'b', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)
ax[0].errorbar(t, cate_u.mean(0), cate_u.std(0), mfc='r', color = 'r', ecolor = 'r', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)
ax[0].set(xlabel='$n^{Conf}$', ylabel=r'$\sqrt{\epsilon_{PEHE}}$',
       title=r'$\beta_\phi=0$, i.e., shared rep.', ylim=(0.5, 4.5))
ax[0].xaxis.set_ticks(np.array([25, 200, 400]))
ax[0].yaxis.set_ticks(np.array([1, 2, 3, 4]))
ax[0].grid(linewidth=1, ls='--')

ax[1].errorbar(t, cate_t_s.mean(0), cate_t_s.std(0), mfc='black', color = 'black', ecolor = 'black', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)
ax[1].errorbar(t, cate_c_s.mean(0), cate_c_s.std(0), mfc='b', color = 'b', ecolor = 'b', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)
ax[1].errorbar(t, cate_u_s.mean(0), cate_u_s.std(0), mfc='r', color = 'r', ecolor = 'r', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)
ax[1].set(xlabel='$n^{Conf}$',
       title=r'$\beta_\phi$ small', ylim=(0.5, 4.5))
ax[1].xaxis.set_ticks(np.array([25, 200, 400]))
ax[1].yaxis.set_ticks(np.array([1, 2, 3, 4]))
ax[1].grid(linewidth=1, ls='--')

ax[2].errorbar(t, cate_t_s_2.mean(0), cate_t_s_2.std(0), mfc='black', color = 'black', ecolor = 'black', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)
ax[2].errorbar(t, cate_c_s_2.mean(0), cate_c_s_2.std(0), mfc='b', color = 'b', ecolor = 'b', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)
ax[2].errorbar(t, cate_u_s_2.mean(0), cate_u_s_2.std(0), mfc='r', color = 'r', ecolor = 'r', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)
ax[2].set(xlabel='$n^{Conf}$',
       title=r'$\beta_\phi$ large', ylim=(0.5, 4.5))
ax[2].xaxis.set_ticks(np.array([25, 200, 400]))
ax[2].yaxis.set_ticks(np.array([1, 2, 3, 4]))
ax[2].grid(linewidth=1, ls='--')

line1 = Line2D([0,1],[0,1],linestyle='-', color='black')
line2 = Line2D([0,1],[0,1],linestyle='-', color='b')
line3 = Line2D([0,1],[0,1],linestyle='-', color='r')

fig.legend([line1, line2, line3], [r'$\tau_{CorNet}$', r'$\tau_{Conf}$', r'$\tau_{Unc}$'], bbox_to_anchor=(0.5, -0.075), loc = 'lower center', ncol=3, numpoints=1)

#fig.savefig("sim_fail_shared_rep.pdf", dpi=300, bbox_inches='tight')
plt.show()


