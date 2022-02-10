from IPython import get_ipython
get_ipython().magic('reset -sf')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path
from simdata import *
from models import *
from scipy.sparse import random
import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib

#-------------------------------------------
# Study 3: Impact of bias
#-------------------------------------------
'''
Compare to NN on RCT with
    1) d_\infty = 1, beta=1
    2) d_\infty > 1, beta=1
    3) d_\infty = 1, beta>1
Increase n/m and see how the threshold (NN on RCT vs. ours) changes across settings
'''

# Define \phi and h^u, h^c
n_cov = 25
n_hidden_cov = 25
sparse_ratio = 0.666
beta = 1
beta_2 = np.sqrt(2)
n_run=15

#Net with weights
net = CorNet(n_cov, 3, n_hidden_cov, 0, n_hidden_cov)

#Sparse \delta 
with torch.no_grad():
    delta1 = deepcopy(net.delta_1.weight)
    delta0 = deepcopy(net.delta_0.weight)
    sparse_index = np.random.choice(range(1,n_hidden_cov), size=int(sparse_ratio * n_hidden_cov), replace = False)
    delta1.detach().numpy()[:,sparse_index] = 0
    sparse_index = np.random.choice(range(1,n_hidden_cov), size=int(sparse_ratio * n_hidden_cov), replace = False)
    delta0.detach().numpy()[:,sparse_index] = 0
    net.delta_1.weight = torch.nn.Parameter(delta1)
    net.delta_0.weight = torch.nn.Parameter(delta0)


#normalize to beta
n_test=10000
x_test = torch.Tensor(np.random.multivariate_normal(np.zeros(n_cov), np.identity(n_cov), size= [n_test]))
with torch.no_grad():
    print(((net.predict_delta(x_test) - net.predict_naive(x_test))**2).mean())
    bias = ((net.predict_delta(x_test).numpy() - net.predict_naive(x_test).numpy())**2).mean()
    
    net.delta_1.weight = torch.nn.Parameter(beta * net.delta_1.weight.detach()/np.sqrt(bias))
    net.delta_1.bias = torch.nn.Parameter(beta * net.delta_1.bias.detach()/np.sqrt(bias))
    
    net.delta_0.weight = torch.nn.Parameter(beta * net.delta_0.weight.detach()/np.sqrt(bias))
    net.delta_0.bias = torch.nn.Parameter(beta * net.delta_0.bias.detach()/np.sqrt(bias))
    
    print(((net.predict_delta(x_test).numpy() - net.predict_naive(x_test).numpy())**2).mean())


cate_trans_l = []
cate_conf_l = []
cate_trans_cov_l = []
cate_conf_cov_l = []
cate_trans_bias_l = []
cate_conf_bias_l = []
cate_unc_l = []

cate_unc_true_l = []
cate_conf_true_l = []

for i in range(10):
    # Simulate data
    n_conf = 300
    n_unc = 25
    nn = deepcopy(net)
    sim_dict = sim_data_condition_conf(nn, beta, beta_2, n_conf, n_unc, n_cov)

    #Unconfounded data
    x_unc = sim_dict['x_unc']
    t_unc = sim_dict['t_unc']
    y_unc = sim_dict['y_unc']
    
    #Unconfounded data WITH covariate shift
    x_unc_cov = sim_dict['x_unc_cov']
    t_unc_cov = sim_dict['t_unc_cov']
    y_unc_cov = sim_dict['y_unc_cov']
    
    #Unconfounded data Bias
    x_conf_bias = sim_dict['x_conf_bias']
    t_conf_bias = sim_dict['t_conf_bias']
    y_conf_bias = sim_dict['y_conf_bias']
    
    #Confounded data
    x_conf = sim_dict['x_conf']
    t_conf = sim_dict['t_conf']
    y_conf = sim_dict['y_conf']
    
    #Validation set
    x_val, y_val, t_val = [], [], [] 
    
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
    cate_trans_cov = []
    cate_trans_bias = []
 
    cate_conf = []
    cate_conf_cov = []
    cate_conf_bias = []
    cate_unc = []
    cate_unc_true = []
    cate_conf_true = []
    
    x_c, t_c, y_c = x_conf[:n_conf,:], t_conf[:n_conf,:], y_conf[:n_conf,:]
    x_c_bias, t_c_bias, y_c_bias = x_conf_bias[:n_conf,:], t_conf_bias[:n_conf,:], y_conf_bias[:n_conf,:]
    
    #Does not depend on n_unc
    ###\tau_cor
    tn_1 = CorNet(n_cov, n_hidden, d_hidden, n_out, d_out)
    #Train both networks on confounded data (Step 1)
    tn_1._train(x_c, y_c, t_c, x_val, y_val, t_val, lambd_h_c, x_c.size(0), lr_train, n_epochs_feature)         
    
    
    tn_bias_1 = CorNet(n_cov, n_hidden, d_hidden, n_out, d_out)
    tn_bias_1._train(x_c_bias, y_c_bias, t_c_bias, x_val, y_val, t_val, lambd_h_c, x_c.size(0), lr_train, n_epochs_feature)         
    
    
    for i in range(n_run):
        cate_conf.append(tn_1.predict_naive(x_test))
        cate_conf_true.append(net.predict_naive(x_test))
        cate_conf_cov.append(tn_1.predict_naive(x_test))
        cate_conf_bias.append(tn_bias_1.predict_naive(x_test))
        
        
    n_unc_sizes = np.concatenate((np.linspace(3, n_unc/2, 10), np.linspace(n_unc/2, n_unc, 6)[1:]))#np.linspace(10, n_conf, n_run)
    #np.linspace(2, n_unc, n_run)
    for i in range(n_run):
        m = int(n_unc_sizes[i])
        print('###################     Training CorNet on ', m,' samples.     ####################')
        x_uc, t_uc, y_uc = x_unc[:m,:], t_unc[:m,:], y_unc[:m,:]
        x_uc_cov, t_uc_cov, y_uc_cov = x_unc_cov[:m,:], t_unc_cov[:m,:], y_unc_cov[:m,:]

        tn = deepcopy(tn_1)
        #Train head regulaized on uncofounded data (Step 2)
        tn._retrain_delta(x_uc, y_uc, t_uc, lambd_delta, lr_ret, n_epochs_retrain)
        #Prediction of CATE on test data
        cate_trans.append(tn.predict_delta(x_test))
          
        tn_cov = deepcopy(tn_1)
        #Train head regulaized on uncofounded data (Step 2)
        tn_cov._retrain_delta(x_uc_cov, y_uc_cov, t_uc_cov, lambd_delta, lr_ret, n_epochs_retrain)
        #Prediction of CATE on test data
        cate_trans_cov.append(tn_cov.predict_delta(x_test))
        
        tn_bias = deepcopy(tn_bias_1)
        #Train head regulaized on uncofounded data (Step 2)
        tn_bias._retrain_delta(x_uc, y_uc, t_uc, lambd_delta, lr_ret, n_epochs_retrain)
        #Prediction of CATE on test data
        cate_trans_bias.append(tn_bias.predict_delta(x_test))
     
            
     
        tn_unc = CorNet(n_cov, n_hidden, d_hidden, n_out, d_out)
        tn_unc._train(x_uc, y_uc, t_uc, x_val, y_val, t_val, lambd_h_c, x_uc.size(0), lr_train, n_epochs_feature)         
        cate_unc.append(tn_unc.predict_naive(x_test))
        cate_unc_true.append(net.predict_delta(x_test))
        
    cate_trans_bias_l.append(np.array([((cate_test - cate_trans_bias[i].numpy())**2).mean() for i in range(n_run)]))
    cate_trans_cov_l.append(np.array([((cate_test - cate_trans_cov[i].numpy())**2).mean() for i in range(n_run)]))
    cate_trans_l.append(np.array([((cate_test - cate_trans[i].numpy())**2).mean() for i in range(n_run)]))        
    cate_conf_bias_l.append(np.array([((cate_test - cate_conf_bias[i].numpy())**2).mean() for i in range(n_run)]))
    cate_conf_cov_l.append(np.array([((cate_test - cate_conf_cov[i].numpy())**2).mean() for i in range(n_run)]))
    cate_conf_l.append(np.array([((cate_test - cate_conf[i].numpy())**2).mean() for i in range(n_run)]))        
    cate_unc_l.append(np.array([((cate_test - cate_unc[i].numpy())**2).mean() for i in range(n_run)]))        
    cate_unc_true_l.append(np.array([((cate_test - cate_unc_true[i].numpy())**2).mean() for i in range(n_run)]))        
    cate_conf_true_l.append(np.array([((cate_test - cate_conf_true[i].numpy())**2).mean() for i in range(n_run)]))        
    


#%%


matplotlib.rcParams.update({'font.size': 12})

# Data for plotting
t = n_unc_sizes 

cate_t = np.array(cate_trans_l)
cate_c = np.array(cate_conf_l)
cate_u = np.array(cate_unc_l)

cate_c_true = np.array(cate_conf_true_l)
cate_u_true = np.array(cate_unc_true_l)


cate_t_cov = np.array(cate_trans_cov_l)
cate_c_cov = np.array(cate_conf_cov_l)

cate_t_bias = np.array(cate_trans_bias_l)
cate_c_bias = np.array(cate_conf_bias_l)

# =============================================================================
# #Save data
# np.save('sim_cond_conf_trans', cate_t)
# np.save('sim_cond_conf_c', cate_c)
# np.save('sim_cond_conf_trans_cov', cate_t_cov)
# np.save('sim_cond_conf_c_cov', cate_c_cov)
# np.save('sim_cond_conf_trans_bias', cate_t_bias)
# np.save('sim_cond_conf_c_bias', cate_c_bias)
# 
# 
# =============================================================================


# =============================================================================
# #Load data
# cate_t = np.load('sim_cond_conf_trans.npy')
# cate_c = np.load('sim_cond_conf_c.npy')
# cate_t_cov = np.load('sim_cond_conf_trans_cov.npy')
# cate_c_cov = np.load('sim_cond_conf_c_cov.npy')
# cate_t_bias = np.load('sim_cond_conf_trans_bias.npy')
# cate_c_bias = np.load('sim_cond_conf_c_bias.npy')
# 
# =============================================================================



ms = 3
lw=1
elw=0.5
cs=2
n_unc = 25
n_unc_sizes = np.concatenate((np.linspace(3, n_unc/2, 10), np.linspace(n_unc/2, n_unc, 6)[1:]))#np.linspace(10, n_conf, n_run)
t = n_unc_sizes

fig, ax = plt.subplots(3, 1)
fig.set_figwidth(5)
fig.set_figheight(5)
ax[0].errorbar(t, cate_t.mean(0), cate_t.std(0), mfc='black', color = 'black', ecolor = 'black', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)#plot(t, cate_t, 'b')
ax[0].errorbar(t, cate_c.mean(0), cate_c.std(0), mfc='b', color = 'b', ecolor = 'b', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)#plot(t, cate_n, 'r')

ax[1].errorbar(t, cate_t_cov.mean(0), cate_t_cov.std(0), mfc='black', color = 'black', ecolor = 'black', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)#.plot(t, cate_t_cov, 'b')
ax[1].errorbar(t, cate_c_cov.mean(0), cate_c_cov.std(0), mfc='b', color = 'b', ecolor = 'b', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)#plot(t, cate_n_cov, 'r')
ax[2].errorbar(t, cate_t_bias.mean(0), cate_t_bias.std(0), mfc='black', color = 'black', ecolor = 'black', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)#.plot(t, cate_t_bias, 'b')
ax[2].errorbar(t, cate_c_bias.mean(0), cate_c_bias.std(0), mfc='b', color = 'b', ecolor = 'b', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)#.plot(t, cate_n_bias, 'r')
ax[0].set_xticklabels(np.repeat("", n_run))
ax[1].set_xticklabels(np.repeat("", n_run))

ax[0].vlines(t[3]+0.5, 0.5, 4,colors='black', linestyles='dashed', zorder=3)
ax[1].vlines(t[4]+0.004, 0.5, 4,colors='black', linestyles='dashed', zorder=3)
ax[2].vlines(t[2]+0.05, 0.5, 4,colors='black', linestyles='dashed', zorder=3)

#ax.grid()
ax[0].set_title(r'$\mathit{d}_\infty=1, \Delta= 1$', fontsize=8)
ax[1].set_title(r'$\mathit{d}_\infty>1, \Delta= 1$', fontsize=8)
ax[2].set_title(r'$\mathit{d}_\infty=1, \Delta= 1.5$', fontsize=8)

ax[0].set(ylabel=r'$\epsilon_{PEHE}$')
ax[1].set(ylabel=r'$\epsilon_{PEHE}$')
ax[2].set(ylabel=r'$\epsilon_{PEHE}$', xlabel=r'$n^{Unc}$')

fig.tight_layout()
for a in ax.flat:
    a.set(ylim = (0.5, 2.25))
    a.grid(linewidth=1, ls='--')

line1 = Line2D([0,1],[0,1],linestyle='-', color='black')
line3 = Line2D([0,1],[0,1],linestyle='-', color='b')

fig.legend([line1, line3],[r'$\tau_{CorNet}$',r'$\tau_{Conf}$'], bbox_to_anchor=(0.5, -0.075),loc = 'lower center', ncol=3)

#fig.savefig("sim_results_condition_conf.pdf", dpi=300, bbox_inches='tight')
plt.show()






