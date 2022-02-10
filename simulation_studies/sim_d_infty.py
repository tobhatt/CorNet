from IPython import get_ipython
get_ipython().magic('reset -sf')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pathlib import Path
from simulation_studies.simdata import *
from model.models import *
from scipy.sparse import random
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib
#-------------------------------------------
# Study 2: Impact of d_\infty
#-------------------------------------------
'''
Fix n_conf, increase n_unc.
Compare our algorithm on n_unc + n_conf (2-step) for d_\infty = 1 and d_\infty > 1
'''

# Define \phi and h^u, h^c
n_cov = 25
n_hidden_cov = 25
sparse_ratio = 0.666
alpha = 10
beta = 2
n_run=10

m=50

#Net with weights
net = CorNet(n_cov, 3, n_hidden_cov, 0, n_hidden_cov)

#normalize outcome weights to || ...|| = alpha
l2_1 = torch.norm(torch.Tensor(net.linear_out_1.weight.detach()), p=2)
l2_0 = torch.norm(torch.Tensor(net.linear_out_0.weight.detach()), p=2)
net.linear_out_1.weight = torch.nn.Parameter(alpha * net.linear_out_1.weight.detach()/(l2_1))
net.linear_out_0.weight = torch.nn.Parameter(alpha * net.linear_out_0.weight.detach()/(l2_0))

#Sparse \delta 
delta1 = net.delta_1.weight.detach()
delta0 = net.delta_0.weight.detach()
sparse_index = np.random.choice(range(1,n_hidden_cov), size=int(sparse_ratio * n_hidden_cov), replace = False)
delta1.detach().numpy()[:,sparse_index] = 0
sparse_index = np.random.choice(range(1,n_hidden_cov), size=int(sparse_ratio * n_hidden_cov), replace = False)
delta0.detach().numpy()[:,sparse_index] = 0
net.delta_1.weight = torch.nn.Parameter(delta1)
net.delta_0.weight = torch.nn.Parameter(delta0)


#normalize to beta
sim_dict = sim_data_d(net, 10000, 50, n_cov, cov_shift=0) 
x_test = sim_dict['x_test_cov']
print(((net.predict_delta(x_test) - net.predict_naive(x_test))**2).mean())
bias = ((net.predict_delta(x_test) - net.predict_naive(x_test))**2).mean()


net.delta_1.weight = torch.nn.Parameter(beta * net.delta_1.weight/torch.sqrt(bias))
net.delta_1.bias = torch.nn.Parameter(beta * net.delta_1.bias/torch.sqrt(bias))

net.delta_0.weight = torch.nn.Parameter(beta * net.delta_0.weight/torch.sqrt(bias))
net.delta_0.bias = torch.nn.Parameter(beta * net.delta_0.bias/torch.sqrt(bias))

print(((net.predict_delta(x_test) - net.predict_naive(x_test))**2).mean())


cate_trans_l, cate_trans_reg_l, cate_unc_l, cate_conf_l = [], [], [], []
cate_trans_cov_l, cate_trans_cov_reg_l, cate_unc_cov_l, cate_conf_cov_l = [], [], [], []

cate_trans_m, cate_trans_reg_m, cate_unc_m, cate_conf_m  = [], [], [], []
cate_trans_cov_m, cate_trans_cov_reg_m, cate_unc_cov_m, cate_conf_cov_m = [], [], [], []

for i in range(10):
    # Simulate data
    n_conf = 500
    n_unc = 100
    sim_dict = sim_data_d3(net, n_conf, n_unc, n_cov, cov_shift=0)     

    #Unconfounded data
    x_unc = sim_dict['x_unc']
    t_unc = sim_dict['t_unc']
    y_unc = sim_dict['y_unc']
    
    #Unconfounded data WITH covariate shift
    x_unc_cov = sim_dict['x_unc_cov']
    t_unc_cov = sim_dict['t_unc_cov']
    y_unc_cov = sim_dict['y_unc_cov']
    
    #Confounded data
    x_conf = sim_dict['x_conf']
    t_conf = sim_dict['t_conf']
    y_conf = sim_dict['y_conf']

    #Validation set
    x_val, t_val, y_val = [], [], []
    
    #Test data
    x_test = sim_dict['x_test']
    cate_test = sim_dict['cate_test']
    
    #Parameter for the net
    n_hidden = 3
    d_hidden = n_hidden_cov
    n_out = 0
    d_out = d_hidden
    
    batch_size = x_conf.size(0)
    lambd_h_c = 0.1
    lambd_delta = 0
    
    #Parameters for training
    n_epochs_feature = 500
    n_epochs_retrain = 200
    lr_train = 0.1
    lr_ret = 0.1
    
    
    cate_unc, cate_unc_cov = [], []
    cate_conf, cate_conf_cov = [], []
    cate_trans = []
    cate_trans_cov = []
    cate_trans_reg = []
    cate_trans_cov_reg = []
    
    n_conf_sizes = np.linspace(25, n_conf, n_run)
    for i in range(n_run):
        n = int(n_conf_sizes[i])
        print('###################     Training CorNet on ', n,' samples.     ####################')
        x_c, t_c, y_c = x_conf[:n,:], t_conf[:n,:], y_conf[:n,:]
        x_uc, t_uc, y_uc = x_unc[:m,:], t_unc[:m,:], y_unc[:m,:]
        x_uc_cov, t_uc_cov, y_uc_cov = x_unc_cov[:m,:], t_unc_cov[:m,:], y_unc_cov[:m,:]
        batch_size = x_c.shape[0]
  
    
        #On unc data only
        
        #Train both networks on confounded data (Step 1)
        tn = CorNet(n_cov, n_hidden, d_hidden, n_out, d_out)
        tn._train(x_uc, y_uc, t_uc, x_val, y_val, t_val, lambd_h_c, batch_size, lr_train, n_epochs_feature)

        #Train both networks on confounded data (Step 1)
        tn_cov = CorNet(n_cov, n_hidden, d_hidden, n_out, d_out)
        tn_cov._train(x_uc_cov, y_uc_cov, t_uc_cov, x_val, y_val, t_val, lambd_h_c, batch_size, lr_train, n_epochs_feature)
        
        
        #Copy \hat{\phi}
        #Prediction of CATE on test data
        for i in range(n_run):
            cate_unc.append(tn.predict_naive(x_test))
            cate_unc_cov.append(tn_cov.predict_naive(x_test))
    
    
        #WITHOUT regularizations
        #Train both networks on confounded data (Step 1)
        tn = CorNet(n_cov, n_hidden, d_hidden, n_out, d_out)
        tn._train(x_c, y_c, t_c, x_val, y_val, t_val, lambd_h_c, batch_size, lr_train, n_epochs_feature)
        
        #Copy \hat{\phi}
        tn_cov = deepcopy(tn)
        
        #Train head regulaized on uncofounded data (Step 2) WITHOUT covariate shift
        tn._retrain_delta(x_uc, y_uc, t_uc, lambd_delta, lr_ret, n_epochs_retrain)
    
        #Train head regulaized on uncofounded data (Step 2) WITH covariate shift
        tn_cov._retrain_delta(x_uc_cov, y_uc_cov, t_uc_cov, lambd_delta, lr_ret, n_epochs_retrain)
        
        #Prediction of CATE on test data
        cate_trans.append(tn.predict_delta(x_test))
        cate_trans_cov.append(tn_cov.predict_delta(x_test))
        cate_conf.append(tn.predict_naive(x_test))
        cate_conf_cov.append(tn_cov.predict_naive(x_test))


    cate_trans_l.append(np.array([((cate_test - cate_trans[i].numpy())**2).mean() for i in range(n_run)]))
    cate_trans_cov_l.append(np.array([((cate_test - cate_trans_cov[i].numpy())**2).mean() for i in range(n_run)]))
    cate_unc_l.append(np.array([((cate_test - cate_unc[i].numpy())**2).mean() for i in range(n_run)]))
    cate_unc_cov_l.append(np.array([((cate_test - cate_unc_cov[i].numpy())**2).mean() for i in range(n_run)]))
    cate_conf_l.append(np.array([((cate_test - cate_conf[i].numpy())**2).mean() for i in range(n_run)]))
    cate_conf_cov_l.append(np.array([((cate_test - cate_conf_cov[i].numpy())**2).mean() for i in range(n_run)]))




cate_trans1 = np.sqrt(cate_trans_l)
cate_unc1 = np.sqrt(cate_unc_l)
cate_conf1 = np.sqrt(cate_conf_l)
cate_trans2 = np.sqrt(cate_trans_cov_l)
cate_unc2 = np.sqrt(cate_unc_cov_l)
cate_conf2 = np.sqrt(cate_conf_cov_l)


#%%
matplotlib.rcParams.update({'font.size': 12})

# =============================================================================
# 
# #Save data
# np.save('sim_d_infty_trans1', cate_trans1)
# np.save('sim_d_infty_unc1', cate_unc1)
# np.save('sim_d_infty_conf1', cate_conf1)
# np.save('sim_d_infty_trans2', cate_trans2)
# np.save('sim_d_infty_unc2', cate_unc2)
# np.save('sim_d_infty_conf2', cate_conf2)
# 
# =============================================================================



# =============================================================================
# #Load data
# cate_trans1 = np.load('sim_d_infty_trans1.npy')
# cate_unc1 = np.load('sim_d_infty_unc1.npy')
# cate_conf1 = np.load('sim_d_infty_conf1.npy')
# cate_trans2 = np.load('sim_d_infty_trans2.npy')
# cate_unc2 = np.load('sim_d_infty_unc2.npy')
# cate_conf2 = np.load('sim_d_infty_conf2.npy')
# 
# 
# =============================================================================

# Data for plotting
n_conf=500
n_conf_sizes = np.linspace(25, n_conf, n_run)
t = n_conf_sizes
cate_t = np.array(cate_trans1).mean(0)
cate_t2 = np.array(cate_trans2).mean(0)
cate_u = np.array(cate_unc1).mean(0)
cate_u2 = np.array(cate_unc2).mean(0)
cate_c = np.array(cate_conf1).mean(0)
cate_c2 = np.array(cate_conf2).mean(0)


ms = 3
lw=1
elw=0.5
cs=2

fig, ax = plt.subplots(1,2)
fig.set_figwidth(10)
fig.set_figheight(5)
ax[0].errorbar(t, cate_t, np.array(cate_trans1).std(0), mfc='black', color = 'black', ecolor = 'black', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)
ax[0].errorbar(t, cate_c, np.array(cate_conf1).std(0), mfc='b', color = 'b', ecolor = 'b', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)
ax[0].errorbar(t, cate_u, np.array(cate_unc1).std(0), mfc='r', color = 'r', ecolor = 'r', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)
ax[0].xaxis.set_ticks(np.array([25, 200, 400]))
ax[0].yaxis.set_ticks(np.array([1, 2, 3]))
ax[0].set(xlabel='$n^{Conf}$', ylabel=r'$\sqrt{\epsilon_{PEHE}}$',
       title=r'$\mathit{d}_\infty(p^{Conf}\mid p^{Unc})=1$ (i.e., $\sigma_u=1$)', ylim=(0.5, 3.5))
ax[0].grid(linewidth=1, ls='--')


ax[1].errorbar(t, cate_t2, np.array(cate_trans2).std(0),  mfc='black', color = 'black', ecolor = 'black', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)
ax[1].errorbar(t, cate_c2, np.array(cate_conf2).std(0), mfc='b', color = 'b', ecolor = 'b', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)
ax[1].errorbar(t, cate_u2, np.array(cate_unc2).std(0), mfc='r', color = 'r', ecolor = 'r', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)
ax[1].xaxis.set_ticks(np.array([25, 200, 400]))
ax[1].yaxis.set_ticks(np.array([1, 2, 3]))
ax[1].set(xlabel='$n^{Conf}$',
       title=r'$\mathit{d}_\infty(p^{Conf}\mid p^{Unc})>1$ (i.e., $\sigma_u=0.1$)', ylim=(0.5, 3.5))
ax[1].grid(linewidth=1, ls='--')

line1 = Line2D([0,1],[0,1],linestyle='-', color='black')
line2 = Line2D([0,1],[0,1],linestyle='-', color='b')
line3 = Line2D([0,1],[0,1],linestyle='-', color='r')

fig.legend([line1, line2, line3], [r'$\tau_{CorNet}$', r'$\tau_{Conf}$', r'$\tau_{Unc}$'], bbox_to_anchor=(0.5, -0.075), loc = 'lower center', ncol=3, numpoints=1)

#fig.savefig("sim_results_dist.pdf", dpi=300, bbox_inches='tight')
plt.show()




