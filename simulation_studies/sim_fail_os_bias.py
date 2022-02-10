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
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib


#-------------------------------------------------
# Study F2: failure study when OS is not confouded
#-------------------------------------------------

# Define \phi and h^u, h^c
n_cov = 25
n_hidden_cov = 25
sparse_ratio = 0.666
sparse_ratio_2 = 0.666
alpha = 10
beta = 1
beta_2 = 2
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
n_test = 10000
#Confounded data (large)
x_test = torch.Tensor(np.random.multivariate_normal(np.zeros(n_cov), np.identity(n_cov), size= [n_test]))
print(((net.predict_delta(x_test) - net.predict_naive(x_test))**2).mean())
bias = ((net.predict_delta(x_test) - net.predict_naive(x_test))**2).mean()

net.delta_1.weight = torch.nn.Parameter(beta * net.delta_1.weight/torch.sqrt(bias))
net.delta_1.bias = torch.nn.Parameter(beta * net.delta_1.bias/torch.sqrt(bias))

net.delta_0.weight = torch.nn.Parameter(beta * net.delta_0.weight/torch.sqrt(bias))
net.delta_0.bias = torch.nn.Parameter(beta * net.delta_0.bias/torch.sqrt(bias))

print(((net.predict_delta(x_test) - net.predict_naive(x_test))**2).mean())



cate_trans_bias_l, cate_trans_small_bias_l, cate_trans_no_bias_l = [], [], []
cate_unc_bias_l, cate_unc_small_bias_l, cate_unc_no_bias_l = [], [], []
cate_conf_bias_l, cate_conf_small_bias_l, cate_conf_no_bias_l = [], [], []

for i in range(10):
    # Simulate data
    n_conf = 500
    n_unc = 50
    sim_dict2 = sim_data_n_conf(net, n_conf, n_conf, n_cov)
    sim_dict = sim_data_bias(net, beta_2, sparse_ratio_2, n_hidden_cov, n_conf, n_unc, n_cov)
    
    
    #Unconfounded data
    x_unc = sim_dict['x_unc']
    t_unc = sim_dict['t_unc']
    y_unc = sim_dict['y_unc']
    
    #Unconfounded data  - confounded with beta_2
    x_conf_bias = sim_dict['x_conf_bias']
    t_conf_bias = sim_dict['t_conf_bias']
    y_conf_bias = sim_dict['y_conf_bias']
    
    #Confounded data - confounded with beta
    x_conf_small_bias = sim_dict['x_conf']
    t_conf_small_bias = sim_dict['t_conf']
    y_conf_small_bias = sim_dict['y_conf']
    
    #Test data
    x_test = sim_dict['x_test']
    cate_test = sim_dict['cate_test']
    
    #Unconfounded data - confounded with beta_3
    x_conf_no_bias = sim_dict2['x_unc']
    t_conf_no_bias = sim_dict2['t_unc']
    y_conf_no_bias = sim_dict2['y_unc']
    
    #Validation set
    x_val, t_val, y_val = [], [], []

    #Parameter for the net
    n_hidden = 3
    d_hidden = n_hidden_cov
    n_out = 0
    d_out = d_hidden
    
    batch_size = x_conf_bias.size(0)
    lambd_h_c = 0.1
    lambd_delta = 0
    
    #Parameters for training
    n_epochs_feature = 500
    n_epochs_retrain = 200
    lr_train = 0.1
    lr_ret = 0.1
    
    cate_trans_bias, cate_trans_small_bias, cate_trans_no_bias = [], [], []
    cate_unc_bias, cate_unc_small_bias, cate_unc_no_bias = [], [], []
    cate_conf_bias, cate_conf_small_bias, cate_conf_no_bias = [], [], []
    
    n_conf_sizes = np.linspace(25, n_conf, n_run)
    for i in range(n_run):
        n = int(n_conf_sizes[i])
        print('###################     Training CorNet on ', n,' samples.     ####################')
        x_c_bias, t_c_bias, y_c_bias = x_conf_bias[:n,:], t_conf_bias[:n,:], y_conf_bias[:n,:]
        x_uc, t_uc, y_uc = x_unc[:m,:], t_unc[:m,:], y_unc[:m,:]
        x_c_small_bias, t_c_small_bias, y_c_small_bias = x_conf_small_bias[:n,:], t_conf_small_bias[:n,:], y_conf_small_bias[:n,:]
        x_c_no_bias, t_c_no_bias, y_c_no_bias = x_conf_no_bias[:n,:], t_conf_no_bias[:n,:], y_conf_no_bias[:n,:]
        batch_size = x_c_bias.shape[0]
        
        #On unc data only
        #Train both networks on confounded data (Step 1)
        tn = CorNet(n_cov, n_hidden, d_hidden, n_out, d_out)
        tn._train(x_uc, y_uc, t_uc, x_val, y_val, t_val, lambd_h_c, batch_size, lr_train, n_epochs_feature)
        
        #Prediction of CATE on test data
        for i in range(n_run):
            cate_unc_bias.append(tn.predict_naive(x_test))
            cate_unc_small_bias.append(tn.predict_naive(x_test))
            cate_unc_no_bias.append(tn.predict_naive(x_test))

        #WITHOUT regularizations

        #Train both networks on confounded data (Step 1)
        tn_bias = CorNet(n_cov, n_hidden, d_hidden, n_out, d_out)
        tn_bias._train(x_c_bias, y_c_bias, t_c_bias, x_val, y_val, t_val, lambd_h_c, batch_size, lr_train, n_epochs_feature)
        
        #Copy \hat{\phi}
        tn_small_bias = CorNet(n_cov, n_hidden, d_hidden, n_out, d_out)
        tn_small_bias._train(x_c_small_bias, y_c_small_bias, t_c_small_bias, x_val, y_val, t_val, lambd_h_c, batch_size, lr_train, n_epochs_feature)
        
        #Copy \hat{\phi}
        tn_no_bias = CorNet(n_cov, n_hidden, d_hidden, n_out, d_out)
        tn_no_bias._train(x_c_no_bias, y_c_no_bias, t_c_no_bias, x_val, y_val, t_val, lambd_h_c, batch_size, lr_train, n_epochs_feature)
        
        #Train head regulaized on uncofounded data (Step 2) WITHOUT covariate shift
        tn_bias._retrain_delta(x_uc, y_uc, t_uc, lambd_delta, lr_ret, n_epochs_retrain)
    
        #Train head regulaized on uncofounded data (Step 2) WITH covariate shift
        tn_small_bias._retrain_delta(x_uc, y_uc, t_uc, lambd_delta, lr_ret, n_epochs_retrain)
        
        #Train head regulaized on uncofounded data (Step 2) WITH covariate shift
        tn_no_bias._retrain_delta(x_uc, y_uc, t_uc, lambd_delta, lr_ret, n_epochs_retrain)
        
        #Prediction of CATE on test data
        cate_trans_bias.append(tn_bias.predict_delta(x_test))
        cate_trans_small_bias.append(tn_small_bias.predict_delta(x_test))
        cate_trans_no_bias.append(tn_no_bias.predict_delta(x_test))
        #Conf
        cate_conf_bias.append(tn_bias.predict_naive(x_test))
        cate_conf_small_bias.append(tn_small_bias.predict_naive(x_test))
        cate_conf_no_bias.append(tn_no_bias.predict_naive(x_test))
        


    cate_trans_bias_l.append(np.array([((cate_test - cate_trans_bias[i].numpy())**2).mean() for i in range(n_run)]))
    cate_trans_small_bias_l.append(np.array([((cate_test - cate_trans_small_bias[i].numpy())**2).mean() for i in range(n_run)]))
    cate_trans_no_bias_l.append(np.array([((cate_test - cate_trans_no_bias[i].numpy())**2).mean() for i in range(n_run)]))
    
    cate_unc_bias_l.append(np.array([((cate_test - cate_unc_bias[i].numpy())**2).mean() for i in range(n_run)]))
    cate_unc_small_bias_l.append(np.array([((cate_test - cate_unc_small_bias[i].numpy())**2).mean() for i in range(n_run)]))
    cate_unc_no_bias_l.append(np.array([((cate_test - cate_unc_no_bias[i].numpy())**2).mean() for i in range(n_run)]))
    
    cate_conf_bias_l.append(np.array([((cate_test - cate_conf_bias[i].numpy())**2).mean() for i in range(n_run)]))
    cate_conf_small_bias_l.append(np.array([((cate_test - cate_conf_small_bias[i].numpy())**2).mean() for i in range(n_run)]))
    cate_conf_no_bias_l.append(np.array([((cate_test - cate_conf_no_bias[i].numpy())**2).mean() for i in range(n_run)]))



#%%


# Data for plotting
cate_trans1 = np.sqrt(cate_trans_bias_l)
cate_unc1 = np.sqrt(cate_unc_bias_l)
cate_conf1 = np.sqrt(cate_conf_bias_l)

cate_trans2 = np.sqrt(cate_trans_small_bias_l)
cate_unc2 = np.sqrt(cate_unc_small_bias_l)
cate_conf2 = np.sqrt(cate_conf_small_bias_l)

cate_trans3 = np.sqrt(cate_trans_no_bias_l)
cate_unc3 = np.sqrt(cate_unc_no_bias_l)
cate_conf3 = np.sqrt(cate_conf_no_bias_l)



# =============================================================================
# #Save data
# np.save('sim_fail_os_bias_trans1', cate_trans1)
# np.save('sim_fail_os_bias_unc1', cate_unc1)
# np.save('sim_fail_os_bias_conf1', cate_conf1)
# np.save('sim_fail_os_bias_trans2', cate_trans2)
# np.save('sim_fail_os_bias_unc2', cate_unc2)
# np.save('sim_fail_os_bias_conf2', cate_conf2)
# np.save('sim_fail_os_bias_trans3', cate_trans3)
# np.save('sim_fail_os_bias_unc3', cate_unc3)
# np.save('sim_fail_os_bias_conf3', cate_conf3)
# 
# =============================================================================



# =============================================================================
# #Load data
# cate_trans1 = np.load('sim_fail_os_bias_trans1.npy')
# cate_unc1 = np.load('sim_fail_os_bias_unc1.npy')
# cate_conf1 = np.load('sim_fail_os_bias_conf1.npy')
# cate_trans2 = np.load('sim_fail_os_bias_trans2.npy')
# cate_unc2 = np.load('sim_fail_os_bias_unc2.npy')
# cate_conf2 = np.load('sim_fail_os_bias_conf2.npy')
# cate_trans3 = np.load('sim_fail_os_bias_trans3.npy')
# cate_unc3 = np.load('sim_fail_os_bias_unc3.npy')
# cate_conf3 = np.load('sim_fail_os_bias_conf3.npy')
# 
# =============================================================================


matplotlib.rcParams.update({'font.size': 12})
# Data for plotting
n_conf = 500
n_conf_sizes = np.linspace(25, n_conf, n_run)
t = n_conf_sizes
cate_t = np.array(cate_trans1).mean(0)
cate_u = np.array(cate_unc1).mean(0)
cate_c = np.array(cate_conf1).mean(0)

cate_t2 = np.array(cate_trans2).mean(0)
cate_u2 = np.array(cate_unc2).mean(0)
cate_c2 = np.array(cate_conf2).mean(0)

cate_t3 = np.array(cate_trans2).mean(0)
cate_u3 = np.array(cate_unc3).mean(0)
cate_c3 = np.array(cate_conf3).mean(0)


ms = 3
lw=1
elw=0.5
cs=2

fig, ax = plt.subplots(1,3)
fig.set_figwidth(12.5)
fig.set_figheight(5)
ax[0].errorbar(t, cate_t, np.array(cate_trans1).std(0), mfc='black', color = 'black', ecolor = 'black', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)
ax[0].errorbar(t, cate_c, np.array(cate_conf1).std(0), mfc='b', color = 'b', ecolor = 'b', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)
ax[0].errorbar(t, cate_u, np.array(cate_unc1).std(0), mfc='r', color = 'r', ecolor = 'r', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)
ax[0].xaxis.set_ticks(np.array([25, 200, 400]))
ax[0].yaxis.set_ticks(np.array([1, 2, 3]))
ax[0].set(xlabel='$n^{Conf}$', ylabel=r'$\sqrt{\epsilon_{PEHE}}$',
       title=r'$\Delta = 4$ (i.e., heavily conf.)', ylim=(0.5, 3.))
ax[0].grid(linewidth=1, ls='--')



ax[1].errorbar(t, cate_t2, np.array(cate_trans2).std(0),  mfc='black', color = 'black', ecolor = 'black', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)
ax[1].errorbar(t, cate_c2, np.array(cate_conf2).std(0), mfc='b', color = 'b', ecolor = 'b', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)
ax[1].errorbar(t, cate_u2, np.array(cate_unc2).std(0), mfc='r', color = 'r', ecolor = 'r', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)
ax[1].xaxis.set_ticks(np.array([25, 200, 400]))
ax[1].yaxis.set_ticks(np.array([1, 2, 3]))
ax[1].set(xlabel='$n^{Conf}$',
       title=r'$\Delta = 1$ (i.e., moderately conf.)', ylim=(0.5, 3.))
ax[1].grid(linewidth=1, ls='--')

ax[2].errorbar(t, cate_t3, np.array(cate_trans3).std(0),  mfc='black', color = 'black', ecolor = 'black', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)
ax[2].errorbar(t, cate_c3, np.array(cate_conf3).std(0), mfc='b', color = 'b', ecolor = 'b', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)
ax[2].errorbar(t, cate_u3, np.array(cate_unc3).std(0), mfc='r', color = 'r', ecolor = 'r', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)
ax[2].xaxis.set_ticks(np.array([25, 200, 400]))
ax[2].yaxis.set_ticks(np.array([1, 2, 3]))
ax[2].set(xlabel='$n^{Conf}$',
       title=r'$\Delta = 0$ (i.e., unconfounded)', ylim=(0.5, 3.))
ax[2].grid(linewidth=1, ls='--')


line1 = Line2D([0,1],[0,1],linestyle='-', color='black')
line2 = Line2D([0,1],[0,1],linestyle='-', color='b')
line3 = Line2D([0,1],[0,1],linestyle='-', color='r')

fig.legend([line1, line2, line3], [r'$\tau_{CorNet}$', r'$\tau_{Conf}$', r'$\tau_{Unc}$'], bbox_to_anchor=(0.5, -0.075), loc = 'lower center', ncol=3, numpoints=1)
#fig.savefig("sim_fail_os_bias.pdf", dpi=300, bbox_inches='tight')
plt.show()

