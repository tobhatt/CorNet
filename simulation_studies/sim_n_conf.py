from IPython import get_ipython
get_ipython().magic('reset -sf')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
from simulation_studies.simdata import *
from model.models import *
from scipy.sparse import random
import seaborn as sns
from matplotlib.lines import Line2D

#-------------------------------------------
# Study 1: Benefit of integrating obs data
#-------------------------------------------
'''
Set d_\infty = 1 (no covariate shift) and fix n_unc, increase n_conf.
Compare nn on n_unc and our algorithm on n_unc + n_conf (2-step)
'''

# Define \phi and h^u, h^c
n_cov = 25
n_hidden_cov = 25
sparse_ratio = 0.666
alpha = 10
beta = 2
n_run=10

m1 = 25
m2 = 50
m3 = 100
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
sim_dict = sim_data_n_conf(net, 1000, 50, n_cov = n_cov, cov_shift = 0)  
x_test = sim_dict['x_test']
print(((net.predict_delta(x_test) - net.predict_naive(x_test))**2).mean())
bias = ((net.predict_delta(x_test) - net.predict_naive(x_test))**2).mean()

net.delta_1.weight = torch.nn.Parameter(beta * net.delta_1.weight/torch.sqrt(bias))
net.delta_1.bias = torch.nn.Parameter(beta * net.delta_1.bias/torch.sqrt(bias))

net.delta_0.weight = torch.nn.Parameter(beta * net.delta_0.weight/torch.sqrt(bias))
net.delta_0.bias = torch.nn.Parameter(beta * net.delta_0.bias/torch.sqrt(bias))

print(((net.predict_delta(x_test) - net.predict_naive(x_test))**2).mean())

cate_trans_l, cate_trans_reg_l, cate_unc_l, cate_conf_l = [], [], [], []
cate_trans_m, cate_trans_reg_m, cate_unc_m, cate_conf_m = [], [], [], []
cate_trans_m2, cate_trans_reg_m2, cate_unc_m2, cate_conf_m2 = [], [], [], []

for i in range(10):
    # Simulate data
    n_conf = 500
    n_unc = 1000
    sim_dict = sim_data_n_conf(net, n_conf, n_unc, n_cov = n_cov, cov_shift = 0)    
    
    #Unconfounded data
    x_unc = sim_dict['x_unc']
    t_unc = sim_dict['t_unc']
    y_unc = sim_dict['y_unc']
    
    #Confounded data
    x_conf = sim_dict['x_conf']
    t_conf = sim_dict['t_conf']
    y_conf = sim_dict['y_conf']
    
    #Validation set
    x_val, t_val, y_val = [], [], []
    
    #Test data
    x_test = sim_dict['x_test']
    cate_test = sim_dict['cate_test']
    
    print(((net.predict_delta(x_test) - net.predict_naive(x_test))**2).mean())
    
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
    
    #n_unc small
    cate_trans = []
    cate_conf = []
    n_conf_sizes = np.linspace(25, n_conf, n_run)
    for i in range(n_run):
        n = int(n_conf_sizes[i])#int((i)*n_conf/n_run)+60
        print('###################     Training CorNet on ', n,' samples.     ####################')
        x_c, t_c, y_c = x_conf[:n,:], t_conf[:n,:], y_conf[:n,:]
        x_uc, t_uc, y_uc = x_unc[:m1,:], t_unc[:m1,:], y_unc[:m1,:]
        batch_size = x_c.shape[0]
        ### Variante: Shared representation between treatment and control group w/o regularization
        tn = CorNet(n_cov, n_hidden, d_hidden, n_out, d_out)
        #Train both networks on confounded data (Step 1)
        tn._train(x_c, y_c, t_c, x_val, y_val, t_val, lambd_h_c, batch_size, lr_train, n_epochs_feature)
        
        #Train head regulaized on uncofounded data (Step 2)
        tn._retrain_delta(x_uc, y_uc, t_uc, lambd_delta, lr_ret, n_epochs_retrain)
        
        #Prediction of CATE on test data
        cate_trans.append(tn.predict_delta(x_test))
        cate_conf.append(tn.predict_naive(x_test))

    cate_trans_l.append(np.array([((cate_test - cate_trans[i].numpy())**2).mean() for i in range(n_run)]))
    cate_conf_l.append(np.array([((cate_test - cate_conf[i].numpy())**2).mean() for i in range(n_run)]))

    #n_unc larger
    cate_trans = []
    cate_conf = []
    for i in range(n_run):
        n = int(n_conf_sizes[i])
        print('###################     Training CorNet on ', n,' samples.     ####################')
        x_c, t_c, y_c = x_conf[:n,:], t_conf[:n,:], y_conf[:n,:]
        x_uc, t_uc, y_uc = x_unc[:m2,:], t_unc[:m2,:], y_unc[:m2,:]
        batch_size = x_c.shape[0]
        ### Variante: Shared representation between treatment and control group w/o regularization
        tn = CorNet(n_cov, n_hidden, d_hidden, n_out, d_out)
        #Train both networks on confounded data (Step 1)
        tn._train(x_c, y_c, t_c, x_val, y_val, t_val, lambd_h_c, batch_size, lr_train, n_epochs_feature)
        
        #Train head regulaized on uncofounded data (Step 2)
        tn._retrain_delta(x_uc, y_uc, t_uc, lambd_delta, lr_ret, n_epochs_retrain)
        
        #Prediction of CATE on test data
        cate_trans.append(tn.predict_delta(x_test))
        cate_conf.append(tn.predict_naive(x_test))
        
    cate_trans_m.append(np.array([((cate_test - cate_trans[i].numpy())**2).mean() for i in range(n_run)]))
    cate_conf_m.append(np.array([((cate_test - cate_conf[i].numpy())**2).mean() for i in range(n_run)]))


    #n_unc larger
    cate_trans = []
    cate_conf = []
    for i in range(n_run):
        n = int(n_conf_sizes[i])
        print('###################     Training CorNet on ', n,' samples.     ####################')
        x_c, t_c, y_c = x_conf[:n,:], t_conf[:n,:], y_conf[:n,:]
        x_uc, t_uc, y_uc = x_unc[:m3,:], t_unc[:m3,:], y_unc[:m3,:]
        batch_size = x_c.shape[0]
        ### Variante: Shared representation between treatment and control group w/o regularization
        tn = CorNet(n_cov, n_hidden, d_hidden, n_out, d_out)
        #Train both networks on confounded data (Step 1)
        tn._train(x_c, y_c, t_c, x_val, y_val, t_val, lambd_h_c, batch_size, lr_train, n_epochs_feature)
        
        #Train head regulaized on uncofounded data (Step 2)
        tn._retrain_delta(x_uc, y_uc, t_uc, lambd_delta, lr_ret, n_epochs_retrain)
        
        #Prediction of CATE on test data
        cate_trans.append(tn.predict_delta(x_test))
        cate_conf.append(tn.predict_naive(x_test))
        
    cate_trans_m2.append(np.array([((cate_test - cate_trans[i].numpy())**2).mean() for i in range(n_run)]))
    cate_conf_m2.append(np.array([((cate_test - cate_conf[i].numpy())**2).mean() for i in range(n_run)]))


    #Train \tau_unc
    #n_unc small
    cate_unc = []

    x_uc, t_uc, y_uc = x_unc[:m1,:], t_unc[:m1,:], y_unc[:m1,:]
    batch_size = x_uc.shape[0]
    ### Variante: Shared representation between treatment and control group w/o regularization
    tarnet_unc = CorNet(n_cov, n_hidden, d_hidden, n_out, d_out)
    #Train both networks on confounded data (Step 1)
    tarnet_unc._train(x_uc, y_uc, t_uc, x_val, y_val, t_val, lambd_h_c, batch_size, lr_train, n_epochs_feature)
       
    #Prediction of CATE on test data
    for i in range(n_run):
        cate_unc.append(tarnet_unc.predict_naive(x_test))
    
    cate_unc_l.append(np.array([((cate_test - cate_unc[i].numpy())**2).mean() for i in range(n_run)]))
     
    #n_unc larger
    cate_unc = []
    x_uc, t_uc, y_uc = x_unc[:m2,:], t_unc[:m2,:], y_unc[:m2,:]
    batch_size = x_uc.shape[0]
    ### Variante: Shared representation between treatment and control group w/o regularization
    tarnet_unc = CorNet(n_cov, n_hidden, d_hidden, n_out, d_out)
    #Train both networks on confounded data (Step 1)
    tarnet_unc._train(x_uc, y_uc, t_uc, x_val, y_val, t_val, lambd_h_c, batch_size, lr_train, n_epochs_feature)
                    
    #Prediction of CATE on test data
    for i in range(n_run):
        cate_unc.append(tarnet_unc.predict_naive(x_test))

    cate_unc_m.append(np.array([((cate_test - cate_unc[i].numpy())**2).mean() for i in range(n_run)]))
     
 
    #n_unc larger
    cate_unc = []
    x_uc, t_uc, y_uc = x_unc[:m3,:], t_unc[:m3,:], y_unc[:m3,:]
    batch_size = x_uc.shape[0]
    ### Variante: Shared representation between treatment and control group w/o regularization
    tarnet_unc = CorNet(n_cov, n_hidden, d_hidden, n_out, d_out)
    #Train both networks on confounded data (Step 1)
    tarnet_unc._train(x_uc, y_uc, t_uc, x_val, y_val, t_val, lambd_h_c, batch_size, lr_train, n_epochs_feature)
                    
    #Prediction of CATE on test data
    for i in range(n_run):
        cate_unc.append(tarnet_unc.predict_naive(x_test))

    cate_unc_m2.append(np.array([((cate_test - cate_unc[i].numpy())**2).mean() for i in range(n_run)]))    
 
    
cate_trans1 = np.sqrt(cate_trans_l)
cate_unc1 = np.sqrt(cate_unc_l)
cate_conf1 = np.sqrt(cate_conf_l)
cate_trans2 = np.sqrt(cate_trans_m)
cate_unc2 = np.sqrt(cate_unc_m)
cate_conf2 = np.sqrt(cate_conf_m)
cate_trans3 = np.sqrt(cate_trans_m2)
cate_unc3 = np.sqrt(cate_unc_m2)
cate_conf3 = np.sqrt(cate_conf_m2)


#%%

matplotlib.rcParams.update({'font.size': 12})

# =============================================================================
# #Save data
# np.save('sim_n_conf_trans1', cate_trans1)
# np.save('sim_n_conf_unc1', cate_unc1)
# np.save('sim_n_conf_conf1', cate_conf1)
# np.save('sim_n_conf_trans2', cate_trans2)
# np.save('sim_n_conf_unc2', cate_unc2)
# np.save('sim_n_conf_conf2', cate_conf2)
# np.save('sim_n_conf_trans3', cate_trans3)
# np.save('sim_n_conf_unc3', cate_unc3)
# np.save('sim_n_conf_conf3', cate_conf3)
# =============================================================================



#Load data
cate_trans1 = np.load('sim_n_conf_trans1.npy')
cate_unc1 = np.load('sim_n_conf_unc1.npy')
cate_conf1 = np.load('sim_n_conf_conf1.npy')
cate_trans2 = np.load('sim_n_conf_trans2.npy')
cate_unc2 = np.load('sim_n_conf_unc2.npy')
cate_conf2 = np.load('sim_n_conf_conf2.npy')
cate_trans3 = np.load('sim_n_conf_trans3.npy')
cate_unc3 = np.load('sim_n_conf_unc3.npy')
cate_conf3 = np.load('sim_n_conf_conf3.npy')


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
cate_t3 = np.array(cate_trans3).mean(0)
cate_u3 = np.array(cate_unc3).mean(0)
cate_c3 = np.array(cate_conf3).mean(0)

ms = 3
lw=1
elw=0.5
cs=2

fig, ax = plt.subplots(1,3)
fig.set_figwidth(12.5)
fig.set_figheight(5)
#ax.plot(t, cate_t, 'b', linewidth=2)
ax[0].errorbar(t, cate_t, np.array(cate_trans1).std(0), mfc='black', color = 'black', ecolor = 'black', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)
ax[0].errorbar(t, cate_c, np.array(cate_conf1).std(0), mfc='b', color = 'b', ecolor = 'b', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)
ax[0].errorbar(t, cate_u, np.array(cate_unc1).std(0), mfc='r', color = 'r', ecolor = 'r', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)
ax[0].xaxis.set_ticks(np.array([25, 200, 400]))
ax[0].yaxis.set_ticks(np.array([1, 2, 3]))
ax[0].set(xlabel='$n^{Conf}$', ylabel=r'$\sqrt{\epsilon_{PEHE}}$',
       title='$n^{Unc}=25$', ylim=(0.5, 3.5))
ax[0].grid(linewidth=1, ls='--')


#fig, ax = plt.subplots()
ax[1].errorbar(t, cate_t2, np.array(cate_trans2).std(0),  mfc='black', color = 'black', ecolor = 'black', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)
ax[1].errorbar(t, cate_c2, np.array(cate_conf2).std(0), mfc='b', color = 'b', ecolor = 'b', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)
ax[1].errorbar(t, cate_u2, np.array(cate_unc2).std(0), mfc='r', color = 'r', ecolor = 'r', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)
ax[1].xaxis.set_ticks(np.array([25, 200, 400]))
ax[1].yaxis.set_ticks(np.array([1, 2, 3]))
ax[1].set(xlabel='$n^{Conf}$',
       title='$n^{Unc}=50$', ylim=(0.5, 3.5))
ax[1].grid(linewidth=1, ls='--')


#fig, ax = plt.subplots()
ax[2].errorbar(t, cate_t3, np.array(cate_trans3).std(0),  mfc='black', color = 'black', ecolor = 'black', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)
ax[2].errorbar(t, cate_c3, np.array(cate_conf3).std(0), mfc='b', color = 'b', ecolor = 'b', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)
ax[2].errorbar(t, cate_u3, np.array(cate_unc3).std(0), mfc='r', color = 'r', ecolor = 'r', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)
ax[2].xaxis.set_ticks(np.array([25, 200, 400]))
ax[2].yaxis.set_ticks(np.array([1, 2, 3]))
ax[2].set(xlabel='$n^{Conf}$',
       title='$n^{Unc}=100$', ylim=(0.5, 3.5))
ax[2].grid(linewidth=1, ls='--')

line1 = Line2D([0,1],[0,1],linestyle='-', color='black')
line2 = Line2D([0,1],[0,1],linestyle='-', color='b')
line3 = Line2D([0,1],[0,1],linestyle='-', color='r')

fig.legend([line1, line2, line3], [r'$\tau_{CorNet}$', r'$\tau_{Conf}$', r'$\tau_{Unc}$'], bbox_to_anchor=(0.5, -0.075), loc = 'lower center', ncol=3, numpoints=1)
#fig.savefig("sim_results_n_conf.pdf", dpi=300, bbox_inches='tight')
plt.show()
