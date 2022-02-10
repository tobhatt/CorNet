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
sparse_ratio_2 = 0.666
alpha = 10
beta = 1#0.1
beta_2 = 2#15
cov_shift = 0 #See data simulation for the covariate shift
n_run=15

#Net with weights
net = CorNet(n_cov, 3, n_hidden_cov, 0, n_hidden_cov)

#with torch.no_grad():
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



cate_trans_l = []
cate_naive_l = []
cate_trans_cov_l = []
cate_naive_cov_l = []
cate_trans_bias_l = []
cate_naive_bias_l = []


m=50

for k in range(1):
    # Simulate data
    n_conf = 300
    n_unc = 100
    sim_dict = sim_data_condition3(net, beta_2, sparse_ratio_2, n_hidden_cov,n_conf, n_unc, n_cov, cov_shift)

    
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
    #lambd_ada = .1
    #alpha = .2
    lambd_delta = 0

    
    x_uc, t_uc, y_uc = x_unc[:m,:], t_unc[:m,:], y_unc[:m,:]
    x_uc_cov, t_uc_cov, y_uc_cov = x_unc_cov[:m,:], t_unc_cov[:m,:], y_unc_cov[:m,:]
        
    ### \tau_unc 
    nn_naive = CorNet(n_cov, n_hidden, d_hidden, n_out, d_out)
    #Train network on unconfounded data
    nn_naive._train(x_uc, y_uc, t_uc, x_val, y_val, t_val, lambd_h_c, batch_size, lr_train, n_epochs_feature)
    #Prediction of CATE on test data
    cate_naive = nn_naive.predict_naive(x_test)
    cate_naive_l.append([((cate_test - cate_naive.numpy())**2).mean() for i in range(n_run)])
    cate_naive_bias_l.append([((cate_test - cate_naive.numpy())**2).mean() for i in range(n_run)])
    
    ### \tau_unc 
    nn_naive = CorNet(n_cov, n_hidden, d_hidden, n_out, d_out)
    #Train network on unconfounded data
    nn_naive._train(x_uc_cov, y_uc_cov, t_uc_cov, x_val, y_val, t_val, lambd_h_c, batch_size, lr_train, n_epochs_feature)
    #Prediction of CATE on test data   
    cate_naive_cov = nn_naive.predict_naive(x_test)
    cate_naive_cov_l.append([((cate_test - cate_naive_cov.numpy())**2).mean() for i in range(n_run)])
    


    cate_trans = []
    cate_trans_cov = []
    cate_trans_bias = []
    #cate_trans_reg = []
    cate_trans_cov_reg = []
    #cate_trans_bias_reg = []
    
    t1 = np.concatenate((np.linspace(0.4472136, 1.5, 10), np.linspace(1.5, 2.44948974, 6)[1:]))
    n_conf_sizes = np.round(t1**2*m)
    #np.sqrt(n_conf_sizes/m )
    for i in range(n_run):
        n = int(n_conf_sizes[i])
        print('###################     Training CorNet on ', n,' samples.     ####################')
        x_c, t_c, y_c = x_conf[:n,:], t_conf[:n,:], y_conf[:n,:]
        x_c_bias, t_c_bias, y_c_bias = x_conf_bias[:n,:], t_conf_bias[:n,:], y_conf_bias[:n,:]


        ###\tau_cor
        tn = CorNet(n_cov, n_hidden, d_hidden, n_out, d_out)
        #Train both networks on confounded data (Step 1)
        tn._train(x_c, y_c, t_c, x_val, y_val, t_val, lambd_h_c, batch_size, lr_train, n_epochs_feature)
            
        tn_cov = deepcopy(tn)
        
        #Train head regulaized on uncofounded data (Step 2)
        tn._retrain_delta(x_uc, y_uc, t_uc, lambd_delta, lr_ret, n_epochs_retrain)
        
        #Prediction of CATE on test data
        cate_trans.append(tn.predict_delta(x_test))
       
        #Train head regulaized on uncofounded data (Step 2)
        tn_cov._retrain_delta(x_uc_cov, y_uc_cov, t_uc_cov, lambd_delta, lr_ret, n_epochs_retrain)
        
        #Prediction of CATE on test data
        cate_trans_cov.append(tn_cov.predict_delta(x_test))
        
        ###\tau_cor bias
        tn_bias = CorNet(n_cov, n_hidden, d_hidden, n_out, d_out)
        #Train both networks on confounded data (Step 1)
        tn_bias._train(x_c_bias, y_c_bias, t_c_bias, x_val, y_val, t_val, lambd_h_c, batch_size, lr_train, n_epochs_feature)
                 
        #Train head regulaized on uncofounded data (Step 2)
        tn_bias._retrain_delta(x_uc, y_uc, t_uc, lambd_delta, lr_ret, n_epochs_retrain)
        
        #Prediction of CATE on test data
        cate_trans_bias.append(tn_bias.predict_delta(x_test))
    
    cate_trans_bias_l.append(np.array([((cate_test - cate_trans_bias[i].numpy())**2).mean() for i in range(n_run)]))
    cate_trans_cov_l.append(np.array([((cate_test - cate_trans_cov[i].numpy())**2).mean() for i in range(n_run)]))
    cate_trans_l.append(np.array([((cate_test - cate_trans[i].numpy())**2).mean() for i in range(n_run)]))        
 



#%%

matplotlib.rcParams.update({'font.size': 12})


# Data for plotting
t = np.sqrt(n_conf_sizes/m)
cate_t = np.sqrt(np.array(cate_trans_l))
cate_u = np.sqrt(np.array(cate_naive_l))

cate_t_cov = np.sqrt(np.array(cate_trans_cov_l))
cate_u_cov = np.sqrt(np.array(cate_naive_cov_l))

cate_t_bias = np.sqrt(np.array(cate_trans_bias_l))
cate_u_bias = np.sqrt(np.array(cate_naive_bias_l))




# =============================================================================
# #Save data
# np.save('sim_cond_1_trans', cate_t)
# np.save('sim_cond_1_unc', cate_u)
# np.save('sim_cond_1_trans_cov', cate_t_cov)
# np.save('sim_cond_1_u_cov', cate_u_cov)
# np.save('sim_cond_1_trans_bias', cate_t_bias)
# np.save('sim_cond_1_unc_bias', cate_u_bias)
# =============================================================================


# =============================================================================
# #Load data
# cate_t = np.load('sim_cond_1_trans.npy')
# cate_u = np.load('sim_cond_1_unc.npy')
# cate_t_cov = np.load('sim_cond_1_trans_cov.npy')
# cate_u_cov = np.load('sim_cond_1_u_cov.npy')
# cate_t_bias = np.load('sim_cond_1_trans_bias.npy')
# cate_u_bias = np.load('sim_cond_1_unc_bias.npy')
# 
# =============================================================================


m=50
t1 = np.concatenate((np.linspace(0.4472136, 1.5, 10), np.linspace(1.5, 2.44948974, 6)[1:]))
n_conf_sizes = np.round(t1**2*m)
t = np.sqrt(n_conf_sizes/m)

ms = 3
lw=1
elw=0.5
cs=2

fig, ax = plt.subplots(3, 1)
fig.set_figwidth(5)
fig.set_figheight(5)
ax[0].errorbar(t, cate_t.mean(0), cate_t.std(0), mfc='black', color = 'black', ecolor = 'black', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)#plot(t, cate_t, 'b')
ax[0].errorbar(t, cate_u.mean(0), cate_u.std(0), mfc='r', color = 'r', ecolor = 'r', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)#plot(t, cate_n, 'r')
ax[1].errorbar(t, cate_t_cov.mean(0), cate_t_cov.std(0), mfc='black', color = 'black', ecolor = 'black', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)#.plot(t, cate_t_cov, 'b')
ax[1].errorbar(t, cate_u_cov.mean(0), cate_u_cov.std(0), mfc='r', color = 'r', ecolor = 'r', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)#plot(t, cate_n_cov, 'r')
ax[2].errorbar(t, cate_t_bias.mean(0), cate_t_bias.std(0), mfc='black', color = 'black', ecolor = 'black', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)#.plot(t, cate_t_bias, 'b')
ax[2].errorbar(t, cate_u_bias.mean(0), cate_u_bias.std(0), mfc='r', color = 'r', ecolor = 'r', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)#.plot(t, cate_n_bias, 'r')
ax[0].set_xticklabels(np.repeat("", n_run))
ax[1].set_xticklabels(np.repeat("", n_run))

ax[0].vlines(t[3]+0.004, 0.5, 4,colors='black', linestyles='dashed', zorder=3)
ax[1].vlines(t[1]+0.004, 0.5, 4,colors='black', linestyles='dashed', zorder=3)
ax[2].vlines(t[4]+0.004, 0.5, 4,colors='black', linestyles='dashed', zorder=3)

#ax.grid()
ax[0].set_title(r'$\mathit{d}_\infty=1, \mathcal{C}_B$ small', fontsize=8)
ax[1].set_title(r'$\mathit{d}_\infty>1, \mathcal{C}_B$ small', fontsize=8)
ax[2].set_title(r'$\mathit{d}_\infty=1, \mathcal{C}_B$ large', fontsize=8)

ax[0].set(ylabel=r'$\sqrt{\epsilon_{PEHE}}$')
ax[1].set(ylabel=r'$\sqrt{\epsilon_{PEHE}}$')
ax[2].set(ylabel=r'$\sqrt{\epsilon_{PEHE}}$', xlabel=r'$\sqrt{\frac{n^{Conf}}{n^{Unc}}}$')

fig.tight_layout()
for a in ax.flat:
    a.set(ylim = (0.5, 2.25))
    a.grid(linewidth=1, ls='--')

line1 = Line2D([0,1],[0,1],linestyle='-', color='black')
line3 = Line2D([0,1],[0,1],linestyle='-', color='r')

fig.legend([line1, line3],[r'$\tau_{CorNet}$',r'$\tau_{Unc}$'], bbox_to_anchor=(0.5, -0.075),loc = 'lower center', ncol=3)

fig.savefig("sim_results_condition_unc.pdf", dpi=300, bbox_inches='tight')
plt.show()




