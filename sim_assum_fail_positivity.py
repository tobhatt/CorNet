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
import matplotlib
from matplotlib.lines import Line2D

#-------------------------------------------
# Study F2: Overlap/Positivity
#-------------------------------------------
'''
p_x^Conf and p_x^Unc do not satisfy positivity
'''

# Define \phi and h^u, h^c
n_cov = 10
n_hidden_cov = 20
sparse_ratio = 0.6666
alpha = 10
beta = 2
n_run=10

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
x_test = torch.Tensor(np.random.multivariate_normal(np.zeros(n_cov), np.identity(n_cov), size= [n_test]))
    
print(((net.predict_delta(x_test) - net.predict_naive(x_test))**2).mean())
bias = ((net.predict_delta(x_test) - net.predict_naive(x_test))**2).mean()

net.delta_1.weight = torch.nn.Parameter(beta * net.delta_1.weight/torch.sqrt(bias))
net.delta_1.bias = torch.nn.Parameter(beta * net.delta_1.bias/torch.sqrt(bias))

net.delta_0.weight = torch.nn.Parameter(beta * net.delta_0.weight/torch.sqrt(bias))
net.delta_0.bias = torch.nn.Parameter(beta * net.delta_0.bias/torch.sqrt(bias))

print(((net.predict_delta(x_test) - net.predict_naive(x_test))**2).mean())

cate_trans_ol_large_l, cate_unc_ol_large_l, cate_conf_ol_large_l = [], [], []
cate_trans_ol_medium_l, cate_unc_ol_medium_l, cate_conf_ol_medium_l = [], [], []
cate_trans_ol_small_l, cate_unc_ol_small_l, cate_conf_ol_small_l = [], [], []

cate_trans_normal_large_l, cate_unc_normal_large_l, cate_conf_normal_large_l = [], [], []
cate_trans_normal_medium_l, cate_unc_normal_medium_l, cate_conf_normal_medium_l = [], [], []
cate_trans_normal_small_l, cate_unc_normal_small_l, cate_conf_normal_small_l = [], [], []

for i in range(10):
    # Simulate data
    n_conf = 500
    n_unc = 50
    
    m=50
    
    sim_dict = sim_data_assum_positivity(net, n_conf, n_unc, n_cov)     

    #Confounded data
    x_conf = sim_dict['x_conf']
    t_conf = sim_dict['t_conf']
    y_conf = sim_dict['y_conf']

    #Unconfounded data
    x_unc_ol_large = sim_dict['x_unc_ol_large']
    t_unc_ol_large = sim_dict['t_unc_ol_large']
    y_unc_ol_large = sim_dict['y_unc_ol_large']

    #Unconfounded data
    x_unc_ol_medium = sim_dict['x_unc_ol_medium']
    t_unc_ol_medium = sim_dict['t_unc_ol_medium']
    y_unc_ol_medium = sim_dict['y_unc_ol_medium']

    #Unconfounded data
    x_unc_ol_small = sim_dict['x_unc_ol_small']
    t_unc_ol_small = sim_dict['t_unc_ol_small']
    y_unc_ol_small = sim_dict['y_unc_ol_small']

    #Unconfounded data
    x_unc_normal_large = sim_dict['x_unc_normal_large']
    t_unc_normal_large = sim_dict['t_unc_normal_large']
    y_unc_normal_large = sim_dict['y_unc_normal_large']

    #Unconfounded data
    x_unc_normal_medium = sim_dict['x_unc_normal_medium']
    t_unc_normal_medium = sim_dict['t_unc_normal_medium']
    y_unc_normal_medium = sim_dict['y_unc_normal_medium']

    #Unconfounded data
    x_unc_normal_small = sim_dict['x_unc_normal_small']
    t_unc_normal_small = sim_dict['t_unc_normal_small']
    y_unc_normal_small = sim_dict['y_unc_normal_small']

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
    
    cate_trans_ol_large, cate_unc_ol_large, cate_conf_ol_large = [], [], []
    cate_trans_normal_large, cate_unc_normal_large, cate_conf_normal_large = [], [], []
    cate_trans_ol_medium, cate_unc_ol_medium, cate_conf_ol_medium = [], [], []
    cate_trans_normal_medium, cate_unc_normal_medium, cate_conf_normal_medium = [], [], []
    cate_trans_ol_small, cate_unc_ol_small, cate_conf_ol_small = [], [], []
    cate_trans_normal_small, cate_unc_normal_small, cate_conf_normal_small = [], [], []
    
    x_uc_ol_large, t_uc_ol_large, y_uc_ol_large = x_unc_ol_large[:m,:], t_unc_ol_large[:m,:], y_unc_ol_large[:m,:]
    x_uc_ol_medium, t_uc_ol_medium, y_uc_ol_medium = x_unc_ol_medium[:m,:], t_unc_ol_medium[:m,:], y_unc_ol_medium[:m,:]
    x_uc_ol_small, t_uc_ol_small, y_uc_ol_small = x_unc_ol_small[:m,:], t_unc_ol_small[:m,:], y_unc_ol_small[:m,:]
    
    x_uc_normal_large, t_uc_normal_large, y_uc_normal_large = x_unc_normal_large[:m,:], t_unc_normal_large[:m,:], y_unc_normal_large[:m,:]
    x_uc_normal_medium, t_uc_normal_medium, y_uc_normal_medium = x_unc_normal_medium[:m,:], t_unc_normal_medium[:m,:], y_unc_normal_medium[:m,:]
    x_uc_normal_small, t_uc_normal_small, y_uc_normal_small = x_unc_normal_small[:m,:], t_unc_normal_small[:m,:], y_unc_normal_small[:m,:]


    #On unc data only        
    #Train both networks on confounded data (Step 1)
    tn_ol_large = CorNet(n_cov, n_hidden, d_hidden, n_out, d_out)
    tn_ol_large._train(x_uc_ol_large, y_uc_ol_large, t_uc_ol_large, x_val, y_val, t_val, lambd_h_c, batch_size, lr_train, n_epochs_feature)

    #Train both networks on confounded data (Step 1)
    tn_ol_medium = CorNet(n_cov, n_hidden, d_hidden, n_out, d_out)
    tn_ol_medium._train(x_uc_ol_medium, y_uc_ol_medium, t_uc_ol_medium, x_val, y_val, t_val, lambd_h_c, batch_size, lr_train, n_epochs_feature)

    #Train both networks on confounded data (Step 1)
    tn_ol_small = CorNet(n_cov, n_hidden, d_hidden, n_out, d_out)
    tn_ol_small._train(x_uc_ol_small, y_uc_ol_small, t_uc_ol_small, x_val, y_val, t_val, lambd_h_c, batch_size, lr_train, n_epochs_feature)
    
    #On unc data only        
    #Train both networks on confounded data (Step 1)
    tn_normal_large = CorNet(n_cov, n_hidden, d_hidden, n_out, d_out)
    tn_normal_large._train(x_uc_normal_large, y_uc_normal_large, t_uc_normal_large, x_val, y_val, t_val, lambd_h_c, batch_size, lr_train, n_epochs_feature)

    #Train both networks on confounded data (Step 1)
    tn_normal_medium = CorNet(n_cov, n_hidden, d_hidden, n_out, d_out)
    tn_normal_medium._train(x_uc_normal_medium, y_uc_normal_medium, t_uc_normal_medium, x_val, y_val, t_val, lambd_h_c, batch_size, lr_train, n_epochs_feature)

    #Train both networks on confounded data (Step 1)
    tn_normal_small = CorNet(n_cov, n_hidden, d_hidden, n_out, d_out)
    tn_normal_small._train(x_uc_normal_small, y_uc_normal_small, t_uc_normal_small, x_val, y_val, t_val, lambd_h_c, batch_size, lr_train, n_epochs_feature)
    
    #Prediction of CATE on test data
    for i in range(n_run):
        cate_unc_ol_large.append(tn_ol_large.predict_naive(x_test))
        cate_unc_ol_medium.append(tn_ol_medium.predict_naive(x_test))
        cate_unc_ol_small.append(tn_ol_small.predict_naive(x_test))
    
        cate_unc_normal_large.append(tn_normal_large.predict_naive(x_test))
        cate_unc_normal_medium.append(tn_normal_medium.predict_naive(x_test))
        cate_unc_normal_small.append(tn_normal_small.predict_naive(x_test))    
    
    
    n_conf_sizes = np.linspace(25, n_conf, n_run)
    for i in range(n_run):
        n = int(n_conf_sizes[i])
        print('###################     Training CorNet on ', n,' samples.     ####################')
        x_c, t_c, y_c = x_conf[:n,:], t_conf[:n,:], y_conf[:n,:]
        batch_size = x_c.shape[0]
 
    
        #WITHOUT regularizations
        
        #Train both networks on confounded data (Step 1)
        tn = CorNet(n_cov, n_hidden, d_hidden, n_out, d_out)
        tn._train(x_c, y_c, t_c, x_val, y_val, t_val, lambd_h_c, batch_size, lr_train, n_epochs_feature)
        
        tn_ol_large = deepcopy(tn)
        tn_ol_medium = deepcopy(tn)
        tn_ol_small = deepcopy(tn)

        tn_normal_large = deepcopy(tn)
        tn_normal_medium = deepcopy(tn)
        tn_normal_small = deepcopy(tn)
        
    
        #Train head regulaized on uncofounded data (Step 2)
        tn_ol_large._retrain_delta(x_uc_ol_large, y_uc_ol_large, t_uc_ol_large, lambd_delta, lr_ret, n_epochs_retrain)
        tn_ol_medium._retrain_delta(x_uc_ol_medium, y_uc_ol_medium, t_uc_ol_medium, lambd_delta, lr_ret, n_epochs_retrain)
        tn_ol_small._retrain_delta(x_uc_ol_small, y_uc_ol_small, t_uc_ol_small, lambd_delta, lr_ret, n_epochs_retrain)    

        #Train head regulaized on uncofounded data (Step 2)
        tn_normal_large._retrain_delta(x_uc_normal_large, y_uc_normal_large, t_uc_normal_large, lambd_delta, lr_ret, n_epochs_retrain)
        tn_normal_medium._retrain_delta(x_uc_normal_medium, y_uc_normal_medium, t_uc_normal_medium, lambd_delta, lr_ret, n_epochs_retrain)
        tn_normal_small._retrain_delta(x_uc_normal_small, y_uc_normal_small, t_uc_normal_small, lambd_delta, lr_ret, n_epochs_retrain)    
        
    
        #Prediction of CATE on test data
        cate_trans_ol_large.append(tn_ol_large.predict_delta(x_test))
        cate_conf_ol_large.append(tn_ol_large.predict_naive(x_test))
        
        cate_trans_ol_medium.append(tn_ol_medium.predict_delta(x_test))
        cate_conf_ol_medium.append(tn_ol_medium.predict_naive(x_test))
        
        cate_trans_ol_small.append(tn_ol_small.predict_delta(x_test))
        cate_conf_ol_small.append(tn_ol_small.predict_naive(x_test))
        
        cate_trans_normal_large.append(tn_normal_large.predict_delta(x_test))
        cate_conf_normal_large.append(tn_normal_large.predict_naive(x_test))
        
        cate_trans_normal_medium.append(tn_normal_medium.predict_delta(x_test))
        cate_conf_normal_medium.append(tn_normal_medium.predict_naive(x_test))
        
        cate_trans_normal_small.append(tn_normal_small.predict_delta(x_test))
        cate_conf_normal_small.append(tn_normal_small.predict_naive(x_test))
        
        
        
    cate_trans_ol_large_l.append(np.array([((cate_test - cate_trans_ol_large[i].numpy())**2).mean() for i in range(n_run)]))
    cate_conf_ol_large_l.append(np.array([((cate_test - cate_conf_ol_large[i].numpy())**2).mean() for i in range(n_run)]))
    cate_unc_ol_large_l.append(np.array([((cate_test - cate_unc_ol_large[i].numpy())**2).mean() for i in range(n_run)]))
    
    cate_trans_ol_medium_l.append(np.array([((cate_test - cate_trans_ol_medium[i].numpy())**2).mean() for i in range(n_run)]))
    cate_conf_ol_medium_l.append(np.array([((cate_test - cate_conf_ol_medium[i].numpy())**2).mean() for i in range(n_run)]))
    cate_unc_ol_medium_l.append(np.array([((cate_test - cate_unc_ol_medium[i].numpy())**2).mean() for i in range(n_run)]))
    
    cate_trans_ol_small_l.append(np.array([((cate_test - cate_trans_ol_small[i].numpy())**2).mean() for i in range(n_run)]))
    cate_conf_ol_small_l.append(np.array([((cate_test - cate_conf_ol_small[i].numpy())**2).mean() for i in range(n_run)]))
    cate_unc_ol_small_l.append(np.array([((cate_test - cate_unc_ol_small[i].numpy())**2).mean() for i in range(n_run)]))
    
    cate_trans_normal_large_l.append(np.array([((cate_test - cate_trans_normal_large[i].numpy())**2).mean() for i in range(n_run)]))
    cate_conf_normal_large_l.append(np.array([((cate_test - cate_conf_normal_large[i].numpy())**2).mean() for i in range(n_run)]))
    cate_unc_normal_large_l.append(np.array([((cate_test - cate_unc_normal_large[i].numpy())**2).mean() for i in range(n_run)]))
    
    cate_trans_normal_medium_l.append(np.array([((cate_test - cate_trans_normal_medium[i].numpy())**2).mean() for i in range(n_run)]))
    cate_conf_normal_medium_l.append(np.array([((cate_test - cate_conf_normal_medium[i].numpy())**2).mean() for i in range(n_run)]))
    cate_unc_normal_medium_l.append(np.array([((cate_test - cate_unc_normal_medium[i].numpy())**2).mean() for i in range(n_run)]))
    
    cate_trans_normal_small_l.append(np.array([((cate_test - cate_trans_normal_small[i].numpy())**2).mean() for i in range(n_run)]))
    cate_conf_normal_small_l.append(np.array([((cate_test - cate_conf_normal_small[i].numpy())**2).mean() for i in range(n_run)]))
    cate_unc_normal_small_l.append(np.array([((cate_test - cate_unc_normal_small[i].numpy())**2).mean() for i in range(n_run)]))


#%%

cate_trans1 = np.sqrt(cate_trans_ol_large_l)
cate_unc1 = np.sqrt(cate_unc_ol_large_l)
cate_conf1 = np.sqrt(cate_conf_ol_large_l)

cate_trans12 = np.sqrt(cate_trans_normal_large_l)
cate_unc12 = np.sqrt(cate_unc_normal_large_l)
cate_conf12 = np.sqrt(cate_conf_normal_large_l)

cate_trans2 = np.sqrt(cate_trans_ol_medium_l)
cate_unc2 = np.sqrt(cate_unc_ol_medium_l)
cate_conf2 = np.sqrt(cate_conf_ol_medium_l)

cate_trans22 = np.sqrt(cate_trans_normal_medium_l)
cate_unc22 = np.sqrt(cate_unc_normal_medium_l)
cate_conf22 = np.sqrt(cate_conf_normal_medium_l)

cate_trans3 = np.sqrt(cate_trans_ol_small_l)
cate_unc3 = np.sqrt(cate_unc_ol_small_l)
cate_conf3 = np.sqrt(cate_conf_ol_small_l)

cate_trans32 = np.sqrt(cate_trans_normal_small_l)
cate_unc32 = np.sqrt(cate_unc_normal_small_l)
cate_conf32 = np.sqrt(cate_conf_normal_small_l)


matplotlib.rcParams.update({'font.size': 12})


# =============================================================================
# #Save data
# np.save('sim_d_infty_trans1', cate_trans1)
# np.save('sim_d_infty_unc1', cate_unc1)
# np.save('sim_d_infty_conf1', cate_conf1)
# 
# np.save('sim_d_infty_trans2', cate_trans2)
# np.save('sim_d_infty_unc2', cate_unc2)
# np.save('sim_d_infty_conf2', cate_conf2)
# 
# np.save('sim_d_infty_trans3', cate_trans3)
# np.save('sim_d_infty_unc3', cate_unc3)
# np.save('sim_d_infty_conf3', cate_conf3)
# 
# np.save('sim_d_infty_trans12', cate_trans12)
# np.save('sim_d_infty_unc12', cate_unc12)
# np.save('sim_d_infty_conf12', cate_conf12)
# 
# np.save('sim_d_infty_trans22', cate_trans22)
# np.save('sim_d_infty_unc22', cate_unc22)
# np.save('sim_d_infty_conf22', cate_conf22)
# 
# np.save('sim_d_infty_trans32', cate_trans32)
# np.save('sim_d_infty_unc32', cate_unc32)
# np.save('sim_d_infty_conf32', cate_conf32)
# =============================================================================

# Define \phi and h^u, h^c
n_cov = 10
n_hidden_cov = 20
sparse_ratio = 0.6666
alpha = 10
beta = 2
n_run=10

#Net with weights
net = CorNet(n_cov, 3, n_hidden_cov, 0, n_hidden_cov)



#Load data
cate_trans1 = np.load('sim_d_infty_trans1.npy')
cate_unc1 = np.load('sim_d_infty_unc1.npy')
cate_conf1 = np.load('sim_d_infty_conf1.npy')
cate_trans2 = np.load('sim_d_infty_trans2.npy')
cate_unc2 = np.load('sim_d_infty_unc2.npy')
cate_conf2 = np.load('sim_d_infty_conf2.npy')
cate_trans3 = np.load('sim_d_infty_trans3.npy')
cate_unc3 = np.load('sim_d_infty_unc3.npy')
cate_conf3 = np.load('sim_d_infty_conf3.npy')

cate_trans12 = np.load('sim_d_infty_trans12.npy')
cate_unc12 = np.load('sim_d_infty_unc12.npy')
cate_conf12 = np.load('sim_d_infty_conf12.npy')
cate_trans22 = np.load('sim_d_infty_trans22.npy')
cate_unc22 = np.load('sim_d_infty_unc22.npy')
cate_conf22 = np.load('sim_d_infty_conf22.npy')
cate_trans32 = np.load('sim_d_infty_trans32.npy')
cate_unc32 = np.load('sim_d_infty_unc32.npy')
cate_conf32 = np.load('sim_d_infty_conf32.npy')


# Data for plotting
n_conf = 500
n_conf_sizes = np.linspace(25, n_conf, n_run)
t = n_conf_sizes
cate_t1 = np.array(cate_trans1).mean(0)
cate_u1 = np.array(cate_unc1).mean(0)
cate_c1 = np.array(cate_conf1).mean(0)

cate_t12 = np.array(cate_trans12).mean(0)
cate_u12 = np.array(cate_unc12).mean(0)
cate_c12 = np.array(cate_conf12).mean(0)

cate_t2 = np.array(cate_trans2).mean(0)
cate_u2 = np.array(cate_unc2).mean(0)
cate_c2 = np.array(cate_conf2).mean(0)

cate_t22 = np.array(cate_trans22).mean(0)
cate_u22 = np.array(cate_unc22).mean(0)
cate_c22 = np.array(cate_conf22).mean(0)

cate_t3 = np.array(cate_trans3).mean(0)
cate_u3 = np.array(cate_unc3).mean(0)
cate_c3 = np.array(cate_conf3).mean(0)

cate_t32 = np.array(cate_trans32).mean(0)
cate_u32 = np.array(cate_unc32).mean(0)
cate_c32 = np.array(cate_conf32).mean(0)



ms = 3
lw=1
elw=0.5
cs=2


#data for histograms
sim_dict = sim_data_assum_positivity(net, 10000, 10000, n_cov)  

x_test = sim_dict['x_test'][:,0].numpy()

x1 = sim_dict['x_unc_ol_large'][:,0].numpy()
x2 = sim_dict['x_unc_ol_medium'][:,0].numpy()
x3 = sim_dict['x_unc_ol_small'][:,0].numpy()

x12 = sim_dict['x_unc_normal_large'][:,0].numpy()
x22 = sim_dict['x_unc_normal_medium'][:,0].numpy()
x32 = sim_dict['x_unc_normal_small'][:,0].numpy()

matplotlib.rcParams.update({'font.size': 12})


# Set up the axes with gridspec
fig = plt.figure(figsize=(9, 6))
#fig, ax = plt.subplots(2,3)
grid = plt.GridSpec(9, 17, hspace=0.2, wspace=0.2)

main_ax1 = fig.add_subplot(grid[0:3, 0:5])
x_hist1 = fig.add_subplot(grid[3, 0:5], yticklabels=[])

main_ax2 = fig.add_subplot(grid[0:3, 6:11])
x_hist2 = fig.add_subplot(grid[3, 6:11], yticklabels=[])

main_ax3 = fig.add_subplot(grid[0:3, 12:17])
x_hist3 = fig.add_subplot(grid[3, 12:17], yticklabels=[])

main_ax12 = fig.add_subplot(grid[5:8, 0:5])
x_hist12 = fig.add_subplot(grid[8, 0:5], yticklabels=[])

main_ax22 = fig.add_subplot(grid[5:8, 6:11])
x_hist22 = fig.add_subplot(grid[8, 6:11], yticklabels=[])

main_ax23 = fig.add_subplot(grid[5:8, 12:17])
x_hist23 = fig.add_subplot(grid[8, 12:17], yticklabels=[])


main_ax1.errorbar(t, cate_t1, np.array(cate_trans1).std(0), mfc='black', color = 'black', ecolor = 'black', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)
main_ax1.errorbar(t, cate_c1, np.array(cate_conf1).std(0), mfc='b', color = 'b', ecolor = 'b', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)
main_ax1.errorbar(t, cate_u1, np.array(cate_unc1).std(0), mfc='r', color = 'r', ecolor = 'r', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)
main_ax1.xaxis.set_ticks(np.array([25, 200, 400]))
main_ax1.set(xlabel='$n^{conf}$', ylabel=r'$\sqrt{\epsilon_{PEHE}}$',
       title=r'$X^{Unc}\sim \mathcal{U}[-3, 3]$', ylim=(0.5, 3.5))
main_ax1.grid(linewidth=1, ls='--')


main_ax2.errorbar(t, cate_t2, np.array(cate_trans2).std(0), mfc='black', color = 'black', ecolor = 'black', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)
main_ax2.errorbar(t, cate_c2, np.array(cate_conf2).std(0), mfc='b', color = 'b', ecolor = 'b', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)
main_ax2.errorbar(t, cate_u2, np.array(cate_unc2).std(0), mfc='r', color = 'r', ecolor = 'r', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)
main_ax2.xaxis.set_ticks(np.array([25, 200, 400]))
main_ax2.set(xlabel='$n^{conf}$',
       title=r'$X^{Unc}\sim \mathcal{U}[-1, 1]$', ylim=(0.5, 3.5))
main_ax2.grid(linewidth=1, ls='--')


main_ax3.errorbar(t, cate_t3, np.array(cate_trans3).std(0),  mfc='black', color = 'black', ecolor = 'black', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)
main_ax3.errorbar(t, cate_c3, np.array(cate_conf3).std(0), mfc='b', color = 'b', ecolor = 'b', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)
main_ax3.errorbar(t, cate_u3, np.array(cate_unc3).std(0), mfc='r', color = 'r', ecolor = 'r', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)
main_ax3.xaxis.set_ticks(np.array([25, 200, 400]))
main_ax3.set(xlabel='$n^{conf}$',
       title=r'$X^{Unc}\sim \mathcal{U}[-1/2, 1/2]$', ylim=(0.5, 3.5))
main_ax3.grid(linewidth=1, ls='--')

main_ax12.errorbar(t, cate_t12, np.array(cate_trans12).std(0), mfc='black', color = 'black', ecolor = 'black', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)
main_ax12.errorbar(t, cate_c12, np.array(cate_conf12).std(0), mfc='b', color = 'b', ecolor = 'b', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)
main_ax12.errorbar(t, cate_u12, np.array(cate_unc12).std(0), mfc='r', color = 'r', ecolor = 'r', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)
main_ax12.xaxis.set_ticks(np.array([25, 200, 400]))
main_ax12.set(xlabel='$n^{conf}$', ylabel=r'$\sqrt{\epsilon_{PEHE}}$',
      title=r'$X^{Unc}\sim \mathcal{N}(0, 1^2)$', ylim=(0.5, 3.5))
main_ax12.grid(linewidth=1, ls='--')

main_ax22.errorbar(t, cate_t22, np.array(cate_trans22).std(0),  mfc='black', color = 'black', ecolor = 'black', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)
main_ax22.errorbar(t, cate_c22, np.array(cate_conf22).std(0), mfc='b', color = 'b', ecolor = 'b', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)
main_ax22.errorbar(t, cate_u22, np.array(cate_unc22).std(0), mfc='r', color = 'r', ecolor = 'r', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)
main_ax22.xaxis.set_ticks(np.array([25, 200, 400]))
main_ax22.set(xlabel='$n^{conf}$',
       title=r'$X^{Unc}\sim \mathcal{N}(0, 1/3^2)$', ylim=(0.5, 3.5))
main_ax22.grid(linewidth=1, ls='--')

main_ax23.errorbar(t, cate_t32, np.array(cate_trans32).std(0),  mfc='black', color = 'black', ecolor = 'black', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)
main_ax23.errorbar(t, cate_c32, np.array(cate_conf32).std(0), mfc='b', color = 'b', ecolor = 'b', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)
main_ax23.errorbar(t, cate_u32, np.array(cate_unc32).std(0), mfc='r', color = 'r', ecolor = 'r', marker='o', markersize=ms, linewidth=lw, elinewidth=elw, capsize=cs)
main_ax23.xaxis.set_ticks(np.array([25, 200, 400]))
main_ax23.set(xlabel='$n^{conf}$',
       title=r'$X^{Unc}\sim \mathcal{N}(0, 1/6^2)$', ylim=(0.5, 3.5))
main_ax23.grid(linewidth=1, ls='--')


# histogram on the attached axes
x_hist1.hist(x_test, histtype='stepfilled', range=(-3.5, 3.5), density=True, bins=20, color='b')
x_hist1.hist(x1, histtype='step', range=(-3, 3), density=True, bins=1, color='r')
x_hist1.xaxis.set_ticks(np.array([-3,3]))
#x_hist1.set_ylim([0, 1])


x_hist2.hist(x_test, histtype='stepfilled', range=(-3.5, 3.5), density=True, bins=20, color='b')
x_hist2.hist(x2, histtype='step', range=(-1, 1), density=True, bins=1, color='r')
x_hist2.xaxis.set_ticks(np.array([-1,1]))
#x_hist2.set_ylim([0, 1])

x_hist3.hist(x_test, histtype='stepfilled', range=(-3.5, 3.5), density=True, bins=20, color='b')
x_hist3.hist(x3, histtype='step', range=(-.5, .5), density=True, bins=1, color='r')
x_hist3.xaxis.set_ticks(np.array([-0.5,0.5]))
x_hist3.xaxis.set_ticklabels(np.array([r'$-\frac{1}{2}$',r'$\frac{1}{2}$']))
#x_hist3.set_ylim([0, 1])

x_hist12.hist(x_test, histtype='stepfilled', range=(-3.5, 3.5), density=True, bins=20, color='b')
x_hist12.hist(x12, histtype='step', range=(-3.5, 3.5), density=True, bins=20, color='r')
x_hist12.xaxis.set_ticks(np.array([-3,3]))

x_hist22.hist(x_test, histtype='stepfilled', range=(-3.5, 3.5), density=True, bins=20, color='b')
x_hist22.hist(x22, histtype='step', range=(-3.5, 3.5), density=True, bins=20, color='r')
x_hist22.xaxis.set_ticks(np.array([-1,1]))

x_hist23.hist(x_test, histtype='stepfilled', range=(-3.5, 3.5), density=True, bins=20, color='b')
x_hist23.hist(x32, histtype='step', range=(-3.5, 3.5), density=True, bins=20, color='r')
x_hist23.xaxis.set_ticks(np.array([-0.5,0.5]))
x_hist23.xaxis.set_ticklabels(np.array([r'$-\frac{1}{2}$',r'$\frac{1}{2}$']))

line1 = Line2D([0,1],[0,1],linestyle='-', color='black')
line2 = Line2D([0,1],[0,1],linestyle='-', color='b')
line3 = Line2D([0,1],[0,1],linestyle='-', color='r')

fig.legend([line1, line2, line3], [r'$\tau_{CorNet}$', r'$\tau_{Conf}$', r'$\tau_{Unc}$'], bbox_to_anchor=(0.5, -0.00), loc = 'lower center', ncol=3, numpoints=1)

#fig.savefig("sim_fail_overlap.pdf", dpi=300, bbox_inches='tight')
plt.show()


bla

