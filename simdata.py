import numpy as np
from scipy.special import expit
from models import CorNet
import torch
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
import scipy
import random

def sim_data_n_conf(net, n_conf, n_unc, n_cov, cov_shift = 1, noise_scale = 0.1):

    cov_matrix = np.identity(n_cov)
    #Unconfounded data (small) WITHOUT covariate shift
    x_unc = torch.Tensor(np.random.multivariate_normal(np.zeros(n_cov), cov_matrix, size= [n_unc]))
    t_unc = torch.Tensor(np.random.binomial(1, 0.5, size=[n_unc, 1]))
    with torch.no_grad():
        rep, y1, y0 = net(x_unc)    
        y1 = net.linear_out_1(rep) + net.delta_1(rep)
        y0 = net.linear_out_0(rep) + net.delta_0(rep)
    y_unc = t_unc * y1 + (1-t_unc) * y0 + 0.5*torch.normal(0, 1, size=(rep.size(0), 1))

    
    #Confounded data (large)
    t_conf = torch.Tensor(np.random.binomial(1, 0.5, size=[n_conf, 1]))
    x_conf = torch.Tensor(np.random.multivariate_normal(np.zeros(n_cov), cov_matrix, size= [n_conf]))
    with torch.no_grad():
        rep, y1, y0 = net(torch.Tensor(x_conf))    
    y_conf = t_conf * y1 + (1-t_conf) * y0 + 0.5*torch.normal(0, 1, size=(rep.size(0), 1))

    #Test data
    n_test = 10000
    #Confounded data (large)
    t_test = torch.Tensor(np.random.binomial(1, 0.5, size=[n_test, 1]))
    x_test = torch.Tensor(np.random.multivariate_normal(np.zeros(n_cov), cov_matrix, size= [n_test]))
    
    with torch.no_grad():
        rep, y1, y0 = net(x_test)    
        y1 = net.linear_out_1(rep) + net.delta_1(rep)
        y0 = net.linear_out_0(rep) + net.delta_0(rep)
    
    cate_test = y1 - y0
    
    return {'cate_test': cate_test,'x_unc': x_unc, 't_unc': t_unc, 'y_unc': y_unc, 'x_conf': x_conf, 't_conf': t_conf, 'y_conf': y_conf, 'x_test': x_test}




def sim_data_d(net, n_conf, n_unc, n_cov, cov_shift = 0, noise_scale = 0.1):

    #Unconfounded data (small) NO covariate shift
    x_unc = torch.Tensor(np.random.multivariate_normal(np.zeros(n_cov), np.identity(n_cov), size= [n_unc]))
    t_unc = torch.Tensor(np.random.binomial(1, 0.5, size=[n_unc, 1]))
    with torch.no_grad():
        rep, y1, y0 = net(x_unc)    
        y1 = net.linear_out_1(rep) + net.delta_1(rep)
        y0 = net.linear_out_0(rep) + net.delta_0(rep)
    y_unc = t_unc * y1 + (1-t_unc) * y0 + 0.5*torch.normal(0, 1, size=(rep.size(0), 1))
    
    #Confounded data (large) with covariate shift
    x_c_cov = torch.Tensor(np.random.multivariate_normal(np.zeros(n_cov)+cov_shift, 2*np.identity(n_cov), size= [n_conf]))
    t_c_cov = torch.Tensor(np.random.binomial(1, 0.5, size=[n_conf, 1]))
    with torch.no_grad():
        rep, y1, y0 = net(x_c_cov)    
    y_c_cov = t_c_cov * y1 + (1-t_c_cov) * y0 + 0.5*torch.normal(0, 1, size=(rep.size(0), 1))
    
    #Confounded data (large)
    t_conf = torch.Tensor(np.random.binomial(1, 0.5, size=[n_conf, 1]))
    x_conf = torch.Tensor(np.random.multivariate_normal(np.zeros(n_cov), np.identity(n_cov), size= [n_conf]))
    with torch.no_grad():
        rep, y1, y0 = net(torch.Tensor(x_conf))    
    y_conf = t_conf * y1 + (1-t_conf) * y0 + 0.5*torch.normal(0, 1, size=(rep.size(0), 1))

    #Test data no covariate shift
    n_test = 10000
    #Confounded data (large)
    t_test = torch.Tensor(np.random.binomial(1, 0.5, size=[n_test, 1]))
    x_test = torch.Tensor(np.random.multivariate_normal(np.zeros(n_cov), np.identity(n_cov), size= [n_test]))
    
    with torch.no_grad():
        rep, y1, y0 = net(x_test)    
        y1 = net.linear_out_1(rep) + net.delta_1(rep)
        y0 = net.linear_out_0(rep) + net.delta_0(rep)
    
    cate_test = y1 - y0
    
    #Test data WITH covariate shift
    n_test = 10000
    #Confounded data (large)
    t_test_cov = torch.Tensor(np.random.binomial(1, 0.5, size=[n_test, 1]))
    x_test_cov = torch.Tensor(np.random.multivariate_normal(np.zeros(n_cov)+cov_shift, 2*np.identity(n_cov), size= [n_test]))
    
    with torch.no_grad():
        rep, y1, y0 = net(x_test_cov)    
        y1 = net.linear_out_1(rep) + net.delta_1(rep)
        y0 = net.linear_out_0(rep) + net.delta_0(rep)
    
    cate_test_cov = y1 - y0
    
    return {'cate_test': cate_test,'cate_test_cov': cate_test_cov,'x_unc': x_unc, 't_unc': t_unc, 'y_unc': y_unc,'x_c_cov': x_c_cov, 't_c_cov': t_c_cov, 'y_c_cov': y_c_cov, 'x_conf': x_conf, 't_conf': t_conf, 'y_conf': y_conf, 'x_test': x_test, 'x_test_cov': x_test_cov}


def sim_data_d3(net, n_conf, n_unc, n_cov, cov_shift = 0, noise_scale = 0.1):

    #Unconfounded data (small) NO covariate shift
    x_unc = torch.Tensor(np.random.multivariate_normal(np.zeros(n_cov), np.identity(n_cov), size= [n_unc]))
    t_unc = torch.Tensor(np.random.binomial(1, 0.5, size=[n_unc, 1]))
    with torch.no_grad():
        rep, y1, y0 = net(x_unc)    
        y1 = net.linear_out_1(rep) + net.delta_1(rep)
        y0 = net.linear_out_0(rep) + net.delta_0(rep)
    y_unc = t_unc * y1 + (1-t_unc) * y0 + 0.5*torch.normal(0, 1, size=(rep.size(0), 1))
    
    #Unconfounded data (small) WITH covariate shift
    x_unc_cov = torch.Tensor(np.random.multivariate_normal(np.zeros(n_cov), 0.1*np.identity(n_cov), size= [n_unc]))
    t_unc_cov = torch.Tensor(np.random.binomial(1, 0.5, size=[n_unc, 1]))
    with torch.no_grad():
        rep, y1, y0 = net(x_unc_cov)    
        y1 = net.linear_out_1(rep) + net.delta_1(rep)
        y0 = net.linear_out_0(rep) + net.delta_0(rep)
    y_unc_cov = t_unc_cov * y1 + (1-t_unc_cov) * y0 + 0.5*torch.normal(0, 1, size=(rep.size(0), 1))
    
    #Confounded data (large)
    t_conf = torch.Tensor(np.random.binomial(1, 0.5, size=[n_conf, 1]))
    x_conf = torch.Tensor(np.random.multivariate_normal(np.zeros(n_cov), np.identity(n_cov), size= [n_conf]))
    with torch.no_grad():
        rep, y1, y0 = net(torch.Tensor(x_conf))    
    y_conf = t_conf * y1 + (1-t_conf) * y0 + 0.5*torch.normal(0, 1, size=(rep.size(0), 1))

    #Test data no covariate shift
    n_test = 10000
    #Confounded data (large)
    t_test = torch.Tensor(np.random.binomial(1, 0.5, size=[n_test, 1]))
    x_test = torch.Tensor(np.random.multivariate_normal(np.zeros(n_cov), np.identity(n_cov), size= [n_test]))
    
    with torch.no_grad():
        rep, y1, y0 = net(x_test)    
        y1 = net.linear_out_1(rep) + net.delta_1(rep)
        y0 = net.linear_out_0(rep) + net.delta_0(rep)
    
    cate_test = y1 - y0

    return {'cate_test': cate_test,'x_unc': x_unc, 't_unc': t_unc, 'y_unc': y_unc,'x_unc_cov': x_unc_cov, 't_unc_cov': t_unc_cov, 'y_unc_cov': y_unc_cov, 'x_conf': x_conf, 't_conf': t_conf, 'y_conf': y_conf, 'x_test': x_test}


def sim_data_bias(nn, beta_2, sparse_ratio, n_hidden_cov, n_conf, n_unc, n_cov):
    net1 = deepcopy(nn)
    #Unconfounded data (small) beta_1
    x_unc = torch.Tensor(np.random.multivariate_normal(np.zeros(n_cov), np.identity(n_cov), size= [n_unc]))
    t_unc = torch.Tensor(np.random.binomial(1, 0.5, size=[n_unc, 1]))
    with torch.no_grad():
        rep, y1, y0 = net1(x_unc)    
        y1 = net1.linear_out_1(rep) + net1.delta_1(rep)
        y0 = net1.linear_out_0(rep) + net1.delta_0(rep)
    y_unc = t_unc * y1 + (1-t_unc) * y0 + 0.5*torch.normal(0, 1, size=(rep.size(0), 1))
    
    
    #First confounded data (large)
    t_conf = torch.Tensor(np.random.binomial(1, 0.5, size=[n_conf, 1]))
    x_conf = torch.Tensor(np.random.multivariate_normal(np.zeros(n_cov), np.identity(n_cov), size=[n_conf]))
    with torch.no_grad():
        rep, y1, y0 = net1(torch.Tensor(x_conf))    
    y_conf = t_conf * y1 + (1-t_conf) * y0 + 0.5*torch.normal(0, 1, size=(rep.size(0), 1))


    #Test data
    n_test = 10000
    #Confounded data (large)
    x_test = torch.Tensor(np.random.multivariate_normal(np.zeros(n_cov), np.identity(n_cov), size= [n_test]))
    
    #with \delta = 1
    with torch.no_grad():
        rep, y1, y0 = net1(x_test)    
        y1 = net1.linear_out_1(rep) + net1.delta_1(rep)
        y0 = net1.linear_out_0(rep) + net1.delta_0(rep)
    
    cate_test = y1 - y0


    #Second confounded data (large)
    t_conf_bias = torch.Tensor(np.random.binomial(1, 0.5, size=[n_conf, 1]))
    x_conf_bias = torch.Tensor(np.random.multivariate_normal(np.zeros(n_cov), np.identity(n_cov), size=[n_conf]))
    with torch.no_grad():
        rep, y1, y0 = net1(torch.Tensor(x_conf_bias))    
        y1 = net1.linear_out_1(rep) + net1.delta_1(rep)
        y0 = net1.linear_out_0(rep) + net1.delta_0(rep)

    
    net1.delta_1.weight = torch.nn.Parameter(beta_2 * net1.delta_1.weight)
    net1.delta_1.bias = torch.nn.Parameter(beta_2 * net1.delta_1.bias)

    net1.delta_0.weight = torch.nn.Parameter(beta_2 * net1.delta_0.weight)
    net1.delta_0.bias = torch.nn.Parameter(beta_2 * net1.delta_0.bias)


    with torch.no_grad():
        y1 = y1 - net1.delta_1(rep)
        y0 = y0 - net1.delta_0(rep)
        
    y_conf_bias = t_conf_bias * y1 + (1-t_conf_bias) * y0 + 0.5*torch.normal(0, 1, size=(rep.size(0), 1))
   
    
    return {'cate_test': cate_test,'x_unc': x_unc, 't_unc': t_unc, 'y_unc': y_unc,'x_conf_bias': x_conf_bias, 't_conf_bias': t_conf_bias, 'y_conf_bias': y_conf_bias, 'x_conf': x_conf, 't_conf': t_conf, 'y_conf': y_conf, 'x_test': x_test}




def sim_data_condition3(nn, beta_2, sparse_ratio, n_hidden_cov, n_conf, n_unc, n_cov, cov_shift = 0):
    net1 = deepcopy(nn)
    #Confounded data (large)
    t_conf = torch.Tensor(np.random.binomial(1, 0.5, size=[n_conf, 1]))
    x_conf = torch.Tensor(np.random.multivariate_normal(np.zeros(n_cov), np.identity(n_cov), size= [n_conf]))
    with torch.no_grad():
        rep, y1, y0 = net1(torch.Tensor(x_conf))    
    y_conf = t_conf * y1 + (1-t_conf) * y0 + 0.5*torch.normal(0, 1, size=(rep.size(0), 1))

    #Unconfounded data (small) NO covariate shift and bias=1
    x_unc = torch.Tensor(np.random.multivariate_normal(np.zeros(n_cov), np.identity(n_cov), size= [n_unc]))
    t_unc = torch.Tensor(np.random.binomial(1, 0.5, size=[n_unc, 1]))
    with torch.no_grad():
        rep, y1, y0 = net1(x_unc)    
        y1 = net1.linear_out_1(rep) + net1.delta_1(rep)
        y0 = net1.linear_out_0(rep) + net1.delta_0(rep)
    y_unc = t_unc * y1 + (1-t_unc) * y0 + 0.5*torch.normal(0, 1, size=(rep.size(0), 1))
    
    #Confounded data (large) with covariate shift
    x_unc_cov = torch.Tensor(np.random.multivariate_normal(np.zeros(n_cov)+cov_shift, 0.2*np.identity(n_cov), size= [n_conf]))
    t_unc_cov = torch.Tensor(np.random.binomial(1, 0.5, size=[n_conf, 1]))
    with torch.no_grad():
        rep, y1, y0 = net1(x_unc_cov)    
    y_unc_cov = t_unc_cov * y1 + (1-t_unc_cov) * y0 + 0.5*torch.normal(0, 1, size=(rep.size(0), 1))

    #Test data
    n_test = 10000
    #Confounded data (large)
    t_test = torch.Tensor(np.random.binomial(1, 0.5, size=[n_conf, 1]))
    x_test = torch.Tensor(np.random.multivariate_normal(np.zeros(n_cov), np.identity(n_cov), size= [n_test]))
    
    with torch.no_grad():
        rep, y1, y0 = net1(x_test)    
        y1 = net1.linear_out_1(rep) + net1.delta_1(rep)
        y0 = net1.linear_out_0(rep) + net1.delta_0(rep)
    
    cate_test = y1 - y0

    
    #Second confounded data (large) with more bias
    t_conf_bias = t_conf
    x_conf_bias = x_conf
    with torch.no_grad():
        rep, y1, y0 = net1(torch.Tensor(x_conf_bias))    
        y1 = net1.linear_out_1(rep) + net1.delta_1(rep)
        y0 = net1.linear_out_0(rep) + net1.delta_0(rep)
   #Sparse \delta 
    
    net1.delta_1.weight = torch.nn.Parameter(beta_2 * net1.delta_1.weight)
    net1.delta_1.bias = torch.nn.Parameter(beta_2 * net1.delta_1.bias)

    net1.delta_0.weight = torch.nn.Parameter(beta_2 * net1.delta_0.weight)
    net1.delta_0.bias = torch.nn.Parameter(beta_2 * net1.delta_0.bias)

    with torch.no_grad():
        y1 = y1 - net1.delta_1(rep)
        y0 = y0 - net1.delta_0(rep)
        
    y_conf_bias = t_conf_bias * y1 + (1-t_conf_bias) * y0 + 0.5*torch.normal(0, 1, size=(rep.size(0), 1))

    
    return {'cate_test': cate_test, 'x_unc': x_unc, 't_unc': t_unc, 'y_unc': y_unc,'x_unc_cov': x_unc_cov, 't_unc_cov': t_unc_cov, 'y_unc_cov': y_unc_cov, 'x_conf_bias': x_conf_bias, 't_conf_bias': t_conf_bias, 'y_conf_bias': y_conf_bias, 'x_conf': x_conf, 't_conf': t_conf, 'y_conf': y_conf, 'x_test': x_test}



def sim_data_condition_conf(nn, beta, beta_2, n_conf, n_unc, n_cov):
    net1 = deepcopy(nn)
    #Confounded data (large)
    t_conf = torch.Tensor(np.random.binomial(1, 0.5, size=[n_conf, 1]))
    x_conf = torch.Tensor(np.random.multivariate_normal(np.zeros(n_cov), np.identity(n_cov), size= [n_conf]))
    with torch.no_grad():
        rep, y1, y0 = net1(x_conf)    
    y_conf = t_conf * y1 + (1-t_conf) * y0 + 0.5*torch.normal(0, 1, size=(rep.size(0), 1))

    #Unconfounded data (small) NO covariate shift and bias=1
    x_unc = torch.Tensor(np.random.multivariate_normal(np.zeros(n_cov), np.identity(n_cov), size= [n_unc]))
    t_unc = torch.Tensor(np.random.binomial(1, 0.5, size=[n_unc, 1]))
    with torch.no_grad():
        rep, y1_c, y0_c = net1(x_unc)
        y1 = y1_c.detach() + net1.delta_1(rep).detach()    
        y0 = y0_c.detach() + net1.delta_0(rep).detach()    
    y_unc = t_unc * y1 + (1-t_unc) * y0 + 0.5*torch.normal(0, 1, size=(rep.size(0), 1))
    
    
    #Confounded data (large) with covariate shift
    x_unc_cov = torch.Tensor(np.random.multivariate_normal(np.zeros(n_cov), 0.2*np.identity(n_cov), size= [n_unc]))
    t_unc_cov = torch.Tensor(np.random.binomial(1, 0.5, size=[n_unc, 1]))
    with torch.no_grad():
        rep, y1_c, y0_c = net1(x_unc_cov)    
        y1 = y1_c.detach() + net1.delta_1(rep).detach()  
        y0 = y0_c.detach() + net1.delta_0(rep).detach()  
    y_unc_cov = t_unc_cov * y1 + (1-t_unc_cov) * y0 + 0.5*torch.normal(0, 1, size=(rep.size(0), 1))

    #Test data
    n_test = 10000
    #Confounded data (large)
    x_test = torch.Tensor(np.random.multivariate_normal(np.zeros(n_cov), np.identity(n_cov), size= [n_test]))
    
    with torch.no_grad():
        rep, y1_c, y0_c = net1(x_test)    
        y1 = y1_c.detach() + net1.delta_1(rep).detach()  
        y0 = y0_c.detach() + net1.delta_0(rep).detach()  
    
    cate_test = y1 - y0
    
    #Second confounded data (large) with more bias
    t_conf_bias = t_conf
    x_conf_bias = x_conf
    with torch.no_grad():
        rep, y1_c, y0_c = net1(x_conf_bias)    
        y1 = y1_c + net1.delta_1(rep).detach()  
        y0 = y0_c + net1.delta_0(rep).detach()  
    
    with torch.no_grad():
        net1.delta_1.weight = torch.nn.Parameter(beta_2 * net1.delta_1.weight.detach()/np.sqrt(beta))
        net1.delta_1.bias = torch.nn.Parameter(beta_2 * net1.delta_1.bias.detach()/np.sqrt(beta))
    
        net1.delta_0.weight = torch.nn.Parameter(beta_2 * net1.delta_0.weight.detach()/np.sqrt(beta))
        net1.delta_0.bias = torch.nn.Parameter(beta_2 * net1.delta_0.bias.detach()/np.sqrt(beta))

    with torch.no_grad(): 
        y1_u = y1 - net1.delta_1(rep).detach()  
        y0_u = y0 - net1.delta_0(rep).detach()  
        
    y_conf_bias = t_conf_bias * y1_u + (1-t_conf_bias) * y0_u + 0.5*torch.normal(0, 1, size=(rep.size(0), 1))

    return {'cate_test': cate_test, 'x_unc': x_unc, 't_unc': t_unc, 'y_unc': y_unc,'x_unc_cov': x_unc_cov, 't_unc_cov': t_unc_cov, 'y_unc_cov': y_unc_cov, 'x_conf_bias': x_conf_bias, 't_conf_bias': t_conf_bias, 'y_conf_bias': y_conf_bias, 'x_conf': x_conf, 't_conf': t_conf, 'y_conf': y_conf, 'x_test': x_test}



def sim_data_assum_shared_rep(nn, norm_infty_2, norm_infty_3, n_conf, n_unc, n_cov):
    n_hidden_cov = nn.rep.layers[0].weight.shape[0]
    net1 = CorNet(n_cov, len(nn.rep.layers), n_hidden_cov, 0, n_hidden_cov)
    net1.load_state_dict(nn.state_dict())
    
    #Confounded data (large) with beta_1 and norm_infty_1
    t_conf = torch.Tensor(np.random.binomial(1, 0.5, size=[n_conf, 1]))
    x_conf = torch.Tensor(np.random.multivariate_normal(np.zeros(n_cov), np.identity(n_cov), size= [n_conf]))
    with torch.no_grad():
        rep, y1, y0 = net1(torch.Tensor(x_conf))    
    y_conf = t_conf * y1 + (1-t_conf) * y0 + 0.5*torch.normal(0, 1, size=(rep.size(0), 1))

    x_unc = torch.Tensor(np.random.multivariate_normal(np.zeros(n_cov), np.identity(n_cov), size= [n_unc]))
    t_unc = torch.Tensor(np.random.binomial(1, 0.5, size=[n_unc, 1]))
    with torch.no_grad():
        rep, y1, y0 = net1(x_unc)    
        y1 = net1.linear_out_1(rep) + net1.delta_1(rep)
        y0 = net1.linear_out_0(rep) + net1.delta_0(rep)
    y_unc = t_unc * y1 + (1-t_unc) * y0 + 0.5*torch.normal(0, 1, size=(rep.size(0), 1))

    #Test data
    n_test = 10000
    #Confounded data (large)
    x_test = torch.Tensor(np.random.multivariate_normal(np.zeros(n_cov), np.identity(n_cov), size= [n_test]))
    
    #with \delta = 1
    with torch.no_grad():
        rep, y1, y0 = net1(x_test)    
        y1 = net1.linear_out_1(rep) + net1.delta_1(rep)
        y0 = net1.linear_out_0(rep) + net1.delta_0(rep)
    
    cate_test = y1 - y0

    #Change weight of representation for confounded data
    net1.rep.layers[2].weight = torch.nn.Parameter(norm_infty_2 * net1.rep.layers[2].weight)#/infty_norm_l_3
    
    #Confounded data (large) WITH bias in representation
    t_conf_shared = t_conf
    x_conf_shared = x_conf
    with torch.no_grad():
        rep, y1, y0 = net1(torch.Tensor(x_conf_shared))    
    y_conf_shared = t_conf_shared * y1 + (1-t_conf_shared) * y0 + 0.5*torch.normal(0, 1, size=(rep.size(0), 1))
    
    net1.rep.layers[2].weight = torch.nn.Parameter(norm_infty_3 * net1.rep.layers[2].weight/norm_infty_2)#/infty_norm_l_3
 
    #Confounded data (large) WITH bias in representation
    with torch.no_grad():
        rep, y1, y0 = net1(torch.Tensor(x_conf_shared))    
    y_conf_shared_2 = t_conf_shared * y1 + (1-t_conf_shared) * y0 + 0.5*torch.normal(0, 1, size=(rep.size(0), 1))
    
    return {'cate_test': cate_test,'x_unc': x_unc, 't_unc': t_unc, 'y_unc': y_unc, 'x_conf': x_conf, 't_conf': t_conf, 'y_conf': y_conf, 'x_conf_shared': x_conf_shared, 't_conf_shared': t_conf_shared, 'y_conf_shared': y_conf_shared, 'x_conf_shared_2': x_conf_shared, 't_conf_shared_2': t_conf_shared, 'y_conf_shared_2': y_conf_shared_2,'x_test': x_test}

def sim_data_assum_positivity(nn, n_conf, n_unc, n_cov):
    net1 = deepcopy(nn)
    
    cov_matrix = np.identity(n_cov)
    
    #Test data
    n_test = 10000
    x_test = torch.Tensor(np.random.multivariate_normal(np.zeros(n_cov), cov_matrix, size= [n_test]))
    with torch.no_grad():
        cate_test = net1.predict_delta(x_test)
        
    
    #Confounded data
    t_conf = torch.Tensor(np.random.binomial(1, 0.5, size=[n_conf, 1]))
    x_conf = torch.Tensor(np.random.multivariate_normal(np.zeros(n_cov), cov_matrix, size= [n_conf]))
    with torch.no_grad():
        rep, y1, y0 = net1(x_conf)    
    y_conf = t_conf * y1 + (1-t_conf) * y0 + 0.5*torch.normal(0, 1, size=(rep.size(0), 1))
    
    #Unconfounded data - large overlap
    x_unc_ol_large = torch.Tensor(np.random.uniform(-3, 3, size = [n_unc, n_cov]))
    t_unc_ol_large = torch.Tensor(np.random.binomial(1, 0.5, size = [n_unc, 1]))
    with torch.no_grad():
        rep, y1, y0 = net1(x_unc_ol_large)    
        y1 = net1.linear_out_1(rep) + net1.delta_1(rep)
        y0 = net1.linear_out_0(rep) + net1.delta_0(rep)
    y_unc_ol_large = t_unc_ol_large * y1 + (1-t_unc_ol_large) * y0 + 0.5*torch.normal(0, 1, size=(rep.size(0), 1))
    
    #Unconfounded data - large overlap
    x_unc_normal_large = torch.Tensor(np.random.multivariate_normal(np.zeros(n_cov), 1*cov_matrix, size= [n_unc]))
    t_unc_normal_large = torch.Tensor(np.random.binomial(1, 0.5, size=[n_unc, 1]))
    with torch.no_grad():
        rep, y1, y0 = net1(x_unc_normal_large)    
        y1 = net1.linear_out_1(rep) + net1.delta_1(rep)
        y0 = net1.linear_out_0(rep) + net1.delta_0(rep)
    y_unc_normal_large = t_unc_normal_large * y1 + (1-t_unc_normal_large) * y0 + 0.5*torch.normal(0, 1, size=(rep.size(0), 1))
    

    #Unconfounded data - medium overlap
    x_unc_ol_medium = torch.Tensor(np.random.uniform(-1, 1, size = [n_unc, n_cov]))
    t_unc_ol_medium = torch.Tensor(np.random.binomial(1, 0.5, size = [n_unc, 1]))
    with torch.no_grad():
        rep, y1, y0 = net1(x_unc_ol_medium)    
        y1 = net1.linear_out_1(rep) + net1.delta_1(rep)
        y0 = net1.linear_out_0(rep) + net1.delta_0(rep)
    y_unc_ol_medium = t_unc_ol_medium * y1 + (1-t_unc_ol_medium) * y0 + 0.5*torch.normal(0, 1, size=(rep.size(0), 1))
    
    #Unconfounded data - large overlap
    x_unc_normal_medium = torch.Tensor(np.random.multivariate_normal(np.zeros(n_cov), 1/3*cov_matrix, size= [n_unc]))
    t_unc_normal_medium = torch.Tensor(np.random.binomial(1, 0.5, size=[n_unc, 1]))
    with torch.no_grad():
        rep, y1, y0 = net1(x_unc_normal_medium)    
        y1 = net1.linear_out_1(rep) + net1.delta_1(rep)
        y0 = net1.linear_out_0(rep) + net1.delta_0(rep)
    y_unc_normal_medium = t_unc_normal_medium * y1 + (1-t_unc_normal_medium) * y0 + 0.5*torch.normal(0, 1, size=(rep.size(0), 1))
    
    #Unconfounded data - medium overlap
    x_unc_ol_small = torch.Tensor(np.random.uniform(-0.5, 0.5, size = [n_unc, n_cov]))
    t_unc_ol_small = torch.Tensor(np.random.binomial(1, 0.5, size = [n_unc, 1]))
    with torch.no_grad():
        rep, y1, y0 = net1(x_unc_ol_small)    
        y1 = net1.linear_out_1(rep) + net1.delta_1(rep)
        y0 = net1.linear_out_0(rep) + net1.delta_0(rep)
    y_unc_ol_small = t_unc_ol_small * y1 + (1-t_unc_ol_small) * y0 + 0.5*torch.normal(0, 1, size=(rep.size(0), 1))
    
    #Unconfounded data - large overlap
    x_unc_normal_small = torch.Tensor(np.random.multivariate_normal(np.zeros(n_cov), 1/6**2*cov_matrix, size= [n_unc]))
    t_unc_normal_small = torch.Tensor(np.random.binomial(1, 0.5, size=[n_unc, 1]))
    with torch.no_grad():
        rep, y1, y0 = net1(x_unc_normal_small)    
        y1 = net1.linear_out_1(rep) + net1.delta_1(rep)
        y0 = net1.linear_out_0(rep) + net1.delta_0(rep)
    y_unc_normal_small = t_unc_normal_small * y1 + (1-t_unc_normal_small) * y0 + 0.5*torch.normal(0, 1, size=(rep.size(0), 1))
    

    
    return {'cate_test': cate_test,
            'x_unc_ol_large': x_unc_ol_large, 't_unc_ol_large': t_unc_ol_large, 'y_unc_ol_large': y_unc_ol_large, 
            'x_unc_ol_medium': x_unc_ol_medium, 't_unc_ol_medium': t_unc_ol_medium, 'y_unc_ol_medium': y_unc_ol_medium, 
            'x_unc_ol_small': x_unc_ol_small, 't_unc_ol_small': t_unc_ol_small, 'y_unc_ol_small': y_unc_ol_small, 
            'x_unc_normal_large': x_unc_normal_large, 't_unc_normal_large': t_unc_normal_large, 'y_unc_normal_large': y_unc_normal_large, 
            'x_unc_normal_medium': x_unc_normal_medium, 't_unc_normal_medium': t_unc_normal_medium, 'y_unc_normal_medium': y_unc_normal_medium, 
            'x_unc_normal_small': x_unc_normal_small, 't_unc_normal_small': t_unc_normal_small, 'y_unc_normal_small': y_unc_normal_small, 
            'x_conf': x_conf, 't_conf': t_conf, 'y_conf': y_conf, 'x_test': x_test}




def sim_data_assum_unc_obs_data(nn, n_conf, n_unc, n_cov, cov_shift = 1, noise_scale = 0.1):
    n_hidden_cov = nn.rep.layers[0].weight.shape[0]
    net = CorNet(n_cov, len(nn.rep.layers), n_hidden_cov, 0, n_hidden_cov)
    net.load_state_dict(nn.state_dict())
    #Unconfounded data (small) WITHOUT covariate shift
    x_unc = torch.Tensor(np.random.multivariate_normal(np.zeros(n_cov), np.identity(n_cov), size= [n_unc]))
    t_unc = torch.Tensor(np.random.binomial(1, 0.5, size=[n_unc, 1]))
    with torch.no_grad():
        rep, y1, y0 = net(x_unc)    
        y1 = net.linear_out_1(rep) + net.delta_1(rep)
        y0 = net.linear_out_0(rep) + net.delta_0(rep)
    y_unc = t_unc * y1 + (1-t_unc) * y0 + 0.5*torch.normal(0, 1, size=(rep.size(0), 1))
    
    #Confounded data (large)
    t_conf = torch.Tensor(np.random.binomial(1, 0.5, size=[n_conf, 1]))
    x_conf = torch.Tensor(np.random.multivariate_normal(np.zeros(n_cov)+cov_shift, np.identity(n_cov), size= [n_conf]))
    with torch.no_grad():
        rep, y1, y0 = net(torch.Tensor(x_conf))    
        y1 = net.linear_out_1(rep) + net.delta_1(rep)
        y0 = net.linear_out_0(rep) + net.delta_0(rep)
    y_conf = t_conf * y1 + (1-t_conf) * y0 + 0.5*torch.normal(0, 1, size=(rep.size(0), 1))

    #Test data
    n_test = 1000
    #Confounded data (large)
    t_test = torch.Tensor(np.random.binomial(1, 0.5, size=[n_conf, 1]))
    x_test = torch.Tensor(np.random.multivariate_normal(np.zeros(n_cov)+cov_shift, np.identity(n_cov), size= [n_conf]))
    
    with torch.no_grad():
        rep, y1, y0 = net(x_test)    
        y1 = net.linear_out_1(rep) + net.delta_1(rep)
        y0 = net.linear_out_0(rep) + net.delta_0(rep)
    
    cate_test = y1 - y0
    
    return {'cate_test': cate_test,'x_unc': x_unc, 't_unc': t_unc, 'y_unc': y_unc, 'x_conf': x_conf, 't_conf': t_conf, 'y_conf': y_conf, 'x_test': x_test}

