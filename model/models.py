import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.helper import EarlyStopping, data_aug
from model.functions import *
#from Ridge import *

'''
Representation Network
'''
class RepNet(nn.Module):
    def __init__(self, n_cov, n_hidden, d_hidden):
        super(RepNet, self).__init__()
        self.n_cov = n_cov
        self.n_hidden = n_hidden
        self.d_hidden = d_hidden
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        #self.batchnorms = nn.ModuleList()
        for k in range(self.n_hidden):
            #self.batchnorms.append(nn.BatchNorm1d(d_hidden))
            if k<self.n_hidden-1:
                self.activations.append(nn.Tanh())
            if k==0:
                self.layers.append(nn.Linear(n_cov, d_hidden))
            else:
                self.layers.append(nn.Linear(d_hidden, d_hidden))
            
    def forward(self, x):
        x1 = x 
        for i in range(self.n_hidden):
            if i<self.n_hidden-1:
                #x1 = self.activations[i](self.batchnorms[i](self.layers[i](x1)))
                x1 = self.activations[i](self.layers[i](x1))
            else:
                x1 = self.layers[i](x1)
        return x1


'''
Architecture: one representation and two outcome heads, \ie, Y_1(x) = h_1(\Phi(x)) and Y_0(x) = h_0(\Phi(x))
h_i are linear
repreesntation balanced (between conf and unc data) via Augmented Distribution Alignment
'''

class CorNet_ADA(nn.Module):
    def __init__(self, n_cov, n_hidden_rep, d_hidden_rep, n_hidden_out, d_hidden_out):
        super(CorNet_ADA, self).__init__()
        #Variables
        self.n_cov = n_cov
        self.n_hidden_rep = n_hidden_rep
        self.d_hidden_rep = d_hidden_rep
        self.n_hidden_out = n_hidden_out
        self.d_hidden_out = d_hidden_out
        
        #Representation networks
        self.rep = RepNet(self.n_cov, self.n_hidden_rep, self.d_hidden_rep)
        self.out1 = RepNet(self.d_hidden_rep, self.n_hidden_out, self.d_hidden_out)
        self.out0 = RepNet(self.d_hidden_rep, self.n_hidden_out, self.d_hidden_out)
        
        #Linear outcome functions
        self.linear_out_1 = nn.Linear(self.d_hidden_out, 1)
        self.linear_out_0 = nn.Linear(self.d_hidden_out, 1)
        self.delta_1 = nn.Linear(self.d_hidden_out, 1)
        self.delta_0 = nn.Linear(self.d_hidden_out, 1)
        self.linear_disc = nn.Linear(self.d_hidden_out, 1)
        
        #self.linear_out_1_ret = nn.Linear(self.d_hidden_out, 1)
        #self.linear_out_0_ret = nn.Linear(self.d_hidden_out, 1)
        
    def forward(self, x, grl_lambd=1):
        x1 = x
        x1 = self.rep(x1)

        x1d = GradientReversalFn.apply(x1, grl_lambd) #Gradient Reversal Layer for the discriminator
        if self.n_hidden_out > 0:
            outcome_1 = self.linear_out_1(self.out1(x1))
            outcome_0 = self.linear_out_0(self.out0(x1))
        else:
            outcome_1 = self.linear_out_1(x1)
            outcome_0 = self.linear_out_0(x1)
        
        disc_prob = torch.sigmoid(self.linear_disc(x1d))
            
        return [x1, outcome_1, outcome_0, disc_prob]
    
    def _train(self, x, y, t, x_uc, t_uc, x_val, y_val, t_val, alpha, lambd_ada, lambd_h_c, batch_size, lr_train, n_epochs, balancing = True, early_stop = False, out_norm = 2):
        optim = torch.optim.SGD(self.parameters(), lr=lr_train)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.1)
        loss_fct = torch.nn.MSELoss()
        loss_BCE = torch.nn.BCELoss()
        early_stopping = EarlyStopping()
        #Training
        best_val_loss = 1e1000
        for e in range(n_epochs):
            p = e / n_epochs
            grl_lambd = lambd_ada 
            
            #Random permutation for mini-batch
            perm = torch.randperm(x.size()[0])
            loss_batch = []
            reg_ada_batch = []
            reg_w_batch = []
            
            for b in range(0, x.size()[0], batch_size):
                ind = perm[b: b + batch_size]
                batch_x, batch_y, batch_t = x[ind, :], y[ind, :], t[ind, :]
            
                optim.zero_grad()
                rep, out1, out0, disc_prob = self.forward(batch_x, grl_lambd)
                        
                batch_pred = batch_t * out1 + (1-batch_t) * out0

                loss = loss_fct(batch_pred, batch_y)
                #Regularize the confounded hypothesis
                if out_norm == 2:
                    reg1 = torch.norm(self.linear_out_1.weight, p=2) + torch.norm(self.linear_out_1.bias, p=2)
                    reg0 = torch.norm(self.linear_out_0.weight, p=2) + torch.norm(self.linear_out_0.bias, p=2)
                if out_norm == 1:
                    reg1 = torch.norm(self.linear_out_1.weight, p=1) + torch.norm(self.linear_out_1.bias, p=2)
                    reg0 = torch.norm(self.linear_out_0.weight, p=1) + torch.norm(self.linear_out_0.bias, p=2)
                reg_w = reg1 + reg0

                if balancing:
                    #Generate interpolated covariate samples
                    x_inter, z_inter = data_aug(batch_x, x_uc, alpha = alpha)
                    rep_inter, out_inter1, out_inter0, domain_prob_inter = self.forward(x_inter, grl_lambd)
                    reg_ada = loss_BCE(domain_prob_inter, z_inter)
                    loss_total = loss + lambd_ada*reg_ada + lambd_h_c * reg_w
                else:
                    loss_total = loss + lambd_h_c * reg_w
                
                loss_total.backward()
                optim.step()

                loss_batch.append(loss.item())
                if balancing:
                    reg_ada_batch.append(reg_ada.item())
                reg_w_batch.append(reg_w.item())

            if early_stop:
                #Validation loss, i.e., objective on the validation set
                with torch.no_grad():
                    rep, out1, out0, disc_prob = self.forward(x_val, grl_lambd)
                    
                    pred = t_val * out1 + (1-t_val) * out0
                    loss = loss_fct(pred, y_val)
                    
                    val_epoch_loss = loss.item() 
        
                    if best_val_loss > val_epoch_loss:
                        best_params = self.state_dict()
        
                    early_stopping(val_epoch_loss)
                    if early_stopping.early_stop:
                        self.load_state_dict(best_params)
                        break

            if balancing:
                if e % 250 == 0:
                    scheduler.step()    
                    
            else:
                if e % 250 == 0:
                    scheduler.step()    
                    
        
    def _retrain_delta(self, x, y, t, reg_coef, lr, n_epochs, bias_norm = 1, bias_learning = True):
        optim_1 = torch.optim.SGD(self.delta_1.parameters(), lr=lr) 
        optim_0 = torch.optim.SGD(self.delta_0.parameters(), lr=lr) 
        scheduler_1 = torch.optim.lr_scheduler.ExponentialLR(optim_1, gamma=0.1)
        scheduler_0 = torch.optim.lr_scheduler.ExponentialLR(optim_0, gamma=0.1)
        loss_fct1 = torch.nn.MSELoss()
        loss_fct0 = torch.nn.MSELoss()
        
        with torch.no_grad():
            rep1, out1_biased, out0, foo = self.forward(x[t.flatten()==1,:], 1)
            rep0, out1, out0_biased, foo = self.forward(x[t.flatten()==0,:], 1)
            
        for e in range(n_epochs):
            optim_1.zero_grad()
            optim_0.zero_grad()
            out1 = self.delta_1(rep1)
            out0 = self.delta_0(rep0)
            
            if bias_learning:
                bias1 = y[t.flatten()==1] - out1_biased
                bias0 = y[t.flatten()==0] - out0_biased
            else:
                bias1 = y[t.flatten()==1]
                bias0 = y[t.flatten()==0]
            
            loss1 = loss_fct1(out1, bias1)
            reg1 = torch.norm(self.delta_1.weight, p=bias_norm) + torch.norm(self.delta_1.bias, p=bias_norm)  
            loss_reg_1 = loss1 + reg_coef*reg1
            loss_reg_1.backward()
            optim_1.step()
            
            loss0 = loss_fct0(out0, bias0)
            reg0 = torch.norm(self.delta_0.weight, p=bias_norm) + torch.norm(self.delta_0.bias, p=bias_norm)  
            loss_reg_0 = loss0 + reg_coef*reg0
            loss_reg_0.backward()
            optim_0.step()

            if e % 250 == 0:
                scheduler_1.step()    
                scheduler_0.step()    

    def predict_delta(self, x):
        self.eval()
        with torch.no_grad():
            rep, out1, out0, foo = self.forward(x, 1)
            pred1 = self.delta_1(rep) + out1
            pred0 = self.delta_0(rep) + out0
        return pred1 - pred0


    def predict_naive(self, x):
        self.eval()
        with torch.no_grad():
            rep, out1, out0, foo = self.forward(x, 1)
        return out1 - out0
    
    def predict_out(self, x):
        self.eval()
        with torch.no_grad():
            rep, out1, out0, foo = self.forward(x)
            pred1 = self.delta_1(rep)
            pred0 = self.delta_0(rep)
        return pred1 - pred0

    def predict_single_outcomes(self, x):
        self.eval()
        with torch.no_grad():
            rep, out1, out0, foo = self.forward(x, 1)
            pred1 = self.delta_1(rep) + out1
            pred0 = self.delta_0(rep) + out0
        return pred1, pred0


'''
Architecture: one representation and two outcome heads, \ie, Y_1(x) = h_1(\Phi(x)) and Y_0(x) = h_0(\Phi(x))
h_i are linear
NO repreesntation balanced (between conf and unc data)
'''

class CorNet(nn.Module):
    def __init__(self, n_cov, n_hidden_rep, d_hidden_rep, n_hidden_out, d_hidden_out):
        super(CorNet, self).__init__()
        #Variables
        self.n_cov = n_cov
        self.n_hidden_rep = n_hidden_rep
        self.d_hidden_rep = d_hidden_rep
        self.n_hidden_out = n_hidden_out
        self.d_hidden_out = d_hidden_out
        
        #Representation networks
        self.rep = RepNet(self.n_cov, self.n_hidden_rep, self.d_hidden_rep)
        self.out1 = RepNet(self.d_hidden_rep, self.n_hidden_out, self.d_hidden_out)
        self.out0 = RepNet(self.d_hidden_rep, self.n_hidden_out, self.d_hidden_out)
        
        #Linear outcome functions
        self.linear_out_1 = nn.Linear(self.d_hidden_out, 1)
        self.linear_out_0 = nn.Linear(self.d_hidden_out, 1)
        self.delta_1 = nn.Linear(self.d_hidden_out, 1)
        self.delta_0 = nn.Linear(self.d_hidden_out, 1)
        
        
    def forward(self, x):
        x1 = x
        x1 = self.rep(x1)

        if self.n_hidden_out > 0:
            outcome_1 = self.linear_out_1(self.out1(x1))
            outcome_0 = self.linear_out_0(self.out0(x1))
        else:
            outcome_1 = self.linear_out_1(x1)
            outcome_0 = self.linear_out_0(x1)
            
        return [x1, outcome_1, outcome_0]
    
    def _train(self, x, y, t, x_val, y_val, t_val, lambd_h_c, batch_size, lr, n_epochs, early_stop = False):
        optim = torch.optim.SGD(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.1)
        loss_fct = torch.nn.MSELoss()
        #Training
        best_val_loss = 1e1000
        for e in range(n_epochs):
            
            #Random permutation for mini-batch
            perm = torch.randperm(x.size()[0])
            loss_batch = []
            reg_w_batch = []
            
            for b in range(0, x.size()[0], batch_size):
                ind = perm[b: b + batch_size]
                batch_x, batch_y, batch_t = x[ind, :], y[ind, :], t[ind, :]
            
                optim.zero_grad()
                rep, out1, out0 = self.forward(batch_x)
                batch_pred = batch_t * out1 + (1-batch_t) * out0
                loss = loss_fct(batch_pred, batch_y)
                
                #Regularize the confounded hypothesis
                reg1 = torch.norm(self.linear_out_1.weight, p=2) + torch.norm(self.linear_out_1.bias, p=2)
                reg0 = torch.norm(self.linear_out_0.weight, p=2) + torch.norm(self.linear_out_0.bias, p=2)
                reg_w = reg1 + reg0

                loss_total = loss + lambd_h_c * reg_w
                loss_total.backward()
                optim.step()

                loss_batch.append(loss.item())
                reg_w_batch.append(reg_w.item())

            if early_stop:
                #Validation loss, i.e., objective on the validation set
                with torch.no_grad():
                    rep, out1, out0 = self.forward(x_val)
                    
                    pred = t_val * out1 + (1-t_val) * out0
                    loss = loss_fct(pred, y_val)
                        
                    #Regularize the confounded hypothesis
                    reg1 = torch.norm(self.linear_out_1.weight, p=2)
                    reg0 = torch.norm(self.linear_out_0.weight, p=2)
                    reg_w = reg1 + reg0
        
                    val_epoch_loss = loss.item() + lambd_h_c * reg_w.item()
        
                    if best_val_loss > val_epoch_loss:
                        best_params = self.state_dict()
        
                    early_stopping(val_epoch_loss)
                    if early_stopping.early_stop:
                        self.load_state_dict(best_params)
                        break


            if e % 250 == 0:
                scheduler.step()    

    def _retrain_delta(self, x, y, t, reg_coef, lr, n_epochs, bias_norm = 1, bias_learning = True):
        optim_1 = torch.optim.SGD(self.delta_1.parameters(), lr=lr) 
        optim_0 = torch.optim.SGD(self.delta_0.parameters(), lr=lr) 
        scheduler_1 = torch.optim.lr_scheduler.ExponentialLR(optim_1, gamma=0.1)
        scheduler_0 = torch.optim.lr_scheduler.ExponentialLR(optim_0, gamma=0.1)
        loss_fct1 = torch.nn.MSELoss()
        loss_fct0 = torch.nn.MSELoss()
        
        with torch.no_grad():
            rep1, out1_biased, out0 = self.forward(x[t.flatten()==1,:])
            rep0, out1, out0_biased = self.forward(x[t.flatten()==0,:])
            
        for e in range(n_epochs):
                
            optim_1.zero_grad()
            optim_0.zero_grad()
            out1 = self.delta_1(rep1)
            out0 = self.delta_0(rep0)
            
            if bias_learning:
                bias1 = y[t.flatten()==1] - out1_biased
                bias0 = y[t.flatten()==0] - out0_biased
            else:
                bias1 = y[t.flatten()==1]
                bias0 = y[t.flatten()==0]
                
            if bias_learning:
                loss1 = loss_fct1(out1, bias1)
                reg1 = torch.norm(self.delta_1.weight, p=bias_norm) + torch.norm(self.delta_1.bias, p=bias_norm)
                loss_reg_1 = loss1 + reg_coef*reg1
                loss_reg_1.backward()
                optim_1.step()
                
                loss0 = loss_fct0(out0, bias0)
                reg0 = torch.norm(self.delta_0.weight, p=bias_norm) + torch.norm(self.delta_0.bias, p=bias_norm)
                loss_reg_0 = loss0 + reg_coef*reg0
                loss_reg_0.backward()
                optim_0.step()
            else:
                loss1 = loss_fct1(out1, bias1)
                reg1 = torch.norm(self.delta_1.weight, p=bias_norm) + torch.norm(self.delta_1.bias, p=bias_norm)
                loss_reg_1 = loss1 + reg_coef*reg1
                loss_reg_1.backward()
                optim_1.step()
                
                loss0 = loss_fct0(out0, bias0)
                reg0 = torch.norm(self.delta_0.weight, p=bias_norm) + torch.norm(self.delta_0.bias, p=bias_norm)
                loss_reg_0 = loss0 + reg_coef*reg0
                loss_reg_0.backward()
                optim_0.step()               

            if e % 250 == 0:
                scheduler_1.step()    
                scheduler_0.step()    

    def predict_naive(self, x):
        self.eval()
        with torch.no_grad():
            rep, out1, out0 = self.forward(x)
        return out1 - out0
    
    def predict_delta(self, x):
        self.eval()
        with torch.no_grad():
            rep, out1, out0 = self.forward(x)
            pred1 = out1 + self.delta_1(rep)
            pred0 = out0 + self.delta_0(rep)
        return pred1 - pred0

    def predict_out(self, x):
        self.eval()
        with torch.no_grad():
            rep, out1, out0 = self.forward(x)
            pred1 = self.delta_1(rep)
            pred0 = self.delta_0(rep)
        return pred1 - pred0


'''
1-step MTL baseline
'''

class MTL(nn.Module):
    def __init__(self, n_cov, n_hidden_rep, d_hidden_rep, n_hidden_out, d_hidden_out):
        super(MTL, self).__init__()
        #Variables
        self.n_cov = n_cov
        self.n_hidden_rep = n_hidden_rep
        self.d_hidden_rep = d_hidden_rep
        self.n_hidden_out = n_hidden_out
        self.d_hidden_out = d_hidden_out
        
        #Representation networks
        self.rep = RepNet(self.n_cov, self.n_hidden_rep, self.d_hidden_rep)
        self.out1 = RepNet(self.d_hidden_rep, self.n_hidden_out, self.d_hidden_out)
        self.out0 = RepNet(self.d_hidden_rep, self.n_hidden_out, self.d_hidden_out)

        self.out1_u = RepNet(self.d_hidden_rep, self.n_hidden_out, self.d_hidden_out)
        self.out0_u = RepNet(self.d_hidden_rep, self.n_hidden_out, self.d_hidden_out)
        
        #Linear outcome functions
        self.linear_out_1 = nn.Linear(self.d_hidden_out, 1)
        self.linear_out_0 = nn.Linear(self.d_hidden_out, 1)
        self.linear_out_unc_0 = nn.Linear(self.d_hidden_out, 1)
        self.linear_out_unc_1 = nn.Linear(self.d_hidden_out, 1)
        self.delta_1 = nn.Linear(self.d_hidden_out, 1)
        self.delta_0 = nn.Linear(self.d_hidden_out, 1)
        self.linear_disc = nn.Linear(self.d_hidden_out, 1)
        
        
    def forward(self, x, grl_lambd=1):
        x1 = x
        x1 = self.rep(x1)

        x1d = GradientReversalFn.apply(x1, grl_lambd) #Gradient Reversal Layer for the discriminator
        if self.n_hidden_out > 0:
            outcome_1 = self.linear_out_1(self.out1(x1))
            outcome_0 = self.linear_out_0(self.out0(x1))
            outcome_1_u = self.linear_out_unc_1(self.out1_u(x1))
            outcome_0_u = self.linear_out_unc_0(self.out0_u(x1))
        else:
            outcome_1 = self.linear_out_1(x1)
            outcome_0 = self.linear_out_0(x1)
            outcome_1_u = self.linear_out_unc_1(x1)
            outcome_0_u = self.linear_out_unc_0(x1)

        
        disc_prob = torch.sigmoid(self.linear_disc(x1d))
            
        return [x1, outcome_1, outcome_0, outcome_1_u, outcome_0_u, disc_prob]
    
    def _train(self, x, y, t, x_uc, t_uc, y_uc, x_val, y_val, t_val, alpha, lambd_ada, lambd_h_c, batch_size, lr_train, n_epochs, balancing = True, early_stop = False, out_norm = 2):
        optim = torch.optim.SGD(self.parameters(), lr=lr_train)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.1)
        loss_fct = torch.nn.MSELoss()
        loss_BCE = torch.nn.BCELoss()
        early_stopping = EarlyStopping()
        #Training
        best_val_loss = 1e1000
        for e in range(n_epochs):
            p = e / n_epochs
            grl_lambd = lambd_ada
            
            #Random permutation for mini-batch
            perm = torch.randperm(x.size()[0])
            loss_batch = []
            reg_ada_batch = []
            reg_w_batch = []
            
            for b in range(0, x.size()[0], batch_size):
                ind = perm[b: b + batch_size]
                batch_x, batch_y, batch_t = x[ind, :], y[ind, :], t[ind, :]
            
                optim.zero_grad()
                n_u = x_uc.shape[0]
                n_c = batch_x.shape[0]
                x_cat=torch.cat((batch_x, x_uc))
                rep, out1, out0, out1_u, out0_u, disc_prob = self.forward(x_cat, grl_lambd)
                        
                batch_pred = batch_t * out1[:n_c] + (1-batch_t) * out0[:n_c]
                pred_u = t_uc * out1_u[n_c:] + (1-t_uc) * out0_u[n_c:]

                loss = loss_fct(batch_pred, batch_y) + loss_fct(pred_u, y_uc)
                #Regularize the confounded hypothesis
                if out_norm == 2:
                    reg1 = torch.norm(self.linear_out_1.weight, p=2)
                    reg0 = torch.norm(self.linear_out_0.weight, p=2)
                    reg1_u = torch.norm(self.linear_out_unc_1.weight, p=2)
                    reg0_u = torch.norm(self.linear_out_unc_0.weight, p=2)
                if out_norm == 1:
                    reg1 = torch.norm(self.linear_out_1.weight, p=1)
                    reg0 = torch.norm(self.linear_out_0.weight, p=1)
                    reg1_u = torch.norm(self.linear_out_unc_1.weight, p=1)
                    reg0_u = torch.norm(self.linear_out_unc_0.weight, p=1)
                reg_w = reg1 + reg0 + reg1_u + reg0_u

                if balancing:
                    #Generate interpolated covariate samples
                    x_inter, z_inter = data_aug(batch_x, x_uc, alpha = alpha)
                    rep_inter, out_inter1, out_inter0, out_inter1_u, out_inter0_u, domain_prob_inter = self.forward(x_inter, grl_lambd)
                    reg_ada = loss_BCE(domain_prob_inter, z_inter)
                    loss_total = loss + lambd_ada*reg_ada + lambd_h_c * reg_w
                else:
                    loss_total = loss + lambd_h_c * reg_w
                
                loss_total.backward()
                optim.step()

                loss_batch.append(loss.item())
                if balancing:
                    reg_ada_batch.append(reg_ada.item())
                reg_w_batch.append(reg_w.item())

            if early_stop:
                #Validation loss, i.e., objective on the validation set
                with torch.no_grad():
                    rep, out1, out0, disc_prob = self.forward(x_val, grl_lambd)
                    
                    pred = t_val * out1 + (1-t_val) * out0
                    loss = loss_fct(pred, y_val)
        
                    val_epoch_loss = loss.item()
        
                    if best_val_loss > val_epoch_loss:
                        best_params = self.state_dict()
        
                    early_stopping(val_epoch_loss)
                    if early_stopping.early_stop:
                        self.load_state_dict(best_params)
                        break

            if balancing:
                if e % 250 == 0:
                    scheduler.step()    
            else:
                if e % 250 == 0:
                    scheduler.step()    

        
    def _retrain_delta(self, x, y, t, reg_coef, lr, n_epochs, delta_norm = 1):
        optim_1 = torch.optim.SGD(self.delta_1.parameters(), lr=lr) 
        optim_0 = torch.optim.SGD(self.delta_0.parameters(), lr=lr) 
        scheduler_1 = torch.optim.lr_scheduler.ExponentialLR(optim_1, gamma=0.1)
        scheduler_0 = torch.optim.lr_scheduler.ExponentialLR(optim_0, gamma=0.1)
        loss_fct1 = torch.nn.MSELoss()
        loss_fct0 = torch.nn.MSELoss()
        
        with torch.no_grad():
            rep1, out1_biased, out0, o1_u, o0_u, foo = self.forward(x[t.flatten()==1,:], 1)
            rep0, out1, out0_biased, o1_u, o0_u, foo = self.forward(x[t.flatten()==0,:], 1)
            
        for e in range(n_epochs):
            optim_1.zero_grad()
            optim_0.zero_grad()
            out1 = self.delta_1(rep1)
            out0 = self.delta_0(rep0)
            
            bias1 = y[t.flatten()==1] - out1_biased
            bias0 = y[t.flatten()==0] - out0_biased
            
            loss1 = loss_fct1(out1, bias1)
            if delta_norm == 1:
                reg1 = torch.norm(self.delta_1.weight, p=1)    
            if delta_norm == 2:
                reg1 = torch.norm(self.delta_1.weight, p=2)    
            loss_reg_1 = loss1 + reg_coef*reg1
            loss_reg_1.backward()
            optim_1.step()
            
            loss0 = loss_fct0(out0, bias0)
            if delta_norm == 1:
                reg0 = torch.norm(self.delta_0.weight, p=1)
            if delta_norm == 2:
                reg0 = torch.norm(self.delta_0.weight, p=2)
            loss_reg_0 = loss0 + reg_coef*reg0
            loss_reg_0.backward()
            optim_0.step()

            if e % 250 == 0:
                scheduler_1.step()    
                scheduler_0.step()    

    def predict_delta(self, x):
        self.eval()
        with torch.no_grad():
            rep, out1, out0, out1_u, out0_u, foo = self.forward(x, 1)
            pred1 = self.delta_1(rep) + out1
            pred0 = self.delta_0(rep) + out0
        return pred1 - pred0


    def predict_naive(self, x):
        self.eval()
        with torch.no_grad():
            rep, out1, out0, out1_u, out0_u, foo = self.forward(x, 1)
        return out1_u - out0_u





'''
Architecture: one representation and two outcome heads, \ie, Y_1(x) = h_1(\Phi(x)) and Y_0(x) = h_0(\Phi(x))
h_i are linear
NO repreesntation balanced (between conf and unc data)
'''

class WeightNet(nn.Module):
    def __init__(self, n_cov, n_hidden_rep, d_hidden_rep, n_hidden_out, d_hidden_out):
        super(WeightNet, self).__init__()
        #Variables
        self.n_cov = n_cov
        self.n_hidden_rep = n_hidden_rep
        self.d_hidden_rep = d_hidden_rep
        self.n_hidden_out = n_hidden_out
        self.d_hidden_out = d_hidden_out
        
        #Representation networks
        self.rep = RepNet(self.n_cov, self.n_hidden_rep, self.d_hidden_rep)
        self.out1 = RepNet(self.d_hidden_rep, self.n_hidden_out, self.d_hidden_out)
        self.out0 = RepNet(self.d_hidden_rep, self.n_hidden_out, self.d_hidden_out)
        
        #Linear outcome functions
        self.linear_out_1 = nn.Linear(self.d_hidden_out, 1)
        self.linear_out_0 = nn.Linear(self.d_hidden_out, 1)
        
        
    def forward(self, x):
        x1 = x
        x1 = self.rep(x1)

        if self.n_hidden_out > 0:
            outcome_1 = self.linear_out_1(self.out1(x1))
            outcome_0 = self.linear_out_0(self.out0(x1))
        else:
            outcome_1 = self.linear_out_1(x1)
            outcome_0 = self.linear_out_0(x1)
            
        return [x1, outcome_1, outcome_0]
    
    def _train(self, x_c, y_c, t_c, x_u, y_u, t_u, x_val, y_val, t_val, lambd_h_c, lambd_weight, batch_size, lr, n_epochs, early_stop = False):
        optim = torch.optim.SGD(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.1)
        loss_fct = torch.nn.MSELoss()
        early_stopping = EarlyStopping()
        n_c = x_c.size()[0]
        n_u = x_u.size()[0]
        #Training
        best_val_loss = 1e1000
        for e in range(n_epochs):
            
            optim.zero_grad()
            x = torch.cat((x_u, x_c))
            t = torch.cat((t_u, t_c))
            y = torch.cat((y_u, y_c))
            
            rep, out1, out0 = self.forward(x)
            pred_c = t[n_u:] * out1[n_u:] + (1-t[n_u:]) * out0[n_u:]
            pred_u = t[:n_u] * out1[:n_u] + (1-t[:n_u]) * out0[:n_u]
            loss_c = loss_fct(pred_c, y[n_u:])
            loss_u = loss_fct(pred_u, y[:n_u])
            loss = 1/(lambd_weight*n_u + n_c)*(lambd_weight*n_u*loss_u + n_c*loss_c)
            
            #Regularize the confounded hypothesis
            reg1 = torch.norm(self.linear_out_1.weight, p=2)
            reg0 = torch.norm(self.linear_out_0.weight, p=2)
            reg_w = reg1 + reg0

            loss_total = loss + lambd_h_c * reg_w
            loss_total.backward()
            optim.step()

            if early_stop:
                #Validation loss, i.e., objective on the validation set
                with torch.no_grad():
                    rep, out1, out0 = self.forward(x_val)
                    
                    pred = t_val * out1 + (1-t_val) * out0
                    loss = loss_fct(pred, y_val)
                        
                    #Regularize the confounded hypothesis
                    reg1 = torch.norm(self.linear_out_1.weight, p=1)
                    reg0 = torch.norm(self.linear_out_0.weight, p=1)
                    reg_w = reg1 + reg0
        
                    val_epoch_loss = loss.item() + lambd_h_c * reg_w.item()
        
                    if best_val_loss > val_epoch_loss:
                        best_params = self.state_dict()
        
                    early_stopping(val_epoch_loss)
                    if early_stopping.early_stop:
                        self.load_state_dict(best_params)
                        break


            if e % 250 == 0:
                scheduler.step()    
       
    def predict(self, x):
        self.eval()
        with torch.no_grad():
            rep, out1, out0 = self.forward(x)
        return out1 - out0
    



