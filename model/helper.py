from simulation_studies.simdata import *
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import NearestNeighbors
import numpy as np

def load_data(sim_tag = 'l'):
    if sim_tag == 'l':
        sim_dict = sim_data_low_dim(1000, 100, noise_scale = 0)
        return sim_dict
        
    if sim_tag == 'h':
        sim_dict = sim_data_high_dim(1000, 100, noise_scale = 0)
        return sim_dict
    
    if sim_tag == 'c':   
        #Import real-world data
        df_all = pd.read_csv('data_all', index_col=0)
        df_unc = pd.read_csv('data_unc', index_col=0)
        df_conf = pd.read_csv('data_conf', index_col=0)
        
        #Full data
        x_all = df_all.values[:, 2:]
        t_all = df_all['treatment'].values.reshape(-1,1)
        y_all = df_all['outcome'].values.reshape(-1,1)
        
        #Unconfouded data
        x_unc = df_unc.values[:, 2:]
        t_unc = df_unc['treatment'].values.reshape(-1,1)
        y_unc = df_unc['outcome'].values.reshape(-1,1)
        
        #Confounded data
        x_conf = df_conf.values[:, 2:]
        t_conf = df_conf['treatment'].values.reshape(-1,1)
        y_conf = df_conf['outcome'].values.reshape(-1,1)
        
        #Test data
        x_test = x_all
        return {'x_unc': x_unc, 't_unc': t_unc, 'y_unc': y_unc, 'u_unc': u_unc, 'x_conf': x_conf, 't_conf': t_conf, 'y_conf': y_conf, 'u_conf': u_conf, 'x_test': x_test}
    
#SVC to predict prob of a sample coming from conf data or unc data
def d_score(x_conf, x_unc):
    s_conf = torch.ones(x_conf.size())
    s_unc = torch.zeros(x_unc.size())
    x_cat = torch.cat((x_conf, x_unc))
    s_cat = torch.cat((s_conf, s_unc)).flatten()
    #clf = LogisticRegression().fit(x_cat, s_cat)
    clf = SVC(gamma='auto', probability=True)
    clf.fit(x_cat, s_cat)
    d_score_conf = clf.predict_proba(x_conf)
    d_score_unc = clf.predict_proba(x_unc)
    
    return [torch.Tensor(d_score_conf[:, 1]), torch.Tensor(d_score_unc[:, 1])]

#Get points in covariate space for MDPM
def points(x_conf, x_unc, d_conf, d_unc):
    i = torch.argmax(torch.abs(d_conf-0.5))
    j = torch.argmax(torch.abs(d_unc-0.5))
    k = torch.argmax(torch.abs(d_unc-d_conf[i]))
    m = torch.argmax(torch.abs(d_conf-d_unc[j]))
    
    return [i,j,k,m]

#Balancing with middle point distance minimization (MPD)
def MPD(rep_conf, rep_unc, points):
    i, j, k, m = points
    z_i = rep_conf[i]
    z_j = rep_unc[j]
    z_k = rep_unc[k]
    z_m = rep_conf[m]
    
    return sum(torch.pow((z_i+z_m)/2 - (z_j+z_k)/2, 2))


def data_aug(x, x_uc, alpha):
    n_conf = x.shape[0]
    n_cov = x.shape[1]
    lam = torch.distributions.beta.Beta(torch.tensor(alpha), torch.tensor(alpha)).sample(torch.Size([n_conf])).reshape(-1,1)
    ind = np.random.choice(x_uc.shape[0], n_conf, replace=True) 
    x_rand = x_uc[ind]#np.random.choice(x_uc, size=[n_conf, n_cov], replace=True)
    x_inter = lam * x_rand + (1-lam) * x
    z_inter = lam * 0 + (1-lam) * 1
    return [x_inter, z_inter]


def data_aug_within_domain(x, x_uc, alpha):
    n_conf = x.shape[0]
    lam = torch.distributions.beta.Beta(torch.tensor([alpha]), torch.tensor([alpha])).sample(torch.Size([n_conf]))
    x_rand = np.random.choice(x_uc.flatten(), size=[n_conf, 2], replace=True)
    x_inter = lam * x_rand[:, 0].reshape(-1,1) + (1-lam) * x_rand[:, 1].reshape(-1,1)
    z_inter = torch.zeros_like(x_inter)
    x_cat = torch.cat((x_inter, x))
    z_cat = torch.cat((z_inter, torch.ones_like(x)))
    return [x_cat, z_cat]


def data_aug_cross(x, x_uc, alpha):
    n_conf = x.shape[0]
    n_unc = x_uc.shape[0]
    
    lam = torch.distributions.beta.Beta(torch.tensor([alpha]), torch.tensor([alpha])).sample(torch.Size([n_conf * n_unc]))
    #x_rand = np.random.choice(x_uc.flatten(), size=[n_conf, 1], replace=True)
    x_inter = torch.ones(size=[n_unc * n_conf, 1])
    z_inter = torch.ones(size=[n_unc * n_conf, 1])
    j=0
    for i in range(0, n_unc * n_conf, n_conf):
        x_inter[i:i+n_conf] = lam[i:i+n_conf] * x_uc[j] + (1-lam[i:i+n_conf]) * x
        z_inter[i:i+n_conf] = lam[i:i+n_conf] * 0 + (1-lam[i:i+n_conf]) * 1
        j += 1
    return [x_inter, z_inter]


def data_aug_modi(x, x_uc, alpha):
    n_conf = x.shape[0]
    lam = torch.distributions.beta.Beta(torch.tensor([alpha]), torch.tensor([alpha])).sample(torch.Size([n_conf]))
    lambd = torch.max(lam, 1-lam)
    x_rand = np.random.choice(x_uc.flatten(), size=[n_conf, 1], replace=True)
    x_inter = (1-lambd) * x_rand + lambd * x
    z_inter = (1-lambd) * 0 + lambd * 1
    return [x_inter, z_inter]

def cross_sample_aug(x, x_uc, y, y_uc, t, t_uc, alpha):
    n_conf = x.shape[0]
    lam = torch.distributions.beta.Beta(torch.tensor([alpha]), torch.tensor([alpha])).sample(torch.Size([n_conf]))
    
    x_1 = x[t.flatten() == 1, :]
    x_0 = x[t.flatten() == 0, :]

    y_1 = y[t.flatten() == 1, :]
    y_0 = y[t.flatten() == 0, :]
    
    x_uc_1 = x_uc[t_uc.flatten() == 1, :]
    x_uc_0 = x_uc[t_uc.flatten() == 0, :]

    y_uc_1 = y_uc[t_uc.flatten() == 1, :]
    y_uc_0 = y_uc[t_uc.flatten() == 0, :]

    ind_rand_1 = np.random.choice(range(x_uc_1.shape[0]), size=[x_1.shape[0], 1], replace=True)
    ind_rand_0 = np.random.choice(range(x_uc_0.shape[0]), size=[x_0.shape[0], 1], replace=True)
    
    x_rand_1 = x_uc_1[ind_rand_1.flatten(), :]
    x_rand_0 = x_uc_0[ind_rand_0.flatten(), :]
    
    y_rand_1 = y_uc_1[ind_rand_1.flatten(), :]
    y_rand_0 = y_uc_0[ind_rand_0.flatten(), :]
    
    x_inter = lam * torch.cat((x_rand_1, x_rand_0)) + (1-lam) * torch.cat((x_1, x_0))
    z_inter = lam * 0 + (1-lam) * 1
    y_inter = lam * torch.cat((y_rand_1, y_rand_0)) + (1-lam) * torch.cat((y_1, y_0))
    t_inter = torch.cat((torch.ones_like(y_1), torch.zeros_like(y_0)))
    return [x_inter, z_inter, y_inter, t_inter]

def pseudo_pehe(x, t, y, cate_hat):
    x_t = x[(t == 1).flatten(), :]
    x_c = x[(t == 0).flatten(), :]

    nbrs_treatment = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(x_t)
    nbrs_control = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(x_c)

    dist_t, ind_t = nbrs_treatment.kneighbors(x)
    dist_c, ind_c = nbrs_control.kneighbors(x)
    
    y_t, y_c = y[ind_t.flatten()].reshape(-1,1), y[ind_c.flatten()].reshape(-1,1)
    pseudo_cate = y_t - y_c
    
    return ((pseudo_cate - cate_hat)**2).mean()
 
def pseudo_cate(x, t, y, x_a, t_a, y_a):
    x_t = x_a[(t_a == 1).flatten(), :]
    x_c = x_a[(t_a == 0).flatten(), :]

    nbrs_treatment = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(x_t)
    nbrs_control = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(x_c)

    dist_t, ind_t = nbrs_treatment.kneighbors(x)
    dist_c, ind_c = nbrs_control.kneighbors(x)
    
    
    y_t, y_c = np.zeros_like(y), np.zeros_like(y)
    for i in range(x.shape[0]):
       if t[i] == 1:
           y_t[i] = y[i]
           y_c[i] = y_a[(t_a == 0).flatten()][ind_c[i]]
       else:
           y_c[i] = y[i]
           y_t[i] = y_a[(t_a == 1).flatten()][ind_t[i]]
        
    pseudo_cate = y_t - y_c
    
    return pseudo_cate
    
class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=15, min_delta=0.001):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0 #Reset counter
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            #print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True


