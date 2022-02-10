import numpy as np
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from model.models import *
from sklearn.tree import DecisionTreeRegressor
from sklearn.kernel_ridge import KernelRidge

def AddFct(fct1, fct0, fct2, x_pred):
    return fct1.predict(x_pred).reshape(-1,1) - fct0.predict(x_pred).reshape(-1,1) + fct2.predict(x_pred)

def TwoStepRidge(t_unc, t_conf, x_unc, x_conf, y_unc, y_conf, x_pred, q = 0.5):
    #Estimate biased function on conf data
    reg_1 = linear_model.RidgeCV()#linear_model.Ridge(alpha=1)
    x=x_conf[(t_conf==1).flatten(),: ]
    y=y_conf[(t_conf==1).flatten(),: ]
    reg_1.fit(x, y)
    
    reg_0 = linear_model.RidgeCV()#linear_model.Ridge(alpha=1)
    x=x_conf[(t_conf==0).flatten(),: ]
    y=y_conf[(t_conf==0).flatten(),: ]
    reg_0.fit(x, y)
    
    #Make prediction on x_unc
    cate_biased = reg_1.predict(x_unc) - reg_0.predict(x_unc)
    
    w = t_unc/q - (1-t_unc)/(1-q)
    bias = w*y_unc - cate_biased
    
    reg_bias = linear_model.LinearRegression()#linear_model.Lasso(alpha=0.1)#
    reg_bias.fit(x_unc, bias)

    cate_biased = reg_1.predict(x_pred) - reg_0.predict(x_pred)
    bias = reg_bias.predict(x_pred).reshape(-1,1)

    return [cate_biased + bias, cate_biased, bias]#AddFct(reg_1, reg_0, reg_bias, x_pred)



def TwoStepRidge_modi(t_unc, t_conf, x_unc, x_conf, y_unc, y_conf, x_pred, q = 0.5):
    #Estimate biased function on conf data
    reg_1 = linear_model.RidgeCV()#linear_model.Ridge(alpha=1)
    x1=x_conf[(t_conf==1).flatten(),: ]
    y1=y_conf[(t_conf==1).flatten(),: ]
    reg_1.fit(x1, y1)
    
    reg_0 = linear_model.RidgeCV()#linear_model.Ridge(alpha=1)
    x0=x_conf[(t_conf==0).flatten(),: ]
    y0=y_conf[(t_conf==0).flatten(),: ]
    reg_0.fit(x0, y0)
    
    #Make prediction on x_unc
    x1=x_unc[(t_unc==1).flatten(),: ]
    y1=y_unc[(t_unc==1).flatten(),: ]

    x0=x_unc[(t_unc==0).flatten(),: ]
    y0=y_unc[(t_unc==0).flatten(),: ]
    
    bias1 = y1 - reg_1.predict(x1).reshape(-1,1)
    bias0 = y0 - reg_0.predict(x0).reshape(-1,1)
    
    reg_bias1 = linear_model.Lasso(alpha=0.1)#linear_model.LinearRegression()
    reg_bias0 = linear_model.Lasso(alpha=0.1)#linear_model.LinearRegression()
    
    reg_bias1.fit(x1, bias1)
    reg_bias0.fit(x0, bias0)

    cate_biased = reg_1.predict(x_pred) - reg_0.predict(x_pred)
    bias = reg_bias1.predict(x_pred).reshape(-1,1) - reg_bias0.predict(x_pred).reshape(-1,1)

    return [cate_biased + bias, cate_biased, bias]#AddFct(reg_1, reg_0, reg_bias, x_pred)



def Ridge(t_unc, x_unc, y_unc, x_pred):
    #Estimate biased function on conf data
    reg_1 = linear_model.Ridge(alpha=1)
    x=x_unc[(t_unc==1).flatten(),: ]
    y=y_unc[(t_unc==1).flatten(),: ]
    reg_1.fit(x, y)
    
    reg_0 = linear_model.Ridge(alpha=1)
    x=x_unc[(t_unc==0).flatten(),: ]
    y=y_unc[(t_unc==0).flatten(),: ]
    reg_0.fit(x, y)
    
    #Make prediction on x_unc
    cate = reg_1.predict(x_pred) - reg_0.predict(x_pred)
    
    return cate



def TwoStepRF(t_unc, t_conf, x_unc, x_conf, y_unc, y_conf, x_pred, q = 0.5):
    #Estimate biased function on conf data
    reg_1 = RandomForestRegressor(n_estimators=200, random_state=0)
    x=x_conf[(t_conf==1).flatten(),: ]
    y=y_conf[(t_conf==1).flatten(),: ].flatten()
    reg_1.fit(x, y)
    
    reg_0 = RandomForestRegressor(n_estimators=200, random_state=0)
    x=x_conf[(t_conf==0).flatten(),: ]
    y=y_conf[(t_conf==0).flatten(),: ].flatten()
    reg_0.fit(x, y)
    
    #Make prediction on x_unc
    cate_biased = (reg_1.predict(x_unc) - reg_0.predict(x_unc)).reshape(-1,1)

    w = t_unc/q - (1-t_unc)/(1-q)
    bias = w*y_unc - cate_biased
    
    reg_bias = linear_model.LinearRegression()#linear_model.Lasso(alpha=0.1)#
    reg_bias.fit(x_unc, bias)

    cate_biased = (reg_1.predict(x_pred) - reg_0.predict(x_pred)).reshape(-1,1)
    bias = reg_bias.predict(x_pred)

    return [cate_biased + bias, cate_biased, bias]

def TwoStepRF_modi(t_unc, t_conf, x_unc, x_conf, y_unc, y_conf, x_pred, q = 0.5):
    #Estimate biased function on conf data
    reg_1 = RandomForestRegressor(n_estimators=200, random_state=0)
    x=x_conf[(t_conf==1).flatten(),: ]
    y=y_conf[(t_conf==1).flatten(),: ].flatten()
    reg_1.fit(x, y)
    
    reg_0 = RandomForestRegressor(n_estimators=200, random_state=0)
    x=x_conf[(t_conf==0).flatten(),: ]
    y=y_conf[(t_conf==0).flatten(),: ].flatten()
    reg_0.fit(x, y)
    
    #Make prediction on x_unc
    x1=x_unc[(t_unc==1).flatten(),: ]
    y1=y_unc[(t_unc==1).flatten(),: ]

    x0=x_unc[(t_unc==0).flatten(),: ]
    y0=y_unc[(t_unc==0).flatten(),: ]
    
    bias1 = y1 - reg_1.predict(x1).reshape(-1,1)
    bias0 = y0 - reg_0.predict(x0).reshape(-1,1)
    
    reg_bias1 = linear_model.Lasso(alpha=0.1)#linear_model.LinearRegression()#
    reg_bias0 = linear_model.Lasso(alpha=0.1)#linear_model.LinearRegression()
    
    reg_bias1.fit(x1, bias1)
    reg_bias0.fit(x0, bias0)

    cate_biased = reg_1.predict(x_pred).reshape(-1,1) - reg_0.predict(x_pred).reshape(-1,1)
    bias = reg_bias1.predict(x_pred).reshape(-1,1) - reg_bias0.predict(x_pred).reshape(-1,1)

    return [cate_biased + bias, cate_biased, bias]



def TwoStepRF(t_unc, t_conf, x_unc, x_conf, y_unc, y_conf, x_pred, q = 0.5):
    #Estimate biased function on conf data
    reg_1 = RandomForestRegressor(n_estimators=200, random_state=0)
    x=x_conf[(t_conf==1).flatten(),: ]
    y=y_conf[(t_conf==1).flatten(),: ].flatten()
    reg_1.fit(x, y)
    
    reg_0 = RandomForestRegressor(n_estimators=200, random_state=0)
    x=x_conf[(t_conf==0).flatten(),: ]
    y=y_conf[(t_conf==0).flatten(),: ].flatten()
    reg_0.fit(x, y)
    
    #Make prediction on x_unc
    cate_biased = (reg_1.predict(x_unc) - reg_0.predict(x_unc)).reshape(-1,1)

    w = t_unc/q - (1-t_unc)/(1-q)
    bias = w*y_unc - cate_biased
    
    reg_bias = linear_model.LinearRegression()#linear_model.Lasso(alpha=0.1)#
    reg_bias.fit(x_unc, bias)

    cate_biased = (reg_1.predict(x_pred) - reg_0.predict(x_pred)).reshape(-1,1)
    bias = reg_bias.predict(x_pred)

    return [cate_biased + bias, cate_biased, bias]



def RF(t_unc, x_unc, y_unc, x_pred):
    #Estimate biased function on conf data
    reg_1 = RandomForestRegressor(n_estimators=2000, random_state=0)
    x=x_unc[(t_unc==1).flatten(),: ]
    y=y_unc[(t_unc==1).flatten(),: ].flatten()
    reg_1.fit(x, y)
    
    reg_0 = RandomForestRegressor(n_estimators=2000, random_state=0)
    x=x_unc[(t_unc==0).flatten(),: ]
    y=y_unc[(t_unc==0).flatten(),: ].flatten()
    reg_0.fit(x, y)
    
    #Make prediction on x_unc
    cate = (reg_1.predict(x_pred) - reg_0.predict(x_pred)).reshape(-1,1)

    return cate


def Tree(t_unc, x_unc, y_unc, x_pred):
    #Estimate biased function on conf data
    reg_1 = DecisionTreeRegressor(random_state=0)
    x=x_unc[(t_unc==1).flatten(),: ]
    y=y_unc[(t_unc==1).flatten(),: ].flatten()
    reg_1.fit(x, y)
    
    reg_0 = DecisionTreeRegressor(random_state=0)
    x=x_unc[(t_unc==0).flatten(),: ]
    y=y_unc[(t_unc==0).flatten(),: ].flatten()
    reg_0.fit(x, y)
    
    #Make prediction on x_unc
    cate = (reg_1.predict(x_pred) - reg_0.predict(x_pred)).reshape(-1,1)

    return cate


def Kernel(t_unc, x_unc, y_unc, x_pred):
    #Estimate biased function on conf data
    reg_1 = KernelRidge(alpha=1.0)
    x=x_unc[(t_unc==1).flatten(),: ]
    y=y_unc[(t_unc==1).flatten(),: ].flatten()
    reg_1.fit(x, y)
    
    reg_0 = KernelRidge(alpha=1.0)
    x=x_unc[(t_unc==0).flatten(),: ]
    y=y_unc[(t_unc==0).flatten(),: ].flatten()
    reg_0.fit(x, y)
    
    #Make prediction on x_unc
    cate = (reg_1.predict(x_pred) - reg_0.predict(x_pred)).reshape(-1,1)

    return cate



def TwoStepTarNet(net, t_unc, x_unc, y_unc, x_pred, q = 0.5):
    #Make prediction on x_unc
    cate_biased = net.predict_naive(torch.Tensor(x_unc)).detach().numpy()

    w = t_unc/q - (1-t_unc)/(1-q)
    bias = w*y_unc - cate_biased
    
    reg_bias = linear_model.LinearRegression()#linear_model.Lasso(alpha=0.1)
    reg_bias.fit(x_unc, bias)

    cate_biased = net.predict_naive(torch.Tensor(x_pred)).detach().numpy()
    bias = reg_bias.predict(x_pred)

    return cate_biased + bias

def TwoStepTarNet_modi(net, t_unc, x_unc, y_unc, x_pred, q = 0.5):
    #Make prediction on x_unc
    x1=x_unc[(t_unc==1).flatten(),: ]
    y1=y_unc[(t_unc==1).flatten(),: ]

    x0=x_unc[(t_unc==0).flatten(),: ]
    y0=y_unc[(t_unc==0).flatten(),: ]
    
    rep, o1, o = net(torch.Tensor(x1))
    rep, o, o0 = net(torch.Tensor(x0))
    
    bias1 = y1 - o1.detach().numpy()
    bias0 = y0 - o0.detach().numpy()
    
    reg_bias1 = linear_model.Lasso(alpha=0.1)#linear_model.LinearRegression()
    reg_bias0 = linear_model.Lasso(alpha=0.1)#linear_model.LinearRegression()
    
    reg_bias1.fit(x1, bias1)
    reg_bias0.fit(x0, bias0)

    rep, o1, o = net(torch.Tensor(x_pred))
    rep, o, o0 = net(torch.Tensor(x_pred))

    cate_biased = o1.detach().numpy() - o0.detach().numpy()
    bias = reg_bias1.predict(x_pred).reshape(-1,1) - reg_bias0.predict(x_pred).reshape(-1,1)

    return cate_biased + bias


