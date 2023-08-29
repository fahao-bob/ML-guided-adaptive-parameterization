# %%
import sklearn
import os
import heapq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint
################# cut data
from sklearn.model_selection import train_test_split
################# transfor data
from sklearn.preprocessing import MinMaxScaler,StandardScaler
################# model
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
################# cross-validation
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_validate
################# feature importance
from sklearn.inspection import permutation_importance
################# tuning hyp
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error,mean_absolute_error
from sklearn.metrics import make_scorer
################ clustering 
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import roc 
weight= roc.RankOrderCentroid()

# %%
cmd='ipynb-py-convert ML_model.ipynb ML_model.py'; res=os.system(cmd)

# %%
#### Default settings for plot
# Color
C = ['#9467bd', '#8c564b','#e377c2','#7f7f7f','k', '#17becf','#ff7f0e','#1f77b4','green','blue']

# Line
LS= ['solid','dashed','dashdot','dotted',(0, (1, 2)),(0, (3, 5, 1, 5)),(0, (3, 1, 1, 1)),(0, (3, 1, 1, 1, 1, 1))]
          
#      'dashdotdotted','densely dotted','dashdotted',
#      'densely dashdotted','dashdotdotted']

L_S = [
'solid',
('loosely dotted', (0, (1, 10))),
('dotted', (0, (1, 1))),
('densely dotted', (0, (1, 2))),
('loosely dashed', (0, (5, 10))),
('dashed', (0, (5, 5))),
('densely dashed', (0, (5, 1))),
('loosely dashdotted', (0, (3, 10, 1, 10))),
('dashdotted', (0, (3, 5, 1, 5))),
('densely dashdotted', (0, (3, 1, 1, 1))),
('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]

lw=1
alpha=1
alpha1=0.8

# Error
bp={'fmt':'o','capsize':6,'ms':10} # keywords of error bar
nw_bp={'fmt':'o','capsize':6,'ms':10,'capthick':3} # keywords of error bar
gebf_bp={'fmt':'s','capsize':6,'ms':10} # keywords of error bar
nw_gebf_bp={'fmt':'s','capsize':6,'ms':10,'capthick':3} # keywords of error bar

# font family
rc = {"font.family" : "serif",
      "mathtext.fontset" : "stix"}
plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]


# font size
font_la ={'weight':'normal','fontsize':35} # font of x/y lable
font_le ={'fontsize':30} # font of legend
font_tick =30

# %%
### LOAD data
DATA={}
def loadtxt(file_path):
    FILE={}
    input={}
    data={}
    FILE=os.popen("{}|sort".format(file_path))
    input=FILE.readlines()
    input=[x.strip('\n') for x in input]
#     print(input)
    n=0
    for i in input:
        data[n]=np.loadtxt(input[n],comments=['#','@'],dtype=str)
        data[n]=np.char.strip(data[n],',')
        data[n]=data[n].astype(np.float64)
        n=n+1
    return data

# %%
def MSE(a,b):
    return np.square(a-b).mean()

# %%
def MAE(a,b):
    return np.absolute(a-b).mean()

# %%
# get the model
def get_model(i):
    model = [RandomForestRegressor(random_state=1),DecisionTreeRegressor(random_state=1),LinearRegression(random_state=1), KNeighborsRegressor(random_state=1)]
    model = model[i]
    return model

# %%
# evaluate the model
def evaluate_model(x, y, i):
    results_tr = list()
    results_te = list()
    n_inputs, n_outputs = x.shape[1], y.shape[1]
    # define evaluation procedure
    cv=10 # 10-fold 
    n_scores= cross_validate(get_model(i),
                              x,y,
                              scoring = ['neg_mean_absolute_error','neg_mean_squared_error','r2'],
                              cv=cv,n_jobs=-1)
#     n_scores=np.absolute(n_scores)
    return n_scores

# %%
# plot feature importance
def plot_fi(model,x, y, title):
    result = permutation_importance(model, x, y, n_repeats=100, random_state=0,n_jobs=-1)
    df = pd.DataFrame({'feature_name': x.columns, 'feature_importance': result.importances_mean})
    plt.figure(figsize=(8, 2))
    sns.barplot(data=df, x='feature_importance', y='feature_name')
    plt.title(title)
    plt.show()

# %%
# plot r2
def plot_scatter(x, y,label):
    fig,ax= plt.subplots(figsize=(6,4))
    ax.scatter(x,y, s=30, c='k',label='{}'.format(label))
    legend=ax.legend(loc='upper right')

# %%
def MET (y_train,y_pred_train,n):
    r=np.array([])
    mse=np.array([])
    mae=np.array([])
    for i in range(0,y_train.shape[1]):
        r2=np.corrcoef(y_train[:,i],y_pred_train[:,i])[0,1]  ## the predicted y is model(x)
        v_mse=MSE(y_train[:,i],y_pred_train[:,i])
        v_mae=MAE(y_train[:,i],y_pred_train[:,i])
        r  =np.append(r,r2)
        mse=np.append(mse,v_mse)
        mae=np.append(mae,v_mae)
    print("r2  :",np.round(r,n))
    print("r2  :",np.round(r.mean(),n))
    print("mse :",np.round(mse,n))
    print("mse :",np.round(mse.mean(),n))
    print("mae :",np.round(mae,n))    
    print("mae :",np.round(mae.mean(),n))

# %%


# %%
def plot_gridsearch_para(model,hp, score):
    results = model.cv_results_
#     pprint(model.best_params_)

    plt.figure(figsize=(13, 13))
    plt.title("GridSearchCV evaluating using multiple scorers simultaneously", fontsize=25)

    plt.xlabel("%s" % hp)
    plt.ylabel("Score")

    ax = plt.gca()
    # ax.set_xlim(0, 402)
    # ax.set_ylim(0.73, 1)

    # Get the regular numpy array from the MaskedArray
    X_axis = np.array(results["param_%s" % hp].data, dtype=float)

    for scorer, color in zip(sorted(score), ["g", "k"]):
        for sample, style in (("train", "--"), ("test", "-")):
            sample_score_mean = results["mean_%s_%s" % (sample, scorer)]
            sample_score_std = results["std_%s_%s" % (sample, scorer)]
            ax.fill_between(
                X_axis,
                sample_score_mean - sample_score_std,
                sample_score_mean + sample_score_std,
                alpha=0.4 if sample == "test" else 0,
                color=color,
            )
            ax.plot(
                X_axis,
                sample_score_mean,
                style,
                color=color,
                alpha=1 if sample == "test" else 0.7,
                label="%s (%s)" % (scorer, sample),
            )
#             print(X_axis)

        best_index = np.nonzero(results["rank_test_%s" % scorer] == 1)[0][0]
        best_score = results["mean_test_%s" % scorer][best_index]

    #     Plot a dotted vertical line at the best score for that scorer marked by x
        ax.plot(
            [
                X_axis[best_index],
            ]
            * 2,
            [0, best_score],
            linestyle="-.",
            color=color,
            marker="x",
            markeredgewidth=3,
            ms=8,
        )

        # Annotate the best score for that scorer
        ax.annotate("%0.2f" % best_score, (X_axis[best_index], best_score + 0.005))

    plt.legend(loc="best")
    plt.grid(False)
    plt.show()

# %%
