# %%
import sklearn
import os
import heapq
import time
import numpy as np
import pandas as pd
from ML_model import loadtxt,MSE,MAE,evaluate_model,MET,plot_scatter,plot_gridsearch_para
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
from paretoset import paretoset

# %%
"""
## 1. import visited space
"""

# %%
################## load iter-0, iter-1,..iter-n
DATA={}
file_path='find . -name  "iter_{}.csv"'.format("*")   # 81 points
DATA=loadtxt(file_path)


# print(DATA.items())

it=13   # the will update it-th iterate|

####################
x_it={}
y_it={}
x=np.array([]).reshape(0,4)
y=np.array([]).reshape(0,7)
for i in range(0,it):
    x_it[i]=DATA[i][:,:4]
#     print(x_it[i].shape)
    y_it[i]=DATA[i][:,[5]+list(range(7,13))]
    x=np.vstack((x,x_it[i]))
    y=np.vstack((y,y_it[i]))

    
###############################  data cut
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1,train_size=0.8)

############################### normalized 
sc = MinMaxScaler()
x_std = sc.fit_transform(x)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)

print(x.shape)

# %%
sub_weight=np.array([0.143,0.143,0.143,0.143,0.143,0.143,0.143])

# %%
################## mean_squared_error
def custom_loss_mse_01(y_true,y_pred):
    scoring=mean_squared_error(y_true,y_pred,multioutput='uniform_average')
    return scoring

# %%
################# mean_absolute_error
def custom_loss_mae_02(y_true,y_pred):
    scoring=mean_absolute_error(y_true,y_pred,multioutput='uniform_average')
    return scoring

# %%
################## weighted mean_squared_error
def custom_loss_w_mse_03(y_true,y_pred,weight):
    scoring=mean_squared_error(y_true,y_pred,multioutput='raw_values')
    scoring=np.sum(scoring*weight)
    return scoring

# %%
################## weighted mean_absolute_error
def custom_loss_w_mae_04(y_true,y_pred,weight):
    scoring=mean_absolute_error(y_true,y_pred,multioutput='raw_values')
    scoring=np.sum(scoring*weight)
    return scoring

# %%
def custom_loss_r2_05(y_true,y_pred):
    scoring=r2_score(y_true,y_pred,multioutput='uniform_average')
    return scoring

# %%
############### custom weighted error
## calculate the total weighted loss function
def eva_01(y,sub_weight):            
    one=np.ones((y.shape[0],y.shape[1]))
    b=np.round(np.sum(np.abs(one-y)*sub_weight,axis=1),4)
    return b

# %%
# search the first n-th item having min sum of abs (1-S) 
def search_m_index(y,n,a):   
    if a=='min':
        ## return the smallest 1-ns y 
        re1=np.array(heapq.nsmallest(n,list(y)))
        ## return the mallest 1-ns y indice
        ind=y.argsort()[:n]
    else:
        ## return the largest 1-ns y 
        re1=np.array(heapq.nlargest(n,list(y)))
        ## return the largest 1-ns y indices
        ind=(-y).argsort()[:n]
    return re1,ind

# %%
##### iter_0
# n: the first n-th item having min sum of abs (1-S)
# ts: threshold ts for the min S
# search the first n-th item having min sum of abs (1-S) and each S larger than ts
def search_m_ts_index(y,n,ts):  
    ##### calculate the loss sum of (1-Si)
    before_score=eva_01(y,sub_weight)
    before_good_ind=search_m_index(before_score,n,'min')
    # print(before_good_ind[0],before_good_ind[1])

    score_old=before_good_ind[0]
    score_old_index=before_good_ind[1]

    score_new=np.array([])
    score_new_index=np.array([])
    ##### the min score Si must larger than ts (treshold)
    for i in score_old_index:
        a=np.where((y[i]>ts),True,False)
#         print(y[i])
#         print(a,i)
        if False not in a:
            score_new_index=np.append(score_new_index,i)
            score_new=np.append(score_new,before_score[i])
#     print(score_new_index)
#     print(score_new)
    return score_new,score_new_index

# %%
########################## Judge the model if improve
n=20
ts=0.4

best=np.array([]).reshape(0,12)
for i in range(0,it):
    print("Iter %s best_score:" %i,search_m_ts_index(y_it[i],n,ts)[0])
    print("Iter %s best_index:" %i,search_m_ts_index(y_it[i],n,ts)[1])
    a=int(search_m_ts_index(y_it[i],n,ts)[1][0])
    c=search_m_ts_index(y_it[i],n,ts)[0][0]
    print(c)
    b=np.hstack((x_it[i][a],y_it[i][a],c))
    best=np.vstack((best,b))
# print(best)

np.savetxt("best.csv",best,delimiter=',',fmt='%.03f')

# %%
from paretoset import paretoset
pareto = paretoset(y, sense=["max", "max","max","max","max","max","max"])
x_pare=x[pareto]; y_pare=y[pareto]

print("pareto best_score:",search_m_ts_index(y_pare,15,0.4)[0])
print("pareto best_index:",search_m_ts_index(y_pare,15,0.4)[1])


aa=search_m_ts_index(y_pare,15,0.4)[1].astype(int)
x_pare_final=x_pare[aa];  y_pare_final=y_pare[aa]

bb=search_m_ts_index(y_pare,15,0.4)[0]
bb=bb.reshape(11,1)


select=np.hstack((x_pare_final,y_pare_final,bb))
print(select[:,0:4])
np.savetxt("select.csv",select,delimiter=',',fmt='%.03f')

# %%
score={}
score['mse']=make_scorer(custom_loss_mse_01,greater_is_better=False)

# %%
############################### Randomsearch tuning hypermeters
# Create the random grid
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 3000, num = 20)]  # Number of trees in random forest
max_features = ['auto', 'sqrt']                                                  # Number of features to consider at every split
max_depth = [int(x) for x in np.linspace(10, 300, num = 20)]                      # Maximum number of levels in tree
max_depth.append(None)
min_samples_split = [2, 5, 10]                                                   # Minimum number of samples required to split a node
min_samples_leaf = [1, 2, 4]                                                     # Minimum number of samples required at each leaf node
bootstrap = [True]                                                               # Method of selecting samples for training each tree

random_grid_1 = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split':min_samples_split,
               'min_samples_leaf':min_samples_leaf,
               'bootstrap':bootstrap}

# ######## define estomator
model=RandomForestRegressor(random_state=1,criterion='mse')

########training for validation
model_thyp_1=RandomizedSearchCV(model, 
                        param_distributions=random_grid_1,
                        scoring=score,
                        n_iter=200,
                        cv=5,
                        verbose=1,
                        random_state=1,
                        n_jobs=-1,
                        refit='mse',    # setting refit='X' defiend in dict score, refits an estimator on the whole dataset
                                        # with the parameter setting has the best performancd on cross-validation 'X' score  
                        return_train_score=True) 

model_thyp_1.fit(x_train_std,y_train)
pprint(model_thyp_1.best_params_)

# %%
############################### Gridsearch tuning hypermeters
# Create the random grid
#### narrow the parameter range
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 10)] +[1473] # Number of trees in random forest
max_features = ['sqrt']                                                            # Number of features to consider at every split
max_depth = [int(x) for x in np.linspace(10, 210, num = 5)] +[132]                       # Maximum number of levels in tree
max_depth.append(None)
min_samples_split = [1.0,2,3]                                                      # Minimum number of samples required to split a node
min_samples_leaf = [1]                                                             # Minimum number of samples required at each leaf node
bootstrap = [True]                                                        # Method of selecting samples for training each tree

random_grid_2 = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split':min_samples_split,
               'min_samples_leaf':min_samples_leaf,
               'bootstrap':bootstrap}

# pprint(random_grid_2)



model_thyp_2=GridSearchCV(estimator=model,   
                        param_grid=random_grid_2,
                        scoring=score,
                        cv=5,
                        verbose=1,
                        n_jobs=-1,
                        refit='mse',    # setting refit='X' defiend in dict score, refits an estimator on the whole dataset
                                        # with the parameter setting has the best performancd on cross-validation 'X' score  
                        return_train_score=True)

model_thyp_2.fit(x_train_std,y_train)

# %%
#### The optimal parameter
pprint(model_thyp_2.best_params_)
para_final_std=model_thyp_2.best_params_


# best model 
model_final=model_thyp_2.best_estimator_

model_final.set_params(random_state=1)


# fit model
model_final.fit(x_train_std,y_train)


# predict
y_pred_train=model_final.predict(x_train_std)
y_pred_test=model_final.predict(x_test_std)



# title=['rg','hb','pi-pi','gra','grb','grc','gra_max','grb_max','grc_max']
title=['hb','gra','grb','grc','gra_max','grb_max','grc_max']
######### save test output
y_test_save=pd.DataFrame(y_test)
y_test_save.columns=title
y_test_save.to_csv('y_test.csv')

y_pred_test_save=pd.DataFrame(y_pred_test)
y_pred_test_save.columns=title
y_pred_test_save.to_csv('y_pred_test.csv')


######### save train output
y_train_save=pd.DataFrame(y_train)
y_train_save.columns=title
y_train_save.to_csv('y_train.csv')

y_pred_train_save=pd.DataFrame(y_pred_train)
y_pred_train_save.columns=title
y_pred_train_save.to_csv('y_pred_train.csv')



a=np.round(r2_score(y_test,y_pred_test,multioutput='raw_values'),3).reshape(1,len(title))
b=np.round(mean_squared_error(y_test,y_pred_test,multioutput='raw_values'),3).reshape(1,len(title))
c=np.round(mean_absolute_error(y_test,y_pred_test,multioutput='raw_values'),3).reshape(1,len(title))


test_r2_save=pd.DataFrame(a)
test_r2_save.columns=title
test_r2_save.to_csv('test_r2.csv')

test_mse_save=pd.DataFrame(b)
test_mse_save.columns=title
test_mse_save.to_csv('test_mse.csv')

test_mae_save=pd.DataFrame(c)
test_mae_save.columns=title
test_mae_save.to_csv('test_mae.csv')


print(a)
print(b)
print(c)

# %%
####################################### feature importance
fi=model_final.feature_importances_
print('feature importance:',fi)

# save
fi=pd.DataFrame(fi.reshape(1,len(fi)))
fi.columns=['la','lb','lc','ele']
fi.to_csv('feature_importance.csv')

# %%
# fit model to total data
model_final.fit(x_train_std,y_train)

# %%
###################################### Screen parameter iter [0]
la_ml,lb_ml,lc_ml,ele_ml={},{},{},{}
x_ml,x_ml_std,x_ml_std_good={},{},{}
y_pred={}


la_ml[it]=np.linspace(0.6,1.0,9)
# print(la_ml[it])
lb_ml[it]=np.linspace(0.6,1.0,9)
lc_ml[it]=np.linspace(0.6,1.0,9)
ele_ml[it]=np.linspace(1.0,1.6,13)
# ele_ml[it]=[1.0,1.4,1.6]
# print(ele_ml[it])


#################################### generate new parameters
idx=0
x_ml[it]=np.array([])
for i in la_ml[it]:
    for j in lb_ml[it]:
        for k in lc_ml[it]:
            for l in ele_ml[it]:
                a=np.array([i,j,k,l])
#                 print(a)
                x_ml[it]=np.append(x_ml[it],a)
                idx=idx+1

x_ml[it]=x_ml[it].reshape(idx,4)


# print(x_ml[it])

###################################### trans form x into according to max and min x (0~1)
x_ml_std[it]  = sc.fit_transform(x_ml[it])
# print(x_ml_std[it])

###################################### predict y with ML model
y_pred[it]=model_final.predict(x_ml_std[it])
print(y_pred[it].shape)

# %%
###################################### predict the weighted MAE score to evaluate 
score=eva_01(y_pred[it],sub_weight)
# # print(score.shape)


# ###################################### predict the weighted MAE score to evaluate 
n=int(y_pred[it].shape[0]*0.5)                         # half of the number of pareto optimal is selected
# print(n)
ts=0.4                                   # the 
a=search_m_ts_index(y_pred[it],n,ts)
# print(a[0],a[1])

######################################## the potential parameter sets
b=[int(x) for x in a[1]]
x_ml_std_sc1=x_ml_std[it][b]
print(x_ml_std_sc1.shape)

# %%
################ compare the mse with all the before simulated parameter sets
threshold=0.007
in_or_out=[]
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
for i in x_ml_std_sc1:
    dist=np.array([])
    for j in x_std:
#       print(k)

#       d=mean_squared_error(final_para_std,k)
        weight=np.array([0.25,0.25,0.25,0.57])
        d=custom_loss_w_mse_03(i,j,weight)
#         print(d)
        dist=np.append(dist,d)
    if np.min(dist) > threshold:
        in_or_out.append(True)
    else:
        in_or_out.append(False)

        # print(in_or_out)
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

x_ml_std_sc2=x_ml_std_sc1[in_or_out]

print(x_ml_std_sc2.shape)

# %%
############################# determine pareto optimal
from paretoset import paretoset
y_ml_std_sc2=model_final.predict(x_ml_std_sc2)
mask = paretoset(y_ml_std_sc2, sense=["max", "max","max","max","max","max","max"])
# print(mask)

# print(type(mask))
x_ml_std_sc3=x_ml_std_sc2[mask]
y_ml_std_sc3=y_ml_std_sc2[mask]
print(y_ml_std_sc3.shape)
# a=np.where(mask==True)[0]

# %%
##################################### determine the optimal K
sum_of_squared_distance=[]
k=range(2,60,5)
for i in k:
    kmeans = KMeans(n_clusters=i, random_state=10)
    kmeans.fit(x_ml_std_sc3)
    sum_of_squared_distance.append(kmeans.inertia_)
plt.plot(k,sum_of_squared_distance,'--')
plt.xlabel('k')
plt.ylabel('sum_of_squared_distance')

# %%
####################### K-means clustering
k_good=20
kmeans = KMeans(n_clusters=k_good, random_state=10)
kmeans.fit(x_ml_std_sc3)

# print(x_new)

# %%
######################################### parameter sets in each group 
print("The iter-th:",it)
group={}
result={}
score={}

final_para_std=np.array([]) # the selected para by k-means

para_final_std=np.array([]) # the selected para by k-means and min distance (threshold)
para_final_std_inver={}


# final_para=
for i in range(0,k_good):                 ## i is the i-th group, start from 0
    #### parameter sets in each group 
    loc=np.where(kmeans.labels_==i)[0]
    group[i]=x_ml_std_sc3[loc]
#     print(sc.inverse_transform(x_ml_std_sc3[loc]))
    result[i]=model_final.predict(group[i])
#     print(result[i].shape)
    score[i]=eva_01(result[i],sub_weight)
#     print(score[i])
    p_chosen_ind=search_m_index(score[i],1,'min')[1]
#     print(p_chosen_ind)
    final_para_std=group[i][p_chosen_ind][0]
    para_final_std=np.append(para_final_std,final_para_std)

        

para_final_std=para_final_std.reshape(int(len(para_final_std)/4),4)
# print(para_final_std.shape)

para_final=sc.inverse_transform(para_final_std)

# print(para_final)

# %%
################################ save the n-th generatedparameter 
temp1=para_final[:,:4]
# ############# the epsilon is converted to k (1~1.6)
# temp2=np.sqrt(2.5/para_final_std_inver[it][:,3]),3).reshape(temp1.shape[0],1)
# temp3=np.hstack((temp1,temp2))

print(temp1)

np.savetxt("para_final_iter{}.dat".format(it),temp1,delimiter='      ',fmt='%.03f')

# %%


# %%


# %%


# %%
