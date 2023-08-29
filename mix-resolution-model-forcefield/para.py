# %%
### date: 2023-02-17
### usage:1. the original sourece should be save in files/itp/ 
###       2. para.py -la la -lb lb -lc lc -ele ele
### result: the top.top, forcefield.itp, ff_hybrid_X.itp ffnonbonded_XX-XX-XX.itp are modified and moved into files. 
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
import argparse

# %%
###################### LOAD data
DATA={}
def loadtxt(file_path,col,q):
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
        if q in 'num':
            data[n]=data[n][:,col].astype(np.float64)
        else:
            data[n]=data[n][:,col]
        n=n+1
    return data

# %%
######################  Delete the old itp  ###################
cmd="rm -rf files/ff_hybrid* files/ffnonbonded_*.itp files/top.top /fiforcefield.itp}"; res=os.system(cmd)

# %%
####################### Defining flags and help messages ############
parser = argparse.ArgumentParser()
parser.add_argument("-la",  help="vdw la")
parser.add_argument("-lb",  help="vdw lb")
parser.add_argument("-lc",  help="vdw lc")
parser.add_argument("-ele", help="ele")
args = parser.parse_args()


la,lb,lc,ele=float(args.la),float(args.lb),float(args.lc),float(args.ele)
# la,lb,lc,ele=1,1,1,1.2
l_vacg=1
print(la,lb,lc,ele)

# %%


# %%
######################  ELE parameter  ###################
cmd="sed -n '1,8p' files/itp/ff_hybrid_original.itp > ff_hybrid_01.itp "; res=os.system(cmd)
cmd="sed -n '9,48p' files/itp/ff_hybrid_original.itp > ff_hybrid_temp.itp "; res=os.system(cmd)
cmd="sed -n '50,297p' files/itp/ff_hybrid_original.itp > ff_hybrid_03.itp "; res=os.system(cmd)


file_path="find . -name  ff_hybrid_temp.itp"

DATA[1]=loadtxt(file_path,6,'num')
size=len(DATA[1][0])
para_ele=np.round(DATA[1][0]*ele,3).reshape(size,1)

all=np.array([]).reshape(size,0)

###### the 0-5 col
for i in range(0,6):
    data=loadtxt(file_path,i,'str')[0].reshape(size,1)
    all=np.hstack((all,data))


###### the 6-7 col
all=np.hstack((all,para_ele,loadtxt(file_path,7,'num')[0].reshape(size,1)))    
np.savetxt('ff_hybrid_02.itp',all,fmt='%10s')


cmd = "cat ff_hybrid_01.itp ff_hybrid_02.itp ff_hybrid_03.itp > ff_hybrid_{}.itp".format(ele); res=os.system(cmd)
cmd = "sed -i 's/ZZZ/{}/g' ff_hybrid_{}.itp && rm -rf ff_hybrid_01.itp ff_hybrid_02.itp ff_hybrid_03.itp ff_hybrid_temp.itp".format(ele,ele); res=os.system(cmd)

# %%
######################  vdw parameter  ###################
cmd="sed -n '1,99p' files/itp/ffnonbonded_original.itp > ffnonbonded_01.itp "; res=os.system(cmd)
cmd="sed -n '100,111p' files/itp/ffnonbonded_original.itp > ffnonbonded_temp.itp "; res=os.system(cmd)
cmd="sed -n '113,3224p' files/itp/ffnonbonded_original.itp > ffnonbonded_03.itp "; res=os.system(cmd)


file_path="find . -name  ffnonbonded_temp.itp"

DATA[2]=loadtxt(file_path,3,'num'); DATA[3]=loadtxt(file_path,4,'num')

size=len(DATA[2][0])

para_vdw_A_C6=np.round(DATA[2][0][0:4]*la,5).reshape(int(size/3),1); para_vdw_A_C12=np.round(DATA[3][0][0:4]*la,7).reshape(int(size/3),1)
para_vdw_B_C6=np.round(DATA[2][0][4:8]*lb,5).reshape(int(size/3),1); para_vdw_B_C12=np.round(DATA[3][0][4:8]*lb,7).reshape(int(size/3),1)
para_vdw_C_C6=np.round(DATA[2][0][8:12]*lc,5).reshape(int(size/3),1); para_vdw_C_C12=np.round(DATA[3][0][8:12]*lc,7).reshape(int(size/3),1)

C6=np.vstack((para_vdw_A_C6,para_vdw_B_C6,para_vdw_C_C6))
C12=np.vstack((para_vdw_A_C12,para_vdw_B_C12,para_vdw_C_C12))
# print(C6,C12)


all=np.array([]).reshape(12,0)

###### the 0-2 col
for i in range(0,3):
    data=loadtxt(file_path,i,'str')[0].reshape(size,1)
    all=np.hstack((all,data))
all=np.hstack((all,C6,C12))
# print(all)

np.savetxt('ffnonbonded_02.itp',all,fmt='%10s')

cmd = "cat ffnonbonded_01.itp ffnonbonded_02.itp ffnonbonded_03.itp > ffnonbonded_{}-{}-{}.itp".format(la,lb,lc);res=os.system(cmd)
cmd = "sed -i 's/AAA/{}/g' ffnonbonded_{}-{}-{}.itp".format(la,la,lb,lc); res=os.system(cmd)
cmd = "sed -i 's/BBB/{}/g' ffnonbonded_{}-{}-{}.itp".format(lb,la,lb,lc); res=os.system(cmd)
cmd = "sed -i 's/CCC/{}/g' ffnonbonded_{}-{}-{}.itp".format(lc,la,lb,lc); res=os.system(cmd)

cmd = "rm -rf ffnonbonded_01.itp ffnonbonded_02.itp ffnonbonded_03.itp ffnonbonded_temp.itp"; res=os.system(cmd)

# %%
#### modify top.top, forcefield.itp
cmd = "cp files/itp/top_original.top top.top && sed -i 's/ZZZ/{}/g' top.top".format(ele); res=os.system(cmd)
cmd = "cp files/itp/forcefield_original.itp forcefield.itp && sed -i 's/AAA-BBB-CCC/{}-{}-{}/g' forcefield.itp".format(la,lb,lc); res=os.system(cmd)
cmd = "mv *.top *.itp files" ; res=os.system(cmd)

# %%


# %%
