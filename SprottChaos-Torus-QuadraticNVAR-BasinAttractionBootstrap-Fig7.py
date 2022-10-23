# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 13:17:10 2021

NVAR with time delays.  Don't be efficient for now.

Apply to the Li and Sprott 4D system that shows co-existing attractors

Li and Sprott, Coexisting Hidden Int. J. Bifurcation Chaos 24, 1450034 (2014)

Add prediction for other attractors

Jul 11, 2021, Fixed the constant term so it is +1, not +d

Feb. 7, 2022 - fixed saving file error and added calculation of similarity

May 2, 2022 - switch to normalized variables

July 23, 2022 - use bootstrap method by training on k=1 and use this
for finding the point needed for the k=2 model
THE CODE IS SPECIFIC FOR k=2 - do not change

Figure 7 of the paper

@author: Dan
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.colors import ListedColormap
from datetime import datetime

d = 4 # input_dimension = 3
k = 2 # number of time delay daps ###  DO NOT CHANGE
k1 = 1 # do not change
dlin = k*d  # size of linear part of outvector  
dlin1 = k1*d
dnonlin = int(dlin*(dlin+1)/2)  # size of nonlinear part of outvector
dnonlin1 = int(dlin1*(dlin1+1)/2)
dtot = 1 + dlin + dnonlin # size of total outvector - add one for the constant term
dtot1 = 1 + dlin1 + dnonlin1 

ridge_param = 4.e-5 # for the k=2 model
ridge_param1 = 4.e-5  # for the k=1 model - optimum found in other program

### do not change these parameters for the k=1 model     
dt=0.05
warmup = k*dt  # need to have warmup_pts >=1  
traintime = 300.
testtime=200.
maxtime = warmup+traintime+testtime

warmup_pts=round(warmup/dt)
traintime_pts=round(traintime/dt)
warmtrain_pts=warmup_pts+traintime_pts
testtime_pts=round(testtime/dt)
maxtime_pts=round(maxtime/dt)

beginplt = int(2*testtime_pts/3)

t_eval=np.linspace(0,maxtime,maxtime_pts+1) # need the +1 here to have a step of dt

a= 6.   # torus + two chaotic attractors
b = 0.1

def sprott(t, y):
  
  dy0 = y[1]-y[0]
  dy1 = -y[0]*y[2]+y[3]
  dy2 = y[0]*y[1]-a
  dy3 = -b*y[1]
  
  # since lorenz is 3-dimensional, dy/dt should be an array of 3 values
  return [dy0, dy1, dy2, dy3]

# initial condition for torus

sprott_soln = solve_ivp(sprott, (0, maxtime), [1.,-1.,1.,-1.] , t_eval=t_eval, method='RK45', rtol=1.e-6, atol=1.e-7)

# find standard deviation for normalization later

sprott_std = np.zeros(d)
for ii in range(d):
    sprott_std[ii] = np.std(sprott_soln.y[ii,:])
#    sprott_soln.y[ii,:] /= sprott_std[ii]   # July 24, 2022 - don't normalize!
    
"""
for the k=1 model
"""
x1 = np.zeros((dlin1,maxtime_pts))
    
for delay in range(k1):
    for j in range(delay,maxtime_pts):
        x1[d*delay:d*(delay+1),j]=sprott_soln.y[:,j-delay]   # don't subtract mean or normalize

out_train1 = np.ones((dtot1,traintime_pts))  # add constant term - do this by initializing entire matrix to 1's

out_train1[1:dlin1+1,:]=x1[:,warmup_pts-1:warmtrain_pts-1]  # don't overwrite first element

cnt=0
for row in range(dlin1):  
    for column in range(row,dlin1):
        # important - dlin here, not d (I was making this mistake previously)
        # add 1 to account for constant terms
        out_train1[dlin1+1+cnt]=x1[row,warmup_pts-1:warmtrain_pts-1]*x1[column,warmup_pts-1:warmtrain_pts-1]
        cnt += 1

W_out1 = np.zeros((d,dtot1))  # the +1 is for the constant terms

# drop the first few points when training
# x has the time delays too, so you need to take the first d components

# use when subtracting linear part of propagator 
W_out1 = (x1[0:d,warmup_pts:warmtrain_pts]-x1[0:d,warmup_pts-1:warmtrain_pts-1]) @ out_train1[:,:].T @ np.linalg.pinv(out_train1[:,:] @ out_train1[:,:].T + ridge_param1*np.identity(dtot1))

# use when not subtracting linear part of propagator
#W_out = x[0:d,warmup_pts:warmtrain_pts] @ out_train[:,:].T @ np.linalg.pinv(out_train[:,:] @ out_train[:,:].T + ridge_param*np.identity(dtot))


"""
for the k=2 model
"""

x = np.zeros((dlin,maxtime_pts))
    
for delay in range(k):
    for j in range(delay,maxtime_pts):
        x[d*delay:d*(delay+1),j]=sprott_soln.y[:,j-delay]   # don't subtract mean or normalize

out_train = np.ones((dtot,traintime_pts))  # add constant term - do this by initializing entire matrix to 1's

out_train[1:dlin+1,:]=x[:,warmup_pts-1:warmtrain_pts-1]  # don't overwrite first element

cnt=0
for row in range(dlin):  
    for column in range(row,dlin):
        # important - dlin here, not d (I was making this mistake previously)
        # add 1 to account for constant terms
        out_train[dlin+1+cnt]=x[row,warmup_pts-1:warmtrain_pts-1]*x[column,warmup_pts-1:warmtrain_pts-1]
        cnt += 1

W_out = np.zeros((d,dtot))  # the +1 is for the constant terms

# drop the first few points when training
# x has the time delays too, so you need to take the first d components

# use when subtracting linear part of propagator 
W_out = (x[0:d,warmup_pts:warmtrain_pts]-x[0:d,warmup_pts-1:warmtrain_pts-1]) @ out_train[:,:].T @ np.linalg.pinv(out_train[:,:] @ out_train[:,:].T + ridge_param*np.identity(dtot))

# use when not subtracting linear part of propagator
#W_out = x[0:d,warmup_pts:warmtrain_pts] @ out_train[:,:].T @ np.linalg.pinv(out_train[:,:] @ out_train[:,:].T + ridge_param*np.identity(dtot))


#  now see where the trajector heads

t_eval_po1=np.linspace(0,testtime,testtime_pts+1) # need the +1 here to have a step of dt
#
x0min = 1.
x0max = 3.
xdelta = .1/5. # 2.5/5. # .1/5.

z0min = 5.
z0max = 10.
zdelta = xdelta*(z0max-z0min)/(x0max-x0min)

x0 = np.arange(x0min, x0max+xdelta, xdelta)
z0 = np.arange(z0min, z0max+zdelta, zdelta)
X0, Z0 = np.meshgrid(x0, z0)

x0_n = np.arange(-x0max,-(x0min-xdelta), xdelta)
X0_n, Z0_n = np.meshgrid(x0_n, z0)

def basin(X0val,Z0val):
    # needed for ground-truth basin
    sprott_soln_po1 = solve_ivp(sprott, (0, testtime), [X0val,0.,Z0val,0.] , t_eval=t_eval_po1, method='RK45', rtol=1.e-6, atol=1.e-7)
    
    # don't forget to normalize these points too!
    #for ii in range(d):
    #    sprott_soln_po1.y[ii,:] /= sprott_std[ii]
    
    #print('X0val,Z0val ',X0val,Z0val)
    x0boot = np.array([X0val,0.,Z0val,0.]) # initial vector
    x0boot = x0boot.T
    
    x_predict1 = np.zeros((d,2))  # only works for k=2
    
    x_predict1[:,0] = x0boot
    
    out1 = np.ones(dtot1)  # add constant term - do this by initializing entire matrix to 1's

    out1[1:dlin1+1]=x0boot  # don't overwrite first element

    cnt=0
    for row in range(dlin1):  
        for column in range(row,dlin1):
            # important - dlin here, not d (I was making this mistake previously)
            # add 1 to account for constant terms
            out1[dlin1+1+cnt]=x0boot[row]*x0boot[column]
            cnt += 1

    x_predict1[:,1] = x0boot + W_out1 @ out1 # this is the bootstrap next step using the k=1 model
    
    #print('x_predict1 ',x_predict1)
    
    #print('x_predict1, truth, error: ',x_predict1[:,1],sprott_soln_po1.y[:,1],np.abs(x_predict1[:,1]-sprott_soln_po1.y[:,1]))
    #print('x_predict1 error: ',np.abs(x_predict1[:,1]-sprott_soln_po1.y[:,1]))
    
    x_test_po1 = np.zeros((dlin,testtime_pts))
    
    out_test_po1 = np.ones(dtot)  
    
    for delay in range(k):
        for j in range(delay,k):
            #x_test_po1[d*delay:d*(delay+1),0]=sprott_soln_po1.y[:,j-delay]   # don't subtract mean or normalize
            x_test_po1[d*delay:d*(delay+1),0]=x_predict1[:,j-delay]
        
    for j in range(testtime_pts-1):
        out_test_po1[1:dlin+1]=x_test_po1[:,j]   # the d:dlin+d is to account for constant layer
        # I am not being efficient here - just calculating the all over again - need to fix
        cnt=0
        for row in range(dlin):
            for column in range(row,dlin):
                out_test_po1[dlin+1+cnt]=x_test_po1[row,j]*x_test_po1[column,j] # the +1 is to account for constant layer
                cnt += 1
        # need to shift down values, then determine latest prediction
        x_test_po1[d:dlin,j+1] = x_test_po1[0:(dlin-d),j]        
        x_test_po1[0:d,j+1] = x_test_po1[0:d,j]+W_out @ out_test_po1[:]
    return sprott_soln_po1.y[3,-1], x_test_po1[3,-1]

v_basin = np.vectorize(basin)

basin_truth, basin_pred = v_basin(X0,Z0)

basin_truth_s = np.where(basin_truth<-2,0,np.where(basin_truth>2,1.,.5))
basin_pred_s = np.where(basin_pred<-2,0,np.where(basin_pred>2,1.,.5))

basin_truth_n, basin_pred_n = v_basin(X0_n,Z0_n)

basin_truth_s_n = np.where(basin_truth_n<-2,0,np.where(basin_truth_n>2,1.,.5))
basin_pred_s_n = np.where(basin_pred_n<-2,0,np.where(basin_pred_n>2,1.,.5))

time_now  = datetime.now().strftime('%m_%d_%Y_%H_%M_%S') 
np.save('X0_bootstrap_'+time_now,X0)
np.save('Z0_bootstrap_'+time_now,Z0)
np.save('basin_truth_bootstrap_'+time_now,basin_truth)
np.save('basin_pred_bootstrap_'+time_now,basin_pred)

np.save('X0_n_bootstrap_'+time_now,X0_n)
np.save('Z0__n_bootstrap_'+time_now,Z0_n)
np.save('basin_truth_n_bootstrap_'+time_now,basin_truth_n)
np.save('basin_pred_n_bootstrap_'+time_now,basin_pred_n)

# make plots

fig, ax = plt.subplots(2,2,figsize=(4.5,4.5))
ax[0,0].imshow(basin_truth_s, cmap=ListedColormap(['g','lightblue','r']),
               origin='lower', extent=[x0min, x0max, z0min, z0max], vmax=1, vmin=0)

ax[0,0].set_aspect(.4)
ax[0,0].title.set_text('true basin')
#ax[0,0].set_xlabel("x$_0$")
ax[0,0].set_ylabel("z$_0$")
#ax[0,0].xaxis.set_ticklabels([])

ax[0,1].imshow(basin_pred_s, cmap=ListedColormap(['g','lightblue','r']),
               origin='lower', extent=[x0min, x0max, z0min, z0max], vmax=1, vmin=0)
ax[0,1].set_aspect(.4)
ax[0,1].title.set_text('predicted basin')
ax[0,1].yaxis.set_ticklabels([])
#ax[0,1].xaxis.set_ticklabels([])
#ax[0,1].set_xlabel("x$_0$")

ax[1,0].imshow(basin_truth_s_n, cmap=ListedColormap(['g','lightblue','r']),
               origin='lower', extent=[-x0max, -x0min, z0min, z0max], vmax=1, vmin=0)

ax[1,0].set_aspect(.4)
#ax[1,0].title.set_text('true basin')
ax[1,0].set_xlabel("x$_0$")
ax[1,0].set_ylabel("z$_0$")

ax[1,1].imshow(basin_pred_s_n, cmap=ListedColormap(['g','lightblue','r']),
               origin='lower', extent=[-x0max, -x0min, z0min, z0max], vmax=1, vmin=0)
ax[1,1].set_aspect(.4)
#ax[1,1].title.set_text('predicted basin')
ax[1,1].yaxis.set_ticklabels([])
ax[1,1].set_xlabel("x$_0$")
plt.savefig('basin_bootstrap_'+time_now+'.svg',format='svg')
plt.savefig('basin_bootstrap_'+time_now+'.pdf',format='pdf', bbox_inches="tight")
plt.show()

cnt = 0
for kk in range(basin_truth_s.shape[0]):
    for ii in range(basin_truth_s.shape[1]):
        if (basin_truth_s[kk,ii] == basin_pred_s[kk,ii]):
            cnt +=1

print(' error for positive basin: '+str(cnt/(basin_truth_s.shape[0]*basin_truth_s.shape[1])))

cnt_n = 0
for kk in range(basin_truth_s_n.shape[0]):
    for ii in range(basin_truth_s_n.shape[1]):
        if (basin_truth_s_n[kk,ii] == basin_pred_s_n[kk,ii]):
            cnt_n +=1

print(' error for negative basin: '+str(cnt_n/(basin_truth_s_n.shape[0]*basin_truth_s_n.shape[1])))

