# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 13:17:10 2021

NVAR with time delays.  Don't be efficient for now.

Apply to the Li and Sprott 4D system that shows co-existing attractors

Li and Sprott, Coexisting Hidden Int. J. Bifurcation Chaos 24, 1450034 (2014)

Add prediction for other attractors

Jul 11, 2021, Fixed the constant term so it is +1, not +d

May 2, 2022: clean up calculation of nrmse and normalize each variable to its rms value

The code generates Figures 1-5 of the paper, plus some additional plots not included in the paper.
It also generates some error measurements used for hyperparameter optimization (just done by hand)

@author: Dan
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from datetime import datetime

time_now  = datetime.now().strftime('%m_%d_%Y_%H_%M_%S') 

d = 4 # input_dimension = 3
k = 2 # number of time delay taps
dlin = k*d  # size of linear part of outvector  
dnonlin = int(dlin*(dlin+1)/2)  # size of nonlinear part of outvector
dtot = 1 + dlin + dnonlin # size of total outvector - add one for the constant term5
ridge_param = 4.e-5 # 2

dt=0.05
warmup = k*dt  # need to have warmup_pts >=1
traintime = 300. # 400.
testtime=150.
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

sprott_soln = solve_ivp(sprott, (0, maxtime), [1.,-1.,1.,-1.] , t_eval=t_eval, method='RK45', rtol=1.e-7, atol=1.e-8)

sprott_std = np.zeros(d)
sprott_var = 0.
for ii in range(d):
    sprott_std[ii] = np.std(sprott_soln.y[ii,:])
    sprott_var += sprott_std[ii]**2
#    sprott_soln.y[ii,:] /= sprott_std[ii]


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

x_predict = np.zeros((d,traintime_pts))

# use when subtracting linear part of propagator
# shift from 0:d to d:2*d to avoid constant terms
x_predict = x[0:d,warmup_pts-1:warmtrain_pts-1] + W_out @ out_train[:,0:traintime_pts]

# use when non subtracting linear part of propagator
#x_predict = W_out @ out_train[:,0:traintime_pts]

fig1, axs1 = plt.subplots(4)
fig1.set_size_inches(5.,3.7)
plt.subplots_adjust(hspace=0.5)
#plt.suptitle('Li/Sprott, torus training phase')
axs1[0].plot(t_eval[warmup_pts:warmtrain_pts],x[0,warmup_pts:warmtrain_pts],label='truth')
axs1[0].plot(t_eval[warmup_pts:warmtrain_pts],x_predict[0,:],label='predict')
axs1[0].set_ylabel('x')
axs1[0].yaxis.set_label_coords(-.11, .5)
axs1[0].set_xlim(-2.,100.)
axs1[0].set_ylim(-5.1,5.1)
axs1[1].plot(t_eval[warmup_pts:warmtrain_pts],x[1,warmup_pts:warmtrain_pts],label='truth')
axs1[1].plot(t_eval[warmup_pts:warmtrain_pts],x_predict[1,:],label='predict')
axs1[1].set_ylabel('y')
axs1[1].yaxis.set_label_coords(-.11, .5)
axs1[1].set_xlim(-2.,100.)
axs1[1].set_ylim(-10.,10.)
axs1[2].plot(t_eval[warmup_pts:warmtrain_pts],x[2,warmup_pts:warmtrain_pts],label='truth')
axs1[2].plot(t_eval[warmup_pts:warmtrain_pts],x_predict[2,:],label='predict')
axs1[2].set_ylabel('z')
axs1[2].yaxis.set_label_coords(-.11, .5)
axs1[2].set_xlim(-2.,100.)
axs1[2].set_ylim(-10.,10.)
axs1[3].plot(t_eval[warmup_pts:warmtrain_pts],x[3,warmup_pts:warmtrain_pts],label='truth')
axs1[3].plot(t_eval[warmup_pts:warmtrain_pts],x_predict[3,:],label='predict')
axs1[3].set_ylabel('u')
axs1[3].yaxis.set_label_coords(-.11, .5)
axs1[3].set_xlabel('time')
axs1[3].set_xlim(-2.,100.)
axs1[3].set_ylim(-1.1,1.)
# Figure 1 of the paper
plt.savefig('torus_training_fine'+time_now+'.pdf',format='pdf', bbox_inches="tight")

fig1b, axs1b = plt.subplots(4)
fig1.set_size_inches(5.,3.7)
plt.subplots_adjust(hspace=0.5)
#plt.suptitle('Li/Sprott, torus training phase')
axs1b[0].plot(t_eval[warmup_pts:warmtrain_pts],x[0,warmup_pts:warmtrain_pts],label='truth')
axs1b[0].plot(t_eval[warmup_pts:warmtrain_pts],x_predict[0,:],label='predict')
axs1b[0].set_ylabel('x')
axs1b[0].yaxis.set_label_coords(-.09, .5)
axs1b[0].set_xlim(-5.,305.)
axs1b[0].set_ylim(-5.1,5.1)
axs1b[1].plot(t_eval[warmup_pts:warmtrain_pts],x[1,warmup_pts:warmtrain_pts],label='truth')
axs1b[1].plot(t_eval[warmup_pts:warmtrain_pts],x_predict[1,:],label='predict')
axs1b[1].set_ylabel('y')
axs1b[1].yaxis.set_label_coords(-.09, .5)
axs1b[1].set_xlim(-5.,305.)
axs1b[1].set_ylim(-10.,10.)
axs1b[2].plot(t_eval[warmup_pts:warmtrain_pts],x[2,warmup_pts:warmtrain_pts],label='truth')
axs1b[2].plot(t_eval[warmup_pts:warmtrain_pts],x_predict[2,:],label='predict')
axs1b[2].set_ylabel('z')
axs1b[2].yaxis.set_label_coords(-.09, .5)
axs1b[2].set_xlim(-5.,305.)
axs1b[2].set_ylim(-10.,10.)
axs1b[3].plot(t_eval[warmup_pts:warmtrain_pts],x[3,warmup_pts:warmtrain_pts],label='truth')
axs1b[3].plot(t_eval[warmup_pts:warmtrain_pts],x_predict[3,:],label='predict')
axs1b[3].set_ylabel('u')
axs1b[3].yaxis.set_label_coords(-.09, .5)
axs1b[3].set_xlabel('time')
axs1b[3].set_xlim(-5.,305.)
axs1b[3].set_ylim(-1.1,1.)
# Figure 2 of the paper
plt.savefig('torus_training_coarse'+time_now+'.pdf',format='pdf', bbox_inches="tight")

fig1a, axs1a = plt.subplots(4)
plt.suptitle(' torus training phase error')
plt.subplots_adjust(hspace=0.43)
axs1a[0].plot(t_eval[warmup_pts:warmtrain_pts],x[0,warmup_pts:warmtrain_pts]-x_predict[0,:],label='predict')
axs1a[0].set_ylabel('x')
axs1a[1].plot(t_eval[warmup_pts:warmtrain_pts],x[1,warmup_pts:warmtrain_pts]-x_predict[1,:],label='predict')
axs1a[1].set_ylabel('y')
axs1a[2].plot(t_eval[warmup_pts:warmtrain_pts],x[2,warmup_pts:warmtrain_pts]-x_predict[2,:],label='predict')
axs1a[2].set_ylabel('z')
axs1a[3].plot(t_eval[warmup_pts:warmtrain_pts],x[3,warmup_pts:warmtrain_pts]-x_predict[3,:],label='predict')
axs1a[3].set_ylabel('u')
axs1a[3].set_xlabel('time')

# has dlin components, need to seledt the first d
#nrms = np.sqrt(np.mean(np.square(x[0:d,warmup_pts:warmtrain_pts]-x_predict[:,:]))) #/np.mean(np.square(x[0:d,warmup_pts:warmtrain_pts])))
nrms = np.sqrt(np.mean(np.square(x[0:d,warmup_pts:warmtrain_pts]-x_predict[:,:]))/sprott_var)

print('training nrms: '+str(nrms))

out_test = np.ones(dtot)  

#out_test = out_train[:,traintime_pts-1]

# I have an issue in that I need data from the past, but using x_test as I have
# in other routines assumes I just have data from the current time
# I need x_test to have the same dimensions as x, which is dlin

x_test = np.zeros((dlin,testtime_pts))

x_test[:,0] = x[:,warmtrain_pts-1]  

for j in range(testtime_pts-1):
    out_test[1:dlin+1]=x_test[:,j]   # the 1:dlin+1 is to account for constant layer
    # I am not being efficient here - just calculating the all over again - need to fix
    cnt=0
    for row in range(dlin):
        for column in range(row,dlin):
            out_test[dlin+1+cnt]=x_test[row,j]*x_test[column,j] # the +1 is to account for constant layer
            cnt += 1
    # need to shift down values, then determine latest prediction
    x_test[d:dlin,j+1]=x_test[0:(dlin-d),j]        
    x_test[0:d,j+1] = x_test[0:d,j]+W_out @ out_test[:]

#nrms_test = np.sqrt(np.mean(np.square(x[0:d,warmtrain_pts-1:maxtime_pts-1]-x_test[0:d,:]))) #/np.mean(np.square(x[0:d,warmtrain_pts-1:maxtime_pts-1])))
nrms_test = np.sqrt(np.mean(np.square(x[0:d,warmtrain_pts-1:maxtime_pts-1]-x_test[0:d,:]))/sprott_var)

print('testing nrms: '+str(nrms_test))

fig2, axs2 = plt.subplots(4)
fig2.set_size_inches(5.,3.7)
plt.subplots_adjust(hspace=0.5)
#plt.suptitle('Li/Sprott, torus testing phase, k='+str(k)+' alpha='+str(ridge_param))
axs2[0].plot(t_eval[warmtrain_pts-1:maxtime_pts-1],x[0,warmtrain_pts-1:maxtime_pts-1],label='truth')
axs2[0].plot(t_eval[warmtrain_pts-1:maxtime_pts-1],x_test[0,:],label='predict')
axs2[0].set_ylabel('x')
axs2[0].set_ylim(-5.1,5.1)
axs2[0].yaxis.set_label_coords(-.11, .5)
axs2[0].set_xlim(298.,452.)
axs2[0].set_ylim(-5.1,5.1)
axs2[1].plot(t_eval[warmtrain_pts-1:maxtime_pts-1],x[1,warmtrain_pts-1:maxtime_pts-1],label='truth')
axs2[1].plot(t_eval[warmtrain_pts-1:maxtime_pts-1],x_test[1,:],label='predict')
axs2[1].set_ylabel('y')
axs2[1].set_ylim(-10.,10.)
axs2[1].yaxis.set_label_coords(-.11, .5)
axs2[1].set_xlim(298.,452.)
axs2[2].plot(t_eval[warmtrain_pts-1:maxtime_pts-1],x[2,warmtrain_pts-1:maxtime_pts-1],label='truth')
axs2[2].plot(t_eval[warmtrain_pts-1:maxtime_pts-1],x_test[2,:],label='predict')
axs2[2].set_ylabel('z')
axs2[2].set_ylim(-10.,10.)
axs2[2].yaxis.set_label_coords(-.11, .5)
axs2[2].set_xlim(298.,452.)
axs2[3].plot(t_eval[warmtrain_pts-1:maxtime_pts-1],x[3,warmtrain_pts-1:maxtime_pts-1],label='truth')
axs2[3].plot(t_eval[warmtrain_pts-1:maxtime_pts-1],x_test[3,:],label='predict')
axs2[3].set_ylabel('u')
axs2[3].yaxis.set_label_coords(-.11, .5)
axs2[3].set_xlim(298.,452.)
axs2[3].set_ylim(-1.1,1.)
axs2[3].set_xlabel('time')
plt.savefig('torus_testing'+time_now+'.pdf',format='pdf', bbox_inches="tight")

fig2a, axs2a = plt.subplots(2,2)
fig2a.set_figheight(5) 
plt.subplots_adjust(hspace=0.3, wspace=0.3)
plt.suptitle('Li/Sprott, torus testing phase, k='+str(k)+' alpha='+str(ridge_param))
axs2a[0,0].plot(x_test[1,beginplt:],x_test[0,beginplt:])
axs2a[0,0].set_ylabel('x')
axs2a[0,0].set_xlabel('y')

axs2a[0,1].plot(x_test[1,beginplt:],x_test[2,beginplt:])
axs2a[0,1].set_ylabel('z')
axs2a[0,1].set_xlabel('y')

axs2a[1,0].plot(x_test[1,beginplt:],x_test[3,beginplt:])
axs2a[1,0].set_ylabel('u')
axs2a[1,0].set_xlabel('y')

axs2a[1,1].plot(x_test[0,beginplt:],x_test[3,beginplt:])
axs2a[1,1].set_ylabel('u')
axs2a[1,1].set_xlabel('x')


#  now try to go to one of the chaotic attractors

# find the warmup points using truth just finding k points
t_eval_po1=np.linspace(0,dt*(k-1),k) # need the +1 here to have a step of dt

#integrate over full range so we have the data to plot
sprott_soln_po1 = solve_ivp(sprott, (0, maxtime), [0.,4.,0.,-5.] , t_eval=t_eval, method='RK45', rtol=1.e-6, atol=1.e-7)

# don't forget to normalize these points too!
#for ii in range(d):
#    sprott_soln_po1.y[ii,:] /= sprott_std[ii]

x_test_po1 = np.zeros((dlin,testtime_pts))

out_test_po1 = np.ones(dtot)  

for delay in range(k):
    for j in range(delay,k):
        x_test_po1[d*delay:d*(delay+1),0]=sprott_soln_po1.y[:,j-delay]   # don't subtract mean or normalize
    
    
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

fig4, axs4 = plt.subplots(4)
fig4.set_size_inches(5.,3.7)
plt.subplots_adjust(hspace=0.5)
#plt.subplots_adjust(hspace=0.43)
#plt.suptitle('Li/Sprott, chaos, testing phase, k='+str(k)+' alpha='+str(ridge_param))
axs4[0].plot(t_eval[warmtrain_pts-1:maxtime_pts-1],sprott_soln_po1.y[0,1:testtime_pts+1],label='truth')
axs4[0].plot(t_eval[warmtrain_pts-1:maxtime_pts-1],x_test_po1[0,:],label='predict')
axs4[0].set_ylabel('x')
axs4[0].set_ylim(-8.,8.)
axs4[0].yaxis.set_label_coords(-.09, .5)
axs4[0].set_xlim(298.,452.)

axs4[1].plot(t_eval[warmtrain_pts-1:maxtime_pts-1],sprott_soln_po1.y[1,1:testtime_pts+1],label='truth')
axs4[1].plot(t_eval[warmtrain_pts-1:maxtime_pts-1],x_test_po1[1,:],label='predict')
axs4[1].set_ylabel('y')
axs4[1].set_ylim(-20.,20.)
axs4[1].yaxis.set_label_coords(-.09, .5)
axs4[1].set_xlim(298.,452.)

axs4[2].plot(t_eval[warmtrain_pts-1:maxtime_pts-1],sprott_soln_po1.y[2,1:testtime_pts+1],label='truth')
axs4[2].plot(t_eval[warmtrain_pts-1:maxtime_pts-1],x_test_po1[2,:],label='predict')
axs4[2].set_ylabel('z')
axs4[2].set_ylim(-20.,20.)
axs4[2].yaxis.set_label_coords(-.09, .5)
axs4[2].set_xlim(298.,452.)

axs4[3].plot(t_eval[warmtrain_pts-1:maxtime_pts-1],sprott_soln_po1.y[3,1:testtime_pts+1],label='truth')
axs4[3].plot(t_eval[warmtrain_pts-1:maxtime_pts-1],x_test_po1[3,:],label='predict')
axs4[3].set_ylabel('u')
axs4[3].set_xlim(298.,452.)
axs4[3].set_ylim(-7.,-3.)
axs4[3].yaxis.set_label_coords(-.09, .5)
axs4[3].set_xlabel('time')
# Figure 4 of the paper
plt.savefig('chaos1_testing'+time_now+'.pdf',format='pdf', bbox_inches="tight")

fig4a, axs4a = plt.subplots(2,2)
fig4a.set_figheight(5) 
plt.subplots_adjust(hspace=0.3, wspace=0.3)
beginplt = int(testtime_pts/2)
#plt.suptitle('Li/Sprott, chaos, testing phase, k='+str(k)+' alpha='+str(ridge_param))
axs4a[0,0].plot(x_test_po1[1,beginplt:],x_test_po1[0,beginplt:],color='orange')
axs4a[0,0].set_ylabel('x')
axs4a[0,0].set_xlabel('y')

axs4a[0,1].plot(x_test_po1[1,beginplt:],x_test_po1[2,beginplt:],color='orange')
axs4a[0,1].set_ylabel('z')
axs4a[0,1].set_xlabel('y')

axs4a[1,0].plot(x_test_po1[1,beginplt:],x_test_po1[3,beginplt:],color='orange')
axs4a[1,0].set_ylabel('u')
axs4a[1,0].set_xlabel('y')

axs4a[1,1].plot(x_test_po1[0,beginplt:],x_test_po1[3,beginplt:],color='orange')
axs4a[1,1].set_ylabel('u')
axs4a[1,1].set_xlabel('x')
plt.savefig('chaosn_testing_attractor'+time_now+'.pdf',format='pdf', bbox_inches="tight")

#ground truth chaotic attractor
plotlen = 1500
fig4b, axs4b = plt.subplots(2,2)
fig4b.set_figheight(5) 
plt.subplots_adjust(hspace=0.3, wspace=0.3)
beginplt = int(testtime_pts/2)
#plt.suptitle('Li/Sprott, chaos, testing phase, k='+str(k)+' alpha='+str(ridge_param))
axs4b[0,0].plot(sprott_soln_po1.y[1,beginplt:beginplt+plotlen],sprott_soln_po1.y[0,beginplt:beginplt+plotlen])
axs4b[0,0].set_ylabel('x')
axs4b[0,0].set_xlabel('y')

axs4b[0,1].plot(sprott_soln_po1.y[1,beginplt:beginplt+plotlen],sprott_soln_po1.y[2,beginplt:beginplt+plotlen])
axs4b[0,1].set_ylabel('z')
axs4b[0,1].set_xlabel('y')

axs4b[1,0].plot(sprott_soln_po1.y[1,beginplt:beginplt+plotlen],sprott_soln_po1.y[3,beginplt:beginplt+plotlen])
axs4b[1,0].set_ylabel('u')
axs4b[1,0].set_xlabel('y')

axs4b[1,1].plot(sprott_soln_po1.y[0,beginplt:beginplt+plotlen],sprott_soln_po1.y[3,beginplt:beginplt+plotlen])
axs4b[1,1].set_ylabel('u')
axs4b[1,1].set_xlabel('x')
plt.savefig('chaosn_truth_attractor'+time_now+'.pdf',format='pdf', bbox_inches="tight")

# now find the other chaotic attractor

t_eval_po2=np.linspace(0,dt*(k-1),k) # need the +1 here to have a step of dt

sprott_soln_po2 = solve_ivp(sprott, (0, maxtime), [0.,-4.,0.,5.] , t_eval=t_eval_po1, method='RK45', rtol=1.e-6, atol=1.e-7)

# don't forget to normalize these points too!
#for ii in range(d):
#    sprott_soln_po2.y[ii,:] /= sprott_std[ii]
    
x_test_po2 = np.zeros((dlin,testtime_pts))

out_test_po2 = np.ones(dtot)  

for delay in range(k):
    for j in range(delay,k):
        x_test_po2[d*delay:d*(delay+1),0]=sprott_soln_po2.y[:,j-delay]   # don't subtract mean or normalize
    
    
for j in range(testtime_pts-1):
    out_test_po2[1:dlin+1]=x_test_po2[:,j]   # the d:dlin+d is to account for constant layer
    # I am not being efficient here - just calculating the all over again - need to fix
    cnt=0
    for row in range(dlin):
        for column in range(row,dlin):
            out_test_po2[dlin+1+cnt]=x_test_po2[row,j]*x_test_po2[column,j] # the +d is to account for constant layer
            cnt += 1
    # need to shift down values, then determine latest prediction
    x_test_po2[d:dlin,j+1] = x_test_po2[0:(dlin-d),j]        
    x_test_po2[0:d,j+1] = x_test_po2[0:d,j]+W_out @ out_test_po2[:]

fig5, axs5 = plt.subplots(4)
fig5.set_size_inches(5.,3.7)
plt.subplots_adjust(hspace=0.5)
#plt.subplots_adjust(hspace=0.43) #, wspace=0.3)
#plt.suptitle('Li/Sprott, chaotic orbit, testing phase, k='+str(k)+' alpha='+str(ridge_param))
axs5[0].plot(t_eval[warmtrain_pts-1:maxtime_pts-1],x_test_po1[0,:],color='red')
axs5[0].plot(t_eval[warmtrain_pts-1:maxtime_pts-1],x_test_po2[0,:],color='green')
axs5[0].set_ylabel('x')
axs5[0].set_ylim(-8.,8.)
axs5[0].yaxis.set_label_coords(-.09, .5)
axs5[0].set_xlim(298.,452.)

axs5[1].plot(t_eval[warmtrain_pts-1:maxtime_pts-1],x_test_po1[1,:],color='red')
axs5[1].plot(t_eval[warmtrain_pts-1:maxtime_pts-1],x_test_po2[1,:],color='green')
axs5[1].set_ylabel('y')
axs5[1].set_ylim(-20.,20.)
axs5[1].yaxis.set_label_coords(-.09, .5)
axs5[1].set_xlim(298.,452.)

axs5[2].plot(t_eval[warmtrain_pts-1:maxtime_pts-1],x_test_po1[2,:],color='red')
axs5[2].plot(t_eval[warmtrain_pts-1:maxtime_pts-1],x_test_po2[2,:],color='green')
axs5[2].set_ylabel('z')
axs5[2].set_ylim(-20.,20.)
axs5[2].yaxis.set_label_coords(-.09, .5)
axs5[2].set_xlim(298.,452.)

axs5[3].plot(t_eval[warmtrain_pts-1:maxtime_pts-1],x_test_po1[3,:],color='red')
axs5[3].plot(t_eval[warmtrain_pts-1:maxtime_pts-1],x_test_po2[3,:],color='green')
axs5[3].set_ylabel('u')
axs5[3].set_xlabel('time')
axs5[3].set_ylim(-8.5,8.5)
axs5[3].yaxis.set_label_coords(-.09, .5)
axs5[3].set_xlim(298.,452.)
# Figure 5 of the paper
plt.savefig('chaosp_testing'+time_now+'.pdf',format='pdf', bbox_inches="tight")


fig5a, axs5a = plt.subplots(2,2)
fig5a.set_figheight(5) 
plt.subplots_adjust(hspace=0.3, wspace=0.3)
#plt.suptitle('Li/Sprott, SYM chaotic orbit, testing phase, k='+str(k)+' alpha='+str(ridge_param))
axs5a[0,0].plot(x_test_po2[1,beginplt:],x_test_po2[0,beginplt:],color='orange')
axs5a[0,0].set_ylabel('x')
axs5a[0,0].set_xlabel('y')

axs5a[0,1].plot(x_test_po2[1,beginplt:],x_test_po2[2,beginplt:],color='orange')
axs5a[0,1].set_ylabel('z')
axs5a[0,1].set_xlabel('y')

axs5a[1,0].plot(x_test_po2[1,beginplt:],x_test_po2[3,beginplt:],color='orange')
axs5a[1,0].set_ylabel('u')
axs5a[1,0].set_xlabel('y')

axs5a[1,1].plot(x_test_po2[0,beginplt:],x_test_po2[3,beginplt:],color='orange')
axs5a[1,1].set_ylabel('u')
axs5a[1,1].set_xlabel('x')
plt.savefig('chaosp_testing_attractor'+time_now+'.pdf',format='pdf', bbox_inches="tight")