# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 12:35:21 2017

@author: ctytl
"""

from pylab import *
import numpy as np
from scipy import linalg
from numpy.random import normal as randn  # Gaussian random variable

def col(x):
    # Receives an array and returns the array as an nx1 column vector
    return np.array([x]).T
    # this is to deal with numpy's lack of column vector support


# Setup simulation time array

dt = 0.1  #propagation sampling time (sec)
dtMeas = 5 #measurement sampling time (sec)
timeTotal = 200  #simulation total time (sec)
nTot = int(timeTotal/dt)  #total number of frames
mTot = int(timeTotal/dtMeas)-1  #total number of measurement frames
t = arange(0,timeTotal,dt)  #time array (sec)


# Setup system error characteristics

accelBias = 0.8  #m/s/s  accelerometer bias
sigmaRange = 1 #meters Range reading stdev
sigmaAccel = 0.5  #m/s/s    Accelerometer reading stdev



# Initialize time history arrays

posTrue = zeros(nTot) #Truth position
velTrue = zeros(nTot) #Truth velocity
uTrue = zeros(nTot)  #Truth u (acceleration input)
zTrue = zeros(mTot) #Truth range measurements
zMeas = zeros(mTot) #Measured range measurements (with error)
ns = 3  #Number of states
# States of x = [position]     in meters
#               [velocity]     in m/s
#               [accel bias]  acceleration bias to calibrate accelerometer
xHat = zeros([ns,nTot])  #KF estimated x


# Define propagation matrices

 # Note that Phi is in discrete time, and x(k+1) = Phi*x(k) + Bmat*u(k)
Phi = np.array([[1, dt,   0],  
                [0,  1, -dt],  
                [0,  0,   1]])  # Bias is constant

Bmat = np.array([[ 0],
                 [dt],  # times u (acceleration)
                 [ 0]])


# Setup truth trajectory

uTrue = 2*np.sin(0.3*t) + np.sin(0.5*t)  # Arbitrary acceleration command
uError = randn(0,sigmaAccel,size(t))
uMeas = uTrue + uError + accelBias
#calculate posTrue and velTrue in loop
#calculate zMeas in loop using posTrue


#Set initial truth and estimate conditions

posTrue[0] = -0.6   #position
velTrue[0] = 0     #velocity

xHat[0,0] = 0     #position
xHat[1,0] = 0     #velocity
xHat[2,0] = 0     #bias     (start with a guess of 0 bias)


#### Kalman Filter Setup #######

  #initialization error sigmas - the accuracy of your intial guesses
sigmaPos0 = 1.0    #m
sigmaVel0 = 0.2  #m/s
sigmaB0 = 2      #m/s/s

P = np.diag([sigmaPos0*sigmaPos0,sigmaVel0*sigmaVel0,sigmaB0*sigmaB0])
Q = np.diag([0,sigmaAccel*sigmaAccel,0])
R = np.diag([sigmaRange*sigmaRange])

# Store Kalman Filter history
P_3sigma = zeros([ns,size(t)])
P_3sigma[:,0] = [sigmaPos0,sigmaVel0,sigmaB0]
res = zeros(mTot)
S_3sigma = zeros(mTot)
khist = zeros([3,mTot])


### Simulation Time Step ###############

# k = propagation sample frames
j = 0  # = measurement sample frames

for k in range(1,size(t)):
    
    #Calculate truth trajectory
    velTrue[k] = velTrue[k-1] + dt*uTrue[k-1]
    posTrue[k] = posTrue[k-1] + dt*velTrue[k-1]

    
    #### Kalman Filter #####
    
    #Propagate states based on accelerometer measurement
    xHat_k = Phi.dot(col(xHat[:,k-1])) + Bmat.dot(uMeas[k-1])
    xHat[:,k] = xHat_k.flatten()
    
    #Propoagate covariance matrix
    P = Phi.dot(P).dot(Phi.T) + Q
    P_3sigma[:,k] = 3*sqrt(P.diagonal())  #Store 3*sigma values
    
    
    # Range measurement update
    if ( mod(t[k],dtMeas) == 0):
        
        #Calculate range measurement
        zTrue[j] = posTrue[k]
        zMeas[j] = posTrue[k] + randn(0,sigmaRange)
        
        #Calculate residual
        H = np.array([[1,0,0]])
        res[j] = zMeas[j] - H.dot(xHat[:,k])
        
        #Calculate Kalman gain
        S = H.dot(P).dot(H.T) + R  #covariance of the residual
        K = P.dot(H.T).dot(linalg.inv(S))
        khist[:,j] = K[:,0]
        S_3sigma[j] = 3*sqrt(S.diagonal())
        
        #Update state and covariance matrices
        xHat_k = xHat_k + K.dot(res[j])
        xHat[:,k] = xHat_k.flatten()
        
        P = (np.eye(ns) - K.dot(H)).dot(P)
        P_3sigma[:,k] = 3*sqrt(P.diagonal())  #Store 3*sigma values
        
        
        j = j + 1  #Have recorded j measurement updates so far
        
    
    
    
    
###### Plot Results  ####################

figure(1)
subplot(311)
title('State Estimation vs. Truth')
plot(t,posTrue,label='Truth')
plot(t,xHat[0,:],label='Estimate')
ylabel('Position [m]')
legend(loc='best')

subplot(312)
plot(t,velTrue,label='Truth')
plot(t,xHat[1,:],label='Estimate')
ylabel('Velocity [m]')
legend(loc='best')

subplot(313)
plot(t,accelBias*np.ones(size(t)),label='Truth')
plot(t,xHat[2,:],label='Estimate')
ylabel('Accel. Bias [m/s/s]')
legend(loc='best')
xlabel('Time [sec]')


figure(2)
subplot(211)
plot(t,uTrue,label='Truth')
plot(t,uMeas,'+',label='Measured')
title('True Acceleration vs. Measured')
ylabel('Acceleration [m/s/s]')
legend(loc='best')
xlabel('Time [sec]')

subplot(212)
plot(range(0,j),S_3sigma,'r--',label='3 Sigma')
plot(range(0,j),res,'b',label='Residual')
plot(range(0,j),-1*S_3sigma,'r--',label='3 Sigma')
title('Residual')
ylabel('Residual [m]')
legend(loc='best')
xlabel('Time [sec]')


figure(3)
subplot(311)
plot(t,P_3sigma[0,:],'r--',label='3 Sigma')
plot(t,xHat[0,:] - posTrue,'b',label='Error')
plot(t,-1*P_3sigma[0,:],'r--',label='3 Sigma')
title('State Estimate Errors')
ylabel('Position Error [m]')
legend(loc='best')

subplot(312)
plot(t,P_3sigma[1,:],'r--',label='3 Sigma')
plot(t,xHat[1,:] - velTrue,'b',label='Error')
plot(t,-1*P_3sigma[1,:],'r--',label='3 Sigma')
ylabel('Velocity Error [m]')
legend(loc='best')

subplot(313)
plot(t,P_3sigma[2,:],'r--',label='3 Sigma')
plot(t,xHat[2,:] - accelBias,'b',label='Error')
plot(t,-1*P_3sigma[2,:],'r--',label='3 Sigma')
ylabel('Accel. Bias Error [m]')
legend(loc='best')
xlabel('Time (sec)')

figure(4)
subplot(311)
plot(range(0,j),khist[0,:])
ylabel('K[0]')
subplot(312)
plot(range(0,j),khist[1,:])
ylabel('K[1]')
subplot(313)
plot(range(0,j),khist[2,:])
ylabel('K[2]')
xlabel('Meas Updates')

