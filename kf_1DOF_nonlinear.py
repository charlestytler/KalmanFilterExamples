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


def f(x,u):
    xNew = zeros(4)
    xNew[0] = x[1]             #Position
    xNew[1] = x[3]*(u-x[2])    #Velocity  = scale*(u - bias)
    xNew[2] = 0                #Accel Bias
    xNew[3] = 0                #Accel Scaling
    return xNew

# 4th Order Runge Kutta Calculation
def RK4(x,u,dt):
    K1 = f(x,u)
    K2 = f(x + K1*dt/2,u)
    K3 = f(x + K2*dt/2,u)
    K4 = f(x + K3*dt,u)
    xest = x + 1/6*(K1 + 2*K2 + 2*K3 + K4)*dt
    return xest



# Setup simulation time array

dt = 0.1  #propagation sampling time (sec)
dtMeas = 5  #measurement sampling time (sec)
timeTotal = 300  #simulation total time (sec)
nTot = int(timeTotal/dt)  #total number of frames
mTot = int(timeTotal/dtMeas)-1  #total number of measurement frames
t = arange(0,timeTotal,dt)  #time array (sec)


# Setup system error characteristics

accelBias = 0.3  #m/s/s  accelerometer bias
accelScale = 1.4  #non-dimensional accelerometer scaling
sigmaRange = 0.1 #meters Range reading stdev
sigmaAccel = 1  #m/s/s    Accelerometer reading stdev



# Initialize time history arrays

posTrue = zeros(nTot) #Truth position
velTrue = zeros(nTot) #Truth velocity
uTrue = zeros(nTot)  #Truth u (acceleration input)
zTrue = zeros(mTot) #Truth range measurements
zMeas = zeros(mTot) #Measured range measurements (with error)
ns = 4  #Number of states
# States of x = [position]     in meters
#               [velocity]     in m/s
#               [accel bias]  acceleration bias to calibrate accelerometer
#               [accel scale] accelerometer scaling
xHat = zeros([ns,nTot])  #KF estimated x


# Define propagation matrices

# # Note that Phi is in discrete time, and x(k+1) = Phi*x(k) + Bmat*u(k)
#Phi = np.array([[1, dt,   0],  
#                [0,  1, -dt],  
#                [0,  0,   1]])  # Bias is constant
#
#Bmat = np.array([[ 0],
#                 [dt],  # times u (acceleration)
#                 [ 0]])


# Setup truth trajectory

uTrue = 0.8*np.sin(0.03*t) + 0.4*np.sin(0.05*t)  # Arbitrary acceleration command
uError = randn(0,sigmaAccel,size(t))
uMeas = accelScale*(uTrue + uError) + accelBias
#calculate posTrue and velTrue in loop
#calculate zMeas in loop using posTrue


#Set initial truth and estimate conditions

posTrue[0] = 0   #position
velTrue[0] = 0     #velocity

xHat[0,0] = 0     #position
xHat[1,0] = 0     #velocity
xHat[2,0] = 0     #bias     (start with a guess of 0 bias)
xHat[3,0] = 1     #scaling


#### Kalman Filter Setup #######

  #initialization error sigmas - the accuracy of your intial guesses
sigmaPos0 = 1.0    #m
sigmaVel0 = 0.2  #m/s
sigmaB0 = 0.5      #m/s/s
sigmaS0 = 2      #non-dimensional

P = np.diag([sigmaPos0*sigmaPos0,sigmaVel0*sigmaVel0,sigmaB0*sigmaB0,sigmaS0*sigmaS0])
Q = np.diag([0,sigmaAccel*sigmaAccel,0,0])
R = np.diag([sigmaRange*sigmaRange])

# Store Kalman Filter history
P_3sigma = zeros([ns,size(t)])
P_3sigma[:,0] = [sigmaPos0,sigmaVel0,sigmaB0,sigmaS0]
res = zeros(mTot)
S_3sigma = zeros(mTot)


### Simulation Time Step ###############

# k = propagation sample frames
j = 0  # = measurement sample frames

for k in range(1,size(t)):
    
    #Calculate truth trajectory
    velTrue[k] = velTrue[k-1] + dt*uTrue[k-1]
    posTrue[k] = posTrue[k-1] + dt*velTrue[k-1]
    
    # Define propagation matrix       # Note that Phi is in discrete time
    b_e = xHat[2,k-1]
    s_e = xHat[3,k-1]
    u_e = uMeas[k-1]
    Phi = np.array([[1, dt, -s_e/2*dt, (u_e)/2*dt],  
                     [0,  1,   -s_e*dt, (u_e-b_e)*dt],  
                     [0,  0,         1, 0],  # Scale and Bias are constants
                     [0,  0,         0, 1]])  
    
#    Bmat = np.array([[ 0],#s_e/2*dt],
#                     [ 0],#  s_e*dt],  # times u (acceleration)
#                     [        0],
#                     [        0]])

    
    #### Kalman Filter #####
    
    #Propagate states based on accelerometer measurement
#    xHat_k = Phi.dot(col(xHat[:,k-1])) + Bmat.dot(uMeas[k-1])
#    xHat[:,k] = xHat_k.flatten()

#    xHat[0,k] = xHat[0,k-1] + dt*xHat[1,k-1] + xHat[3,k-1]/2*dt*(uMeas[k-1]-xHat[2,k-1])
#    xHat[1,k] = xHat[1,k-1] + dt*xHat[3,k-1]*(uMeas[k-1]-xHat[2,k-1])
#    xHat[2,k] = xHat[2,k-1]
#    xHat[3,k] = xHat[3,k-1]
    xHat[:,k] = RK4(xHat[:,k-1],uMeas[k-1],dt)
    
        
    #Propoagate covariance matrix
    P = Phi.dot(P).dot(Phi.T) + Q
    P_3sigma[:,k] = 3*sqrt(P.diagonal())  #Store 3*sigma values
    
    
    # Range measurement update
    if ( mod(t[k],dtMeas) == 0):
        
        #Calculate range measurement
        zTrue[j] = posTrue[k]
        zMeas[j] = posTrue[k] + randn(0,sigmaRange)
        
        #Calculate residual
        H = np.array([[1,0,0,0]])
        res[j] = zMeas[j] - H.dot(xHat[:,k])
        
        #Calculate Kalman gain
        S = H.dot(P).dot(H.T) + R  #covariance of the residual
        K = P.dot(H.T).dot(linalg.inv(S))
        S_3sigma[j] = 3*sqrt(S.diagonal())
        
        #Update state and covariance matrices
        xHat[:,k] = xHat[:,k] + K.dot(res[j]).flatten()
        #xHat[:,k] = xHat_k.flatten()
        
        
        P = (np.eye(ns) - K.dot(H)).dot(P)
        P_3sigma[:,k] = 3*sqrt(P.diagonal())  #Store 3*sigma values
        
        
        j = j + 1  #Have recorded j measurement updates so far
        
    
    
    
    
###### Plot Results  ####################

figure(1)
subplot(411)
title('State Estimation vs. Truth')
plot(t,posTrue,label='Truth')
plot(t,xHat[0,:],label='Estimate')
ylabel('Position [m]')
legend(loc='best')

subplot(412)
plot(t,velTrue,label='Truth')
plot(t,xHat[1,:],label='Estimate')
ylabel('Velocity [m/s]')
legend(loc='best')

subplot(413)
plot(t,accelBias*np.ones(size(t)),label='Truth')
plot(t,xHat[2,:],label='Estimate')
ylabel('Accel. Bias [m/s/s]')
legend(loc='best')
xlabel('Time [sec]')

subplot(414)
plot(t,1/accelScale*np.ones(size(t)),label='Truth')
plot(t,xHat[3,:],label='Estimate')
ylabel('Accel. Scaling [ND]')
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
subplot(411)
plot(t,P_3sigma[0,:],'r--',label='3 Sigma')
plot(t,xHat[0,:] - posTrue,'b',label='Error')
plot(t,-1*P_3sigma[0,:],'r--',label='3 Sigma')
title('State Estimate Errors')
ylabel('Position Error [m]')
legend(loc='best')

subplot(412)
plot(t,P_3sigma[1,:],'r--',label='3 Sigma')
plot(t,xHat[1,:] - velTrue,'b',label='Error')
plot(t,-1*P_3sigma[1,:],'r--',label='3 Sigma')
ylabel('Velocity Error [m]')
legend(loc='best')

subplot(413)
plot(t,P_3sigma[2,:],'r--',label='3 Sigma')
plot(t,xHat[2,:] - accelBias,'b',label='Error')
plot(t,-1*P_3sigma[2,:],'r--',label='3 Sigma')
ylabel('Accel. Bias Error [m]')
legend(loc='best')

subplot(414)
plot(t,P_3sigma[3,:],'r--',label='3 Sigma')
plot(t,xHat[3,:] - 1/accelScale,'b',label='Error')
plot(t,-1*P_3sigma[3,:],'r--',label='3 Sigma')
ylabel('Accel. Scale Error [ND]')
legend(loc='best')
xlabel('Time (sec)')
