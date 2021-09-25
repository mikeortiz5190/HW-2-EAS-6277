#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: michaelortiz
"""
from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
import csv

'''NOTES'''
#Tuning delta_c and the learning rate have a significant impact on the gradient
#previous parameters that yielded acceptable results:
    #delta_c=1 , learning rate = 1e20
    #delta_c=2.5e-13 , learning rate = 1e10

#************************************
'''Solving the state space formulation'''


    
def odes(x, t, C1, C2, C3):
        
    #Set up the state space formualtion matricies
    
    A = np.array([[0,0,1,0],[0,0,0,1],[-1.5e5,5e4,-0.05*(C1+C2),0.05*(C2)],[1e5,-1.5e5,0.1*(C2),-0.1*(C2+C3)]])
    
    B = np.array([[0,0],[0,0],[0.05,0],[0,0.1]])
    
    #C = np.array([[1,0,0,0],[0,1,0,0]])
    
    #D = np.array([[0,0],[0,0]])
    
    #Extract the four first order ODE's from the  state space matricies
        
    # constants from state space matrix
    a1 = A[0][0]
    a2 = A[0][1]
    a3 = A[0][2]
    a4 = A[0][3]
        
    a5 = A[1][0]
    a6 = A[1][1]
    a7 = A[1][2]
    a8 = A[1][3]
        
    a9 = A[2][0]
    a10= A[2][1]
    a11= A[2][2]
    a12= A[2][3]
        
    a13= A[3][0]
    a14= A[3][1]
    a15= A[3][2]
    a16= A[3][3]
        
    b1 = B[0][0]
    b2 = B[0][1]
    b3 = B[1][0]
    b4 = B[1][1]
    b5 = B[2][0]
    b6 = B[2][1]
    b7 = B[3][0]
    b8 = B[3][1]
        
    
    # assign each ODE to a vector element
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
        
        
    #Define outputs
    u1 = 0
    u2 = 1
    
    
    # define each ODE
          
    dAdt = a1*x1 + a2*x2 + a3*x3 + a4*x4 + b1*u1 + b2*u2
    dBdt = a5*x1 + a6*x2 + a7*x3 + a8*x4 + b3*u1 + b4*u2
    dCdt = a9*x1 + a10*x2 + a11*x3 + a12*x4 + b5*u1 + b6*u2
    dDdt = a13*x1 + a14*x2 + a15*x3 + a16*x4 + b7*u1 + b8*u2
    
    return [dAdt, dBdt, dCdt, dDdt]

# initial guesses for C values
c1=1000
c2=1000
c3=1000

print('Initial guess values: c1='+str(c1)+'  c2='+str(c2)+' c3='+str(c3))
print()


#************************************
'''Output from data.csv'''

#extract the data from data.csv transform the string values to floating point numbers
data = 'data.csv'
time = []
u1 = []
u2 = []
zT1 = []
zT2 = []

with open(data, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    
    for row in csvreader:
        if row[0]=='t':
            pass
        else:
            time.append(float(row[0]))
            u1.append(float(row[1]))
            u2.append(float(row[2]))
            zT1.append(float(row[3]))
            zT2.append(float(row[4]))
            
#************************************
'''MSE to be calcualted here'''




#************************************
'''Gradiants'''

delta_C = 2.5e-13
for i in range(500):
    
    print('iteration: '+str(i))
    
    #C +/- Delta C  for finite difference         
    Dc1_f = c1 + delta_C
    Dc2_f = c2 + delta_C
    Dc3_f = c3 + delta_C
        
    Dc1_b = c1 - delta_C
    Dc2_b = c2 - delta_C
    Dc3_b = c3 - delta_C
        
    # initial conditions
    v0 = [0, 0, 0, 0]
                        
    # declare a time vector (time window)
    t = np.linspace(0,0.25,1001)
    

    #****************************************
    
    #partial of Z1/C1
    F1 = odeint(odes,v0,t,(Dc1_f,c2,c3))
    B1 = odeint(odes,v0,t,(Dc1_b,c2,c3))
        
    #solution for displacement vectors
    Z1_C1_foward = F1[:,0]
    Z1_C1_backward = B1[:,0]
    
    #grad_z1_c1 = -(Z1_C1_foward - Z1_C1_backward)/(2*delta_C)
        
    #****************************************
        
    #partial of Z1/C2
    F2 = odeint(odes,v0,t,(c1,Dc2_f,c3))
    B2 = odeint(odes,v0,t,(c1,Dc2_b,c3))
        
    #solution for displacement vectors
    Z1_C2_foward = F2[:,0]
    Z1_C2_backward = B2[:,0]
    
    #grad_z1_c2 = -(Z1_C2_foward - Z1_C2_backward)/(2*delta_C)
        
    #****************************************
        
    #partial of Z2/C2
    F3 = odeint(odes,v0,t,(c1,Dc2_f,c3))
    B3 = odeint(odes,v0,t,(c1,Dc2_b,c3))
        
    #solution for displacement vectors
    Z2_C2_foward = F3[:,1]
    Z2_C2_backward = B3[:,1]
    
    #grad_z2_c2 = -(Z2_C2_foward - Z2_C2_backward)/(2*delta_C)
        
    #****************************************
        
    #partial of Z2/C3
    F4 = odeint(odes,v0,t,(c1,c2,Dc3_f))
    B4 = odeint(odes,v0,t,(c1,c2,Dc3_b))
        
    #solution for displacement vectors
    Z2_C3_foward = F4[:,1]
    Z2_C3_backward = B4[:,1]
    
    #grad_z2_c3 = -(Z2_C3_foward - Z2_C3_backward)/(2*delta_C)
    
    
    '''MSE central point difference gradient with respect to C1'''
    
        
    MSE_Z1_C1_foward_Diff = 0
    MSE_Z1_C1_backward_Diff = 0
    
    for i in range(len(t)):
        MSE_Z1_C1_foward_Diff =+ (Z1_C1_foward[i]-zT1[i])**2
        MSE_Z1_C1_backward_Diff =+ (Z1_C1_backward[i]-zT1[i])**2
    
    MSE_Z1_C1_foward_Diff = MSE_Z1_C1_foward_Diff/len(t)
    MSE_Z1_C1_backward_Diff = MSE_Z1_C1_backward_Diff/len(t)
    #Partial of MSE with respect to C1 for n iterations
    MSE_Partial_for_Z1_C1 = -abs(MSE_Z1_C1_foward_Diff - MSE_Z1_C1_backward_Diff)/(2*delta_C)


    '''MSE central point difference gradient with respect to C2'''
    
        
    MSE_Z1_C2_foward_Diff = 0
    MSE_Z1_C2_backward_Diff = 0
    
    for i in range(len(t)):
        MSE_Z1_C2_foward_Diff =+ (Z1_C2_foward[i]-zT1[i])**2
        MSE_Z1_C2_backward_Diff =+ (Z1_C2_backward[i]-zT1[i])**2
    
    MSE_Z1_C2_foward_Diff = MSE_Z1_C2_foward_Diff/len(t)
    MSE_Z1_C2_backward_Diff = MSE_Z1_C2_backward_Diff/len(t)
    #Partial of MSE with respect to C2 for n iterations
    MSE_Partial_for_Z1_C2 = -abs(MSE_Z1_C2_foward_Diff - MSE_Z1_C2_backward_Diff)/(2*delta_C)
    

        
    MSE_Z2_C2_foward_Diff = 0
    MSE_Z2_C2_backward_Diff = 0
    
    for i in range(len(t)):
        MSE_Z2_C2_foward_Diff =+ (Z2_C2_foward[i]-zT2[i])**2
        MSE_Z2_C2_backward_Diff =+ (Z2_C2_backward[i]-zT2[i])**2
    
    MSE_Z2_C2_foward_Diff = MSE_Z2_C2_foward_Diff/len(t)
    MSE_Z2_C2_backward_Diff = MSE_Z2_C2_backward_Diff/len(t)
    #Partial of MSE with respect to C1 for n iterations
    MSE_Partial_for_Z2_C2 = -abs(MSE_Z2_C2_foward_Diff - MSE_Z2_C2_backward_Diff)/(2*delta_C)
    
    
    '''MSE central point difference gradient with respect to C3'''

        
    MSE_Z2_C3_foward_Diff = 0
    MSE_Z2_C3_backward_Diff = 0
    
    for i in range(len(t)):
        MSE_Z2_C3_foward_Diff =+ (Z2_C3_foward[i]-zT2[i])**2
        MSE_Z2_C3_backward_Diff =+ (Z2_C3_backward[i]-zT2[i])**2
    
    MSE_Z2_C3_foward_Diff = MSE_Z2_C3_foward_Diff/len(t)
    MSE_Z2_C3_backward_Diff = MSE_Z2_C3_backward_Diff/len(t)
    #Partial of MSE with respect to C1 for n iterations
    MSE_Partial_for_Z2_C3 = -abs(MSE_Z2_C3_foward_Diff - MSE_Z2_C3_backward_Diff)/(2*delta_C)
    
    '''FIND NEW C COEFFICIENTS'''
    
    learning_rate = 1e10
    
    c1 = c1 + learning_rate * MSE_Partial_for_Z1_C1
    c2 = c2 + learning_rate * (MSE_Partial_for_Z1_C2 + MSE_Partial_for_Z2_C2)
    c3 = c3 + learning_rate * MSE_Partial_for_Z2_C3
    
    print('gradient for F/c1 = '+str(MSE_Partial_for_Z1_C1))
    print('New c1 = '+str(c1))
    print()
    print('gradient for F/c2 = '+str(MSE_Partial_for_Z1_C2 + MSE_Partial_for_Z2_C2))
    print('New c2 = '+str(c2))
    print()
    print('gradient for F/c3 = '+str(MSE_Partial_for_Z2_C3))
    print('New c3 = '+str(c3))
    print()


print('Final values: c1='+str(c1)+'  c2='+str(c2)+' c3='+str(c3))
print()

#************************************
'''FINAL MEAN SQUARED ERROR'''

x = odeint(odes,v0,t,(c1,c2,c3))
    
#solution for displacement vctors x1=z1 & x2=z2
x1 = x[:,0]
x2 = x[:,1]
MSE1 = 0
MSE2 = 0

for i in range(len(t)):
    MSE1=+(x1[i]-zT1[i])**2
    MSE2=+(x2[i]-zT2[i])**2

MSE1 = MSE1/len(t)
MSE2 = MSE2/len(t)
MSE_T = (MSE1+MSE2)/2

print('MSE1 = '+str(MSE1))
print()
print('MSE2 = '+str(MSE2))
print()
print('Total MSE = '+str(MSE_T))


            
#************************************
'''plot the results'''


fig = plt.figure(figsize=(12, 6))

ax = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

x = odeint(odes,v0,t,(c1,c2,c3))

#solution for displacement vctors x1=z1 & x2=z2
x1 = x[:,0]
x2 = x[:,1]

z1_delta = []
z2_delta = []

for i in range(len(t)):
    #d1=x1[i]-zT1[i]
    #d2=x2[i]-zT2[i]
    d1=(x1[i]-zT1[i])**2
    d2=(x2[i]-zT2[i])**2
    z1_delta.append(d1)
    z2_delta.append(d2)

ax.set_title('Gradient decent w/ init guess: c1,c2,c3 =1000')
ax.plot(t, x1, color='red', label='z1')
ax.plot(t, zT1, color='blue', label='zT1')
ax.legend(["optimized output z1", "observed output zT1"])
ax.set_xlabel('Time')
ax.set_ylabel('Displacement')

ax2.set_title('Gradient decent w/ init guess: c1,c2,c3 =1000')
ax2.plot(t, x2, color='red', label='z2')
ax2.plot(t, zT2, color='blue', label='zT2')
ax2.legend(["optimized output z2", "observed output zT2"])
ax2.set_xlabel('Time')
ax2.set_ylabel('Displacement')


ax.set_xlim([0, 0.25])
ax2.set_xlim([0, 0.25])



plt.show()












