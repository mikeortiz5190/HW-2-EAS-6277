#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: michaelortiz
"""
from scipy.integrate import odeint
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import csv
from numdifftools import Jacobian, Hessian

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
'''BFGS Optimizer'''

#Initial guess values for c1,  c2,  c3

c1=100
c2=110
c3=120

c_initial_guess = [c1,c2,c3]


def objective_function(c):
    
    # initial conditions
    z0 = [0, 0, 0, 0]
    
    
    # declare a time vector (time window)
    t = np.linspace(0,0.25,1001)
    
    c1 = c[0]  #f(c1)
    c2 = c[1]  #f(c2)
    c3 = c[2]  #f(c3)
    
    z = odeint(odes,z0,t,(c1,c2,c3))
    
    #solution for displacement vctors x1=z1 & x2=z2
    z1 = z[:,0]
    z2 = z[:,1]
    
    return (sum((z1-zT1)**2)+sum((z2-zT2)**2))/(2*len(t))


def fun_der(x):
    return Jacobian(lambda x: objective_function(x))(x).ravel()


bounds1 = [[0,10000],[0,10000],[0,10000]]

res = minimize(
    objective_function,
    c_initial_guess,
    method = 'BFGS',
    jac = fun_der
)

print(res)
print()

sol = res.x

fig = plt.figure(figsize=(12, 6))

ax = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

# initial conditions
z0 = [0, 0, 0, 0]
              
# declare a time vector (time window)
t = np.linspace(0,0.25,1001)
        
z = odeint(odes,z0,t,(sol[0],sol[1],sol[2]))
            
#solution for displacement vctors x1=z1 & x2=z2
z1 = z[:,0]
z2 = z[:,1]
        
      
ax.set_title('BFGS w/init guess: c1=100,c2=110,c3=120')
ax.plot(t, z1, color='red', label='z1')
ax.plot(t, zT1, color='blue', label='zT1')
ax.legend(["optimized output z1", "observed output zT1"])
ax.set_xlabel('Time')
ax.set_ylabel('Displacement')
        
ax2.set_title('BFGS w/init guess: c1=100,c2=110,c3=120')
ax2.plot(t, z2, color='red', label='z2')
ax2.plot(t, zT2, color='blue', label='zT2')
ax2.legend(["optimized output z2", "observed output zT2"])
ax2.set_xlabel('Time')
ax2.set_ylabel('Displacement')
        
       
ax.set_xlim([0, 0.25])
ax2.set_xlim([0, 0.25])
               
plt.show()



















