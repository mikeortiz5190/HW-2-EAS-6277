#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 17:40:50 2021

@author: michaelortiz
"""


from __future__ import division
import random
import math
from scipy.integrate import odeint
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import csv

#--- COST FUNCTION ------------------------------------------------------------+

# function we are attempting to optimize (minimize)
'''def func1(x):
    total=0
    for i in range(len(x)):
        total+=x[i]**2
    return total
'''

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
'''Particle swarm Optimizer'''


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


class Particle:
    def __init__(self,x0):
        self.position_i=[]          # particle position
        self.velocity_i=[]          # particle velocity
        self.pos_best_i=[]          # best position individual
        self.err_best_i=-1          # best error individual
        self.err_i=-1               # error individual

        for i in range(0,num_dimensions):
            self.velocity_i.append(random.uniform(-1,1))
            self.position_i.append(x0[i])

    # evaluate current fitness
    def evaluate(self,costFunc):
        self.err_i=costFunc(self.position_i)

        # check to see if the current position is an individual best
        if self.err_i < self.err_best_i or self.err_best_i==-1:
            self.pos_best_i=self.position_i
            self.err_best_i=self.err_i

    # update new particle velocity
    def update_velocity(self,pos_best_g):
        w=0.5       # constant inertia weight (how much to weigh the previous velocity)
        c1=1        # cognative constant
        c2=2        # social constant

        for i in range(0,num_dimensions):
            r1=random.random()
            r2=random.random()

            vel_cognitive=c1*r1*(self.pos_best_i[i]-self.position_i[i])
            vel_social=c2*r2*(pos_best_g[i]-self.position_i[i])
            self.velocity_i[i]=w*self.velocity_i[i]+vel_cognitive+vel_social

    # update the particle position based off new velocity updates
    def update_position(self,bounds):
        for i in range(0,num_dimensions):
            self.position_i[i]=self.position_i[i]+self.velocity_i[i]

            # adjust maximum position if necessary
            if self.position_i[i]>bounds[i][1]:
                self.position_i[i]=bounds[i][1]

            # adjust minimum position if neseccary
            if self.position_i[i] < bounds[i][0]:
                self.position_i[i]=bounds[i][0]
                
             
class PSO():
    def __init__(self,costFunc,x0,bounds,num_particles,maxiter):
        global num_dimensions

        num_dimensions=len(x0)
        err_best_g=-1                   # best error for group
        pos_best_g=[]                   # best position for group

        # establish the swarm
        swarm=[]
        for i in range(0,num_particles):
            swarm.append(Particle(x0))

        # begin optimization loop
        i=0
        while i < maxiter:
            #print i,err_best_g
            # cycle through particles in swarm and evaluate fitness
            for j in range(0,num_particles):
                swarm[j].evaluate(costFunc)

                # determine if current particle is the best (globally)
                if swarm[j].err_i < err_best_g or err_best_g == -1:
                    pos_best_g=list(swarm[j].position_i)
                    err_best_g=float(swarm[j].err_i)

            # cycle through swarm and update velocities and position
            for j in range(0,num_particles):
                swarm[j].update_velocity(pos_best_g)
                swarm[j].update_position(bounds)
            i+=1

        # print final results
        print ('FINAL:')
        print (pos_best_g)
        print (err_best_g)

        
        fig = plt.figure(figsize=(12, 6))

        ax = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        # initial conditions
        z0 = [0, 0, 0, 0]
              
        # declare a time vector (time window)
        t = np.linspace(0,0.25,1001)
        
        z = odeint(odes,z0,t,(pos_best_g[0],pos_best_g[1],pos_best_g[2]))
            
        #solution for displacement vctors x1=z1 & x2=z2
        z1 = z[:,0]
        z2 = z[:,1]
        
        
        ax.set_title('Particale Swarm w/ init guess: c1,c2,c3 =5000')
        ax.plot(t, z1, color='red', label='z1')
        ax.plot(t, zT1, color='blue', label='zT1')
        ax.legend(["optimized output z1", "observed output zT1"])
        ax.set_xlabel('Time')
        ax.set_ylabel('Displacement')
        
        ax2.set_title('Particale Swarm w/ init guess: c1,c2,c3 =5000')
        ax2.plot(t, z2, color='red', label='z2')
        ax2.plot(t, zT2, color='blue', label='zT2')
        ax2.legend(["optimized output z2", "observed output zT2"])
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Displacement')
        
        
        ax.set_xlim([0, 0.25])
        ax2.set_xlim([0, 0.25])
        
        
        plt.show()

#************************************
'''Conditions'''

initial=[5000,5000,5000]               # initial starting location [x1,x2...]
bounds=[(0,10000),(0,10000),(0,10000)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...]
PSO(objective_function,initial,bounds,num_particles=500,maxiter=50)



