#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
import numpy as np
from fast_dawson import *
import time


class MomentActivation():
    def __init__(self):
        '''Moment activation function'''
        self.ds1 = Dawson1()
        self.ds2 = Dawson2()
        self.L = 0.05
        self.Tref = 5.0
        self.u = 0.0
        self.s = 0.0
        self.eps = 1e-3
        self.ub = 0.0
        self.lb = 0.0
        self.Vth = 20
        return
    
    
    def mean(self,ubar, sbar, ignore_Tref = True, cut_off = 10):
        '''Calculates the mean output firing rate given the mean & std of input firing rate'''
        
        # Divide input domain to several regions
        indx0 = sbar > 0
        indx1 = (self.Vth*self.L - ubar) < (cut_off*np.sqrt(self.L)*sbar)
        indx2 = indx0 & indx1
        
        u = np.zeros(ubar.shape)
        
        # Region 0 is approx zero for sufficiently large cut_off
        # Region 1 is calculate normally 
        ub = (1-ubar[indx2])/(sbar[indx2]*np.sqrt(self.L))
        lb = -ubar[indx2]/(sbar[indx2]*np.sqrt(self.L))
        
        meanT = 2/self.L*(self.ds1.int_fast(ub) - self.ds1.int_fast(lb))
        
        u[indx2] = 1/(meanT + self.Tref)    
        
        
        # Region 2 is calculated with analytical limit as sbar --> 0
        indx3 = np.logical_and(~indx0, ubar <= self.Vth*self.L)
        indx4 = np.logical_and(~indx0, ubar > self.Vth*self.L)
        u[indx3] = 0.0
        u[indx4] = 1/(self.Tref - 1/self.L*np.log(1-1/ubar[indx4]))     
        
        # cache the results
        self.u = u
        
        return u
           
    
    def std(self,ubar,sbar, cut_off = 10, use_cache = True):
        '''Calculates the std of output firing rate given the mean & std of input firing rate'''
        
        # Divide input domain to several regions
        indx0 = sbar > 0
        indx1 = (self.Vth*self.L - ubar) < (cut_off*np.sqrt(self.L)*sbar)
        indx2 = indx0 & indx1
        
        FF = np.zeros(ubar.shape) #Fano factor
        
        # Region 0 is approx zero for sufficiently large cut_off
        # Region 1 is calculate normally 
        ub = (1-ubar[indx2])/(sbar[indx2]*np.sqrt(self.L))
        lb = -ubar[indx2]/(sbar[indx2]*np.sqrt(self.L))        
        
        #cached mean used
        varT = 8/self.L/self.L*(  self.ds2.int_fast(ub) - self.ds2.int_fast(lb)  )                
        FF[indx2] = varT*self.u[indx2]*self.u[indx2]
        
        # Region 2 is calculated with analytical limit as sbar --> 0
        FF[~indx0] = (ubar[~indx0]<1)+0.0
        
        s = np.sqrt(FF*self.u)
        
        self.s = s
        
        return s, FF
    
    def chi(self, ubar, sbar, cut_off=10, use_cache = True):
        '''Calculates the linear response coefficient of output firing rate given the mean & std of input firing rate'''
        
        # Divide input domain to several regions
        indx0 = sbar > 0
        indx1 = (self.Vth*self.L - ubar) < (cut_off*np.sqrt(self.L)*sbar)
        indx2 = indx0 & indx1
        
        X = np.zeros(ubar.shape)
        
        # Region 0 is approx zero for sufficiently large cut_off
        # Region 1 is calculate normally 
        ub = (1-ubar[indx2])/(sbar[indx2]*np.sqrt(self.L))
        lb = -ubar[indx2]/(sbar[indx2]*np.sqrt(self.L))
        
        delta_g = self.ds1.dawson1(ub) - self.ds1.dawson1(lb)
        X[indx2] = self.u[indx2]*self.u[indx2]/(self.s[indx2])*delta_g*2/self.L/np.sqrt(self.L)

        # delta_H = self.ds2.int_fast(ub) - self.ds2.int_fast(lb)
        # X[indx2] = np.sqrt(self.u[indx2])*delta_g/np.sqrt(delta_H)/np.sqrt(2*self.L) # alternative method
        
        # Region 2 is calculated with analytical limit as sbar --> 0
        indx3 = np.logical_and(~indx0, ubar <= self.Vth*self.L)
        indx4 = np.logical_and(~indx0, ubar > self.Vth*self.L)
        
        X[indx3] = 0.0
        X[indx4] = np.sqrt(2/self.L)/np.sqrt(self.Tref - 1/self.L*np.log(1-1/ubar[indx4]))/np.sqrt(2*ubar[indx4]-1)

        #self.vol_th * u_a[indx4] * u_a[indx4] / ubar[indx4] / (ubar[indx4] - self.vol_th * self.L)
        
        self.X = X
        return X 
    
    def grad_mean(self,ubar,sbar, cut_off = 10, use_cache = True):
        '''Calculates the gradient of the mean firing rate with respect to the mean & std of input firing rate'''   
        # Divide input domain to several regions
        indx0 = sbar > 0
        indx1 = (self.Vth*self.L - ubar) < (cut_off*np.sqrt(self.L)*sbar)
        indx2 = indx0 & indx1
        
        grad_uu = np.zeros(ubar.shape) #Fano factor
        
        # Region 0 is approx zero for sufficiently large cut_off
        # Region 1 is calculate normally 
        ub = (1-ubar[indx2])/(sbar[indx2]*np.sqrt(self.L))
        lb = -ubar[indx2]/(sbar[indx2]*np.sqrt(self.L))
        
        delta_g = self.ds1.dawson1(ub) - self.ds1.dawson1(lb)        
        grad_uu[indx2] = self.u[indx2]*self.u[indx2]/sbar[indx2]*delta_g*2/self.L/np.sqrt(self.L)
                
        # Region 2 is calculated with analytical limit as sbar --> 0
        indx6 = np.logical_and(~indx0, ubar <= 1)
        indx4 = np.logical_and(~indx0, ubar > 1)
        
        grad_uu[indx6] = 0.0
        grad_uu[indx4] = self.Vth*self.u[indx4]*self.u[indx4]/ubar[indx4]/(ubar[indx4]-self.Vth*self.L)
        
        
        self.grad_uu = grad_uu
        
        #---------------
        
        grad_us = np.zeros(ubar.shape)
        temp = self.ds1.dawson1(ub)*ub - self.ds1.dawson1(lb)*lb
        grad_us[indx2] = self.u[indx2]*self.u[indx2]/sbar[indx2]*temp*2/self.L
        
        self.grad_us = grad_us        
                
        return grad_uu, grad_us
    
    def grad_std(self, ubar, sbar, cut_off = 10, use_cache = True):
        '''Calculates the gradient of the std of the firing rate with respect to the mean & std of input firing rate'''   
        
        # Divide input domain to several regions
        indx0 = sbar > 0
        indx1 = (self.Vth*self.L - ubar) < (cut_off*np.sqrt(self.L)*sbar)
        indx2 = indx0 & indx1
        
        ub = (1-ubar[indx2])/(sbar[indx2]*np.sqrt(self.L))
        lb = -ubar[indx2]/(sbar[indx2]*np.sqrt(self.L))
        
        grad_su = np.zeros(ubar.shape)
        
        delta_g = self.ds1.dawson1(ub) - self.ds1.dawson1(lb)        
        delta_h = self.ds2.dawson2(ub) - self.ds2.dawson2(lb)  
        delta_H = self.ds2.int_fast(ub) - self.ds2.int_fast(lb)
        
        temp1 = 3/self.L/np.sqrt(self.L)*self.s[indx2]/sbar[indx2]*self.u[indx2]*delta_g        
        temp2 = - 1/2/np.sqrt(self.L)*self.s[indx2]/sbar[indx2]*delta_h/delta_H
        
        grad_su[indx2] = temp1+temp2
                
        self.grad_su = grad_su
        
        #-----------
        
        grad_ss = np.zeros(ubar.shape)
        
        temp_dg = self.ds1.dawson1(ub)*ub - self.ds1.dawson1(lb)*lb
        temp_dh = self.ds2.dawson2(ub)*ub - self.ds2.dawson2(lb)*lb
        
        grad_ss[indx2] = 3/self.L*self.s[indx2]/sbar[indx2]*self.u[indx2]*temp_dg \
            - 1/2*self.s[indx2]/sbar[indx2]*temp_dh/delta_H
        
        indx4 = np.logical_and(~indx0, ubar > 1)
        
        grad_ss[indx4] = 1/np.sqrt(2*self.L)*np.power(self.u[indx4],1.5)*np.sqrt(1/(1-ubar[indx4])/(1-ubar[indx4]) - 1/ubar[indx4]/ubar[indx4])
        
        self.grad_ss = grad_ss
        
        return grad_su, grad_ss
    
    
    def grad_chi(self, ubar, sbar, cut_off = 10, use_cache = True):
        '''Calculates the gradient of the linear response coefficient with respect to the mean & std of input firing rate'''   
        
        # Divide input domain to several regions
        indx0 = sbar > 0
        indx1 = (self.Vth*self.L - ubar) < (cut_off*np.sqrt(self.L)*sbar)
        indx2 = indx0 & indx1
        
        ub = (1-ubar[indx2])/(sbar[indx2]*np.sqrt(self.L))
        lb = -ubar[indx2]/(sbar[indx2]*np.sqrt(self.L))
        
        grad_chu = np.zeros(ubar.shape)
        
        tmp1 =  self.ds1.dawson1(ub)*ub - self.ds1.dawson1(lb)*lb
        delta_g = self.ds1.dawson1(ub) - self.ds1.dawson1(lb)
        delta_H = self.ds2.int_fast(ub) - self.ds2.int_fast(lb)
        delta_h = self.ds2.dawson2(ub) - self.ds2.dawson2(lb)
        
        grad_chu[indx2] = 0.5*self.X[indx2]/self.u[indx2]*self.grad_uu[indx2] \
                 - np.sqrt(2)/self.L*np.sqrt(self.u[indx2]/delta_H)*tmp1/sbar[indx2] \
                 + self.X[indx2]*delta_h/delta_H/2/np.sqrt(self.L)/sbar[indx2]
        
        indx4 = np.logical_and(~indx0, ubar > 1)
        
        tmp_grad_uu = self.Vth*self.u[indx4]*self.u[indx4]/ubar[indx4]/(ubar[indx4]-self.Vth*self.L)
        
        grad_chu[indx4] = 1/np.sqrt(2*self.L)/np.sqrt(self.u[indx4]*(2*ubar[indx4]-1))*tmp_grad_uu \
            - np.sqrt(2/self.L)/(self.Vth*self.L)*np.sqrt(self.u[indx4])*np.power(2*ubar[indx4]-1,-1.5)
        
        
        self.grad_chu = grad_chu
        
        #-----------
        
        grad_chs = np.zeros(ubar.shape)
        
        temp_dg =  2*self.ds1.dawson1(ub)*ub*ub - 2*self.ds1.dawson1(lb)*lb*lb \
            + self.Vth*self.L/np.sqrt(self.L)/sbar[indx2]
        temp_dh = self.ds2.dawson2(ub)*ub - self.ds2.dawson2(lb)*lb
        #temp_dH = self.ds2.int_fast(ub)*ub - self.ds2.int_fast(lb)*lb

        grad_chs[indx2] = 0.5*self.X[indx2]/self.u[indx2]*self.grad_us[indx2] + \
            - self.X[indx2]/sbar[indx2]*(temp_dg/delta_g) \
                + 0.5*self.X[indx2]/sbar[indx2]/delta_H*temp_dh
        
        
        self.grad_chs = grad_chs
        
        return grad_chu, grad_chs
