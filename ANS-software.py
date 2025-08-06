# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 21:08:31 2025

@author: alexa
"""

import numpy as np
from numpy import random as rd
from scipy.integrate import quad
from scipy.integrate import dblquad
import streamlit as st

st.title("Opportunistic Maintenance Policy - Performance Analysis")

st.markdown("Enter the parameters below and get the recommended maintenance policy:")

# Entrada de parâmetros
beta_x = st.number_input("Shape parameter (βx) for time to defect (Weibull distribution)")
eta_x = st.number_input("Scale parameter (ηx) for time to defect (Weibull distribution)")
beta_h = st.number_input("Shape parameter (βh) for delay-time (Weibull distribution)")
eta_h = st.number_input("Scale parameter (ηh) for delay-time (Weibull distribution)")
lbda = st.number_input("Rate of opportunities arrival (λ)")

Cp = st.number_input("Cost of pre-programmed preventive replacement (Cp)")
Cop = st.number_input("Cost of opportunistic preventive replacement (Cop)")
Ci = st.number_input("Cost of regular inspection (Ci)")
Coi = st.number_input("Cost of opportunistic inspection (Coi)")
Cf = st.number_input("Cost of failure (Cf)")

Cep_max = st.number_input("Cost of early preventive replacement with minimal waiting")

delta_min = st.number_input("Minimum wait for preventive maintenance after regular inspection")
delta_lim = st.number_input("Regular waiting time to provide resources for preventive maintenance")

Dp = st.number_input("Downtime for preventive (Dp)")
Df = st.number_input("Downtime for corrective (Df)")

var = st.number_input("Level of imprecision in parameters estimation (%)")

st.markdown("Enter the maintenance policy decision variables:")

# Entrada de parâmetros
L = st.number_input("L")
L = int(L)
M = st.number_input("M")
M = int(M)
N = st.number_input("N")
N = int(N)
T = st.number_input("T")
delta = st.number_input("delta")

def policy(L,M,N,T,delta,beta_x,eta_x,beta_h,eta_h,lbda,Cp,Cop,Ci,Coi,Cf,Cep_max,delta_min,delta_lim,Dp,Df):
    
    C1 = (Cp - Cep_max)/(delta_lim - delta_min)
    C2 = Cep_max - C1*delta_min
    C3 = Cp
     
    def Cep(time_lag):
        if time_lag <= delta_lim:
            Cep_val = C1*time_lag + C2
        else:
            Cep_val = C3
        return (Cep_val)
    
    Z = int(delta / T)
    Y = max(0, N - Z - 1)
    
    # Functions for X (time to defect arrival)
    def fx(x):
        if x == 0:
            fx_ = 0
        else:
            fx_ = (beta_x / eta_x) * ((x / eta_x) ** (beta_x - 1)) * np.exp(-((x / eta_x) ** beta_x))
        return fx_
    def Rx(x):
        return np.exp(-((x / eta_x) ** beta_x))
    def Fx(x):
        return 1 - np.exp(-((x / eta_x) ** beta_x))

    # Functions for H (delay-time)
    def fh(h):
        if h == 0:
            fh_ = 0
        else:
            fh_ = (beta_h / eta_h) * ((h / eta_h) ** (beta_h - 1)) * np.exp(-((h / eta_h) ** beta_h))
        return fh_
    def Rh(h):
        return np.exp(-((h / eta_h) ** beta_h))
    def Fh(h):
        return 1 - np.exp(-((h / eta_h) ** beta_h))

    # Functions for W (time between two consecutive opportunities)
    def fw(w):
        return lbda * np.exp(- lbda * w)
    def Rw(w):
        return np.exp(- lbda * w)
    def Fw(w):
        return 1 - np.exp(- lbda * w)
    
    def scenario_1(): 
        # Preventive replacement at NT, with system in good state
        P1 = Rx(N*T)*Rw((N-M)*T)
        EC1 = (Y*Ci + (M-L)*T*lbda*Coi + Cp)*P1
        EV1 = (N*T + Dp)*P1
        ED1 = Dp*P1
        return (P1, EC1, EV1, ED1)
    
    def scenario_2():
        # Opportunistic preventive replacement between MT and NT, with system in good state
        if (M < N) and (M < Y):
            P2_1 = 0; EC2_1 = 0; EV2_1 = 0
            for i in range(1, Y-M+1):
                prob2_1 = quad(lambda w: fw(w)*Rx(M*T + w), (i-1)*T, i*T)[0] 
                P2_1 = P2_1 + prob2_1
                EC2_1 = EC2_1 + ((M+i-1)*Ci + (M-L)*T*lbda*Coi + Cop)*prob2_1
                EV2_1 = EV2_1 + quad(lambda w: (M*T + w + Dp)*fw(w)*Rx(M*T + w), (i-1)*T, i*T)[0]
            
            P2_2 = quad(lambda w: fw(w)*Rx(M*T + w), (Y-M)*T, (N-M)*T)[0]
            EC2_2 = (Y*Ci + (M-L)*T*lbda*Coi + Cop)*P2_2
            EV2_2 = quad(lambda w: (M*T + w + Dp)*fw(w)*Rx(M*T + w), (Y-M)*T, (N-M)*T)[0]
            
            P2 = P2_1 + P2_2
            EC2 = EC2_1 + EC2_2
            EV2 = EV2_1 + EV2_2
            
            #EV2 = quad(lambda w: (M*T + w + Dp)*fw(w)*Rx(M*T + w), 0, (N-M)*T)[0]
            ED2 = Dp*P2
            
        if (M < N) and (M >= Y):
            P2 = quad(lambda w: fw(w)*Rx(M*T + w), 0, (N-M)*T)[0]       
            EC2 = (Y*Ci + (M-L)*T*lbda*Coi + Cop)*P2
            EV2 = quad(lambda w: (M*T + w + Dp)*fw(w)*Rx(M*T + w), 0, (N-M)*T)[0]
            ED2 = Dp*P2
        
        if (M == N):
            P2 = 0; EC2 = 0; EV2 = 0; ED2 = 0
        
        return (P2, EC2, EV2, ED2)
    
    def scenario_3():
        # Early preventive replacement after a positive in-house inspection (time lag delta)
        if (L >= 0) and (L < M) and (M < N) and (M < Y):
            P3_1 = 0; EC3_1 = 0; EV3_1 = 0
            for i in range(1, L+1):
                prob3_1 = quad(lambda x: fx(x)*Rh(i*T + delta - x)*Rw(delta), (i-1)*T, i*T)[0]
                P3_1 = P3_1 + prob3_1
                EC3_1 = EC3_1 + (i*Ci + Cep(delta))*prob3_1
                EV3_1 = EV3_1 + (i*T + delta + Dp)*prob3_1
                
            P3_2 = 0; EC3_2 = 0; EV3_2 = 0
            for i in range(L+1, M+1):
                prob3_2 = quad(lambda x: fx(x)*Rh(i*T + delta - x)*Rw(i*T + delta - x), (i-1)*T, i*T)[0]
                P3_2 = P3_2 + prob3_2
                EC3_2 = EC3_2 + quad(lambda x: (i*Ci + (x-L*T)*lbda*Coi + Cep(delta))*fx(x)*Rh(i*T + delta - x)*Rw(i*T + delta - x), (i-1)*T, i*T)[0]
                EV3_2 = EV3_2 + (i*T + delta + Dp)*prob3_2
            
            P3_3 = 0; EC3_3 = 0; EV3_3 = 0
            for i in range(M+1, Y+1):
                prob3_3 = quad(lambda x: fx(x)*Rh(i*T + delta - x)*Rw(i*T + delta - M*T), (i-1)*T, i*T)[0]
                P3_3 = P3_3 + prob3_3
                EC3_3 = EC3_3 + (i*Ci + (M-L)*T*lbda*Coi + Cep(delta))*prob3_3
                EV3_3 = EV3_3 + (i*T + delta + Dp)*prob3_3
            
            P3 = P3_1 + P3_2 + P3_3
            EC3 = EC3_1 + EC3_2 + EC3_3
            EV3 = EV3_1 + EV3_2 + EV3_3
            ED3 = Dp*P3
            
        if (L >= 0) and (L < M) and (M >= Y) and (L < Y):
            P3_1 = 0; EC3_1 = 0; EV3_1 = 0
            for i in range(1, L+1):
                prob3_1 = quad(lambda x: fx(x)*Rh(i*T + delta - x)*Rw(delta), (i-1)*T, i*T)[0]
                P3_1 = P3_1 + prob3_1
                EC3_1 = EC3_1 + (i*Ci + Cep(delta))*prob3_1
                EV3_1 = EV3_1 + (i*T + delta + Dp)*prob3_1
                
            P3_2 = 0; EC3_2 = 0; EV3_2 = 0
            for i in range(L+1, Y+1):
                prob3_2 = quad(lambda x: fx(x)*Rh(i*T + delta - x)*Rw(i*T + delta - x), (i-1)*T, i*T)[0]
                P3_2 = P3_2 + prob3_2
                EC3_2 = EC3_2 + quad(lambda x: (i*Ci + (x-L*T)*lbda*Coi + Cep(delta))*fx(x)*Rh(i*T + delta - x)*Rw(i*T + delta - x), (i-1)*T, i*T)[0]
                EV3_2 = EV3_2 + (i*T + delta + Dp)*prob3_2
            
            P3 = P3_1 + P3_2
            EC3 = EC3_1 + EC3_2
            EV3 = EV3_1 + EV3_2
            ED3 = Dp*P3
            
        if (L >= 0) and (L == M) and (M < Y):
            P3_1 = 0; EC3_1 = 0; EV3_1 = 0
            for i in range(1, L+1):
                prob3_1 = quad(lambda x: fx(x)*Rh(i*T + delta - x)*Rw(delta), (i-1)*T, i*T)[0]
                P3_1 = P3_1 + prob3_1
                EC3_1 = EC3_1 + (i*Ci + Cep(delta))*prob3_1
                EV3_1 = EV3_1 + (i*T + delta + Dp)*prob3_1
            
            P3_3 = 0; EC3_3 = 0; EV3_3 = 0
            for i in range(M+1, Y+1):
                prob3_3 = quad(lambda x: fx(x)*Rh(i*T + delta - x)*Rw(i*T + delta - M*T), (i-1)*T, i*T)[0]
                P3_3 = P3_3 + prob3_3
                EC3_3 = EC3_3 + (i*Ci + (M-L)*T*lbda*Coi + Cep(delta))*prob3_3
                EV3_3 = EV3_3 + (i*T + delta + Dp)*prob3_3
            
            P3 = P3_1 + P3_3
            EC3 = EC3_1 + EC3_3
            EV3 = EV3_1 + EV3_3
            ED3 = Dp*P3
            
        if (L >= Y) and (Y >= 1):
            P3 = 0; EC3 = 0; EV3 = 0
            for i in range(1, Y+1):
                prob3 = quad(lambda x: fx(x)*Rh(i*T + delta - x)*Rw(delta), (i-1)*T, i*T)[0]
                P3 = P3 + prob3
                EC3 = EC3 + (i*Ci + Cep(delta))*prob3
                EV3 = EV3 + (i*T + delta + Dp)*prob3
            ED3 = Dp*P3
            
        if (Y == 0):
            P3 = 0
            EC3 = 0
            EV3 = 0
            ED3 = 0
        
        return (P3, EC3, EV3, ED3)
    
    def scenario_4():
        #Opportunistic preventive replacement of a defective system
        if (L >= 0) and (L < M) and (M < N) and (M < Y):
            P4_1 = 0; EC4_1 = 0; EV4_1 = 0
            P4_2 = 0; EC4_2 = 0; EV4_2 = 0
            for i in range(1, L+1):
                #prob4_1 = 0
                prob4_2 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(i*T+w-x), (i-1)*T, i*T, lambda x: 0, lambda x: delta)[0]
                #P4_1 = P4_1 + prob4_1
                P4_2 = P4_2 + prob4_2
                #EC4_1 = EC4_1 + ((i-1)*Ci + Cop)*prob4_1
                EC4_2 = EC4_2 + (i*Ci + Cop)*prob4_2
                #EV4_1 = EV4_1 + dblquad(lambda w, x: (x + w + Dp)*fx(x)*fw(w)*Rh(w), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                EV4_2 = EV4_2 + dblquad(lambda w, x: (i*T + w + Dp)*fx(x)*fw(w)*Rh(i*T+w-x), (i-1)*T, i*T, lambda x: 0, lambda x: delta)[0]
                      
            P4_3 = sum(
                dblquad(lambda w, x: fx(x)*fw(w)*Rh(w), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                for i in range(L+1,M+1))
            P4_4 = sum(
                dblquad(lambda w, x: fx(x)*fw(w)*Rh(w), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
                for i in range(L+1,M+1))
            EC4_3 = sum(
                dblquad(lambda w, x: ((i-1)*Ci + (x-L*T)*lbda*Coi + Cop)*fx(x)*fw(w)*Rh(w), 
                        (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                for i in range(L+1,M+1))
            EC4_4 = sum(
                dblquad(lambda w, x: (i*Ci + (x-L*T)*lbda*Coi + Cop)*fx(x)*fw(w)*Rh(w), 
                        (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
                for i in range(L+1,M+1))
            EV4_3 = sum(
                dblquad(lambda w, x: (x + w + Dp)*fx(x)*fw(w)*Rh(w), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                for i in range(L+1,M+1))
            EV4_4 = sum(
                dblquad(lambda w, x: (x + w + Dp)*fx(x)*fw(w)*Rh(w), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
                for i in range(L+1,M+1))
            
            P4_5 = 0; EC4_5 = 0; EV4_5 = 0
            P4_6 = 0; EC4_6 = 0; EV4_6 = 0
            for i in range(M+1, Y+1):
                prob4_5 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(M*T + w - x), (i-1)*T, i*T, lambda x: x-M*T, lambda x: (i-M)*T)[0]
                prob4_6 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(M*T + w - x), (i-1)*T, i*T, lambda x: (i-M)*T, lambda x: (i-M)*T+delta)[0]
                P4_5 = P4_5 + prob4_5
                P4_6 = P4_6 + prob4_6
                EC4_5 = EC4_5 + ((i-1)*Ci + (M-L)*T*lbda*Coi + Cop)*prob4_5
                EC4_6 = EC4_6 + (i*Ci + (M-L)*T*lbda*Coi + Cop)*prob4_6
                EV4_5 = EV4_5 + dblquad(lambda w, x: (M*T + w + Dp)*fx(x)*fw(w)*Rh(M*T + w - x), (i-1)*T, i*T, lambda x: x-M*T, lambda x: (i-M)*T)[0]
                EV4_6 = EV4_6 + dblquad(lambda w, x: (M*T + w + Dp)*fx(x)*fw(w)*Rh(M*T + w - x), (i-1)*T, i*T, lambda x: (i-M)*T, lambda x: (i-M)*T+delta)[0]

            P4_7 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(M*T + w - x), Y*T, N*T, lambda x: x-M*T, lambda x: (N-M)*T)[0]
            EC4_7 = (Y*Ci + (M-L)*T*lbda*Coi + Cop)*P4_7
            EV4_7 = dblquad(lambda w, x: (M*T + w + Dp)*fx(x)*fw(w)*Rh(M*T + w - x), Y*T, N*T, lambda x: x-M*T, lambda x: (N-M)*T)[0]
                 
            P4 = P4_1 + P4_2 + P4_3 + P4_4 + P4_5 + P4_6 + P4_7
            EC4 = EC4_1 + EC4_2 + EC4_3 + EC4_4 + EC4_5 + EC4_6 + EC4_7
            EV4 = EV4_1 + EV4_2 + EV4_3 + EV4_4 + EV4_5 + EV4_6 + EV4_7
            ED4 = Dp*P4
            
        if (L >= 0) and (L < M) and (M >= Y) and (Y > L):
            P4_1 = 0; EC4_1 = 0; EV4_1 = 0
            P4_2 = 0; EC4_2 = 0; EV4_2 = 0
            for i in range(1, L+1):
                #prob4_1 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(w), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                prob4_2 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(i*T+w-x), (i-1)*T, i*T, lambda x: 0, lambda x: delta)[0]
                #P4_1 = P4_1 + prob4_1
                P4_2 = P4_2 + prob4_2
                #EC4_1 = EC4_1 + ((i-1)*Ci + Cop)*prob4_1
                EC4_2 = EC4_2 + (i*Ci + Cop)*prob4_2
                #EV4_1 = EV4_1 + dblquad(lambda w, x: (x + w + Dp)*fx(x)*fw(w)*Rh(w), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                EV4_2 = EV4_2 + dblquad(lambda w, x: (i*T + w + Dp)*fx(x)*fw(w)*Rh(i*T+w-x), (i-1)*T, i*T, lambda x: 0, lambda x: delta)[0]
                      
            P4_3 = sum(
                dblquad(lambda w, x: fx(x)*fw(w)*Rh(w), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                for i in range(L+1,Y+1))
            P4_4 = sum(
                dblquad(lambda w, x: fx(x)*fw(w)*Rh(w), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
                for i in range(L+1,Y+1))
            EC4_3 = sum(
                dblquad(lambda w, x: ((i-1)*Ci + (x-L*T)*lbda*Coi + Cop)*fx(x)*fw(w)*Rh(w), 
                        (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                for i in range(L+1,Y+1))
            EC4_4 = sum(
                dblquad(lambda w, x: (i*Ci + (x-L*T)*lbda*Coi + Cop)*fx(x)*fw(w)*Rh(w), 
                        (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
                for i in range(L+1,Y+1))
            EV4_3 = sum(
                dblquad(lambda w, x: (x + w + Dp)*fx(x)*fw(w)*Rh(w), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                for i in range(L+1,Y+1))
            EV4_4 = sum(
                dblquad(lambda w, x: (x + w + Dp)*fx(x)*fw(w)*Rh(w), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
                for i in range(L+1,Y+1))
            
            
            P4_5 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(w), Y*T, M*T, lambda x: 0, lambda x: N*T-x)[0]
            P4_6 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(M*T + w - x), M*T, N*T, lambda x: x-M*T, lambda x: (N-M)*T)[0]
            EC4_5 = dblquad(lambda w, x: (Y*Ci + (x-L*T)*lbda*Coi + Cop)*fx(x)*fw(w)*Rh(w), Y*T, M*T, lambda x: 0, lambda x: N*T-x)[0]
            EC4_6 = (Y*Ci + (M-L)*T*lbda*Coi + Cop)*P4_6
            EV4_5 = dblquad(lambda w, x: (x + w + Dp)*fx(x)*fw(w)*Rh(w), Y*T, M*T, lambda x: 0, lambda x: N*T-x)[0]
            EV4_6 = dblquad(lambda w, x: (M*T + w + Dp)*fx(x)*fw(w)*Rh(M*T + w - x), M*T, N*T, lambda x: x-M*T, lambda x: (N-M)*T)[0]
                
            P4 = P4_1 + P4_2 + P4_3 + P4_4 + P4_5 + P4_6
            EC4 = EC4_1 + EC4_2 + EC4_3 + EC4_4 + EC4_5 + EC4_6
            EV4 = EV4_1 + EV4_2 + EV4_3 + EV4_4 + EV4_5 + EV4_6 
            ED4 = Dp*P4
            
        if (L >= 0) and (L == M) and (M < Y):
            P4_1 = 0; EC4_1 = 0; EV4_1 = 0
            P4_2 = 0; EC4_2 = 0; EV4_2 = 0
            for i in range(1, L+1):
                #prob4_1 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(w), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                prob4_2 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(i*T+w-x), (i-1)*T, i*T, lambda x: 0, lambda x: delta)[0]
                #P4_1 = P4_1 + prob4_1
                P4_2 = P4_2 + prob4_2
                #EC4_1 = EC4_1 + ((i-1)*Ci + Cop)*prob4_1
                EC4_2 = EC4_2 + (i*Ci + Cop)*prob4_2
                #EV4_1 = EV4_1 + dblquad(lambda w, x: (x + w + Dp)*fx(x)*fw(w)*Rh(w), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                EV4_2 = EV4_2 + dblquad(lambda w, x: (i*T + w + Dp)*fx(x)*fw(w)*Rh(i*T+w-x), (i-1)*T, i*T, lambda x: 0, lambda x: delta)[0]
            
            P4_3 = 0; EC4_3 = 0; EV4_3 = 0
            P4_4 = 0; EC4_4 = 0; EV4_4 = 0
            for i in range(L+1, Y+1):
                prob4_3 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(M*T + w - x), (i-1)*T, i*T, lambda x: x-M*T, lambda x: (i-M)*T)[0]
                prob4_4 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(M*T + w - x), (i-1)*T, i*T, lambda x: (i-M)*T, lambda x: (i-M)*T+delta)[0]
                P4_3 = P4_3 + prob4_3
                P4_4 = P4_4 + prob4_4
                EC4_3 = EC4_3 + ((i-1)*Ci + Cop)*prob4_3
                EC4_4 = EC4_4 + (i*Ci + Cop)*prob4_4
                EV4_3 = EV4_3 + dblquad(lambda w, x: (M*T + w + Dp)*fx(x)*fw(w)*Rh(M*T + w - x), (i-1)*T, i*T, lambda x: x-M*T, lambda x: (i-M)*T)[0]
                EV4_4 = EV4_4 + dblquad(lambda w, x: (M*T + w + Dp)*fx(x)*fw(w)*Rh(M*T + w - x), (i-1)*T, i*T, lambda x: (i-M)*T, lambda x: (i-M)*T+delta)[0] 
                
            P4_5 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(M*T + w - x), Y*T, N*T, lambda x: x-M*T, lambda x: (N-M)*T)[0]
            EC4_5 = (Y*Ci + Cop)*dblquad(lambda w, x: fx(x)*fw(w)*Rh(M*T + w - x), Y*T, N*T, lambda x: x-M*T, lambda x: (N-M)*T)[0]
            EV4_5 = dblquad(lambda w, x: (M*T + w + Dp)*fx(x)*fw(w)*Rh(M*T + w - x), Y*T, N*T, lambda x: x-M*T, lambda x: (N-M)*T)[0]
            
            P4 = P4_1 + P4_2 + P4_3 + P4_4 + P4_5
            EC4 = EC4_1 + EC4_2 + EC4_3 + EC4_4 + EC4_5
            EV4 = EV4_1 + EV4_2 + EV4_3 + EV4_4 + EV4_5
            ED4 = Dp*P4
           
        if (Y >= 1) and (Y <= L):
            P4_1 = 0; EC4_1 = 0; EV4_1 = 0
            P4_2 = 0; EC4_2 = 0; EV4_2 = 0
            for i in range(1, Y+1):
                #prob4_1 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(w), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                prob4_2 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(i*T+w-x), (i-1)*T, i*T, lambda x: 0, lambda x: delta)[0]
                #P4_1 = P4_1 + prob4_1
                P4_2 = P4_2 + prob4_2
                #EC4_1 = EC4_1 + ((i-1)*Ci + Cop)*prob4_1
                EC4_2 = EC4_2 + (i*Ci + Cop)*prob4_2
                #EV4_1 = EV4_1 + dblquad(lambda w, x: (x + w + Dp)*fx(x)*fw(w)*Rh(w), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                EV4_2 = EV4_2 + dblquad(lambda w, x: (i*T + w + Dp)*fx(x)*fw(w)*Rh(i*T+w-x), (i-1)*T, i*T, lambda x: 0, lambda x: delta)[0]
                
            P4_3 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(L*T+w-x), Y*T, L*T, lambda x: 0, lambda x: (N-L)*T)[0]
            P4_4 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(w), L*T, M*T, lambda x: 0, lambda x: N*T-x)[0]
            P4_5 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(M*T + w - x), M*T, N*T, lambda x: x-M*T, lambda x: (N-M)*T)[0]
            P4 = P4_1 + P4_2 + P4_3 + P4_4 + P4_5
            
            EC4_3 = (Y*Ci + Cop)*P4_3
            EC4_4 = dblquad(lambda w, x: (Y*Ci + (x-L*T)*lbda*Coi + Cop)*fx(x)*fw(w)*Rh(w), L*T, M*T, lambda x: 0, lambda x: N*T-x)[0]
            EC4_5 = (Y*Ci + (M-L)*T*lbda*Coi + Cop)*P4_5
            EC4 = EC4_1 + EC4_2 + EC4_3 + EC4_4 + EC4_5
            
            EV4_3 = dblquad(lambda w, x: (L*T + w + Dp)*fx(x)*fw(w)*Rh(L*T+w-x), Y*T, L*T, lambda x: 0, lambda x: (N-L)*T)[0]
            EV4_4 = dblquad(lambda w, x: (x + w + Dp)*fx(x)*fw(w)*Rh(w), L*T, M*T, lambda x: 0, lambda x: N*T-x)[0]
            EV4_5 = dblquad(lambda w, x: (M*T + w + Dp)*fx(x)*fw(w)*Rh(M*T + w - x), M*T, N*T, lambda x: x-M*T, lambda x: (N-M)*T)[0]
            EV4 = EV4_1 + EV4_2 + EV4_3 + EV4_4 + EV4_5
            
            ED4 = Dp*P4
              
        if (Y == 0):
            P4_1 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(L*T+w-x), 0, L*T, lambda x: 0, lambda x: (N-L)*T)[0]
            P4_2 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(w), L*T, M*T, lambda x: 0, lambda x: N*T-x)[0]
            P4_3 = dblquad(lambda w, x: fx(x)*fw(w)*Rh(M*T + w - x), M*T, N*T, lambda x: x-M*T, lambda x: (N-M)*T)[0]
                
            P4 = P4_1 + P4_2 + P4_3
            
            EC4_1 = Cop*P4_1
            EC4_2 = dblquad(lambda w, x: ((x-L*T)*lbda*Coi + Cop)*fx(x)*fw(w)*Rh(w), L*T, M*T, lambda x: 0, lambda x: N*T-x)[0]
            EC4_3 = ((M-L)*T*lbda*Coi + Cop)*P4_3
                
            EC4 = EC4_1 + EC4_2 + EC4_3
            
            EV4_1 = dblquad(lambda w, x: (L*T + w + Dp)*fx(x)*fw(w)*Rh(L*T+w-x), 0, L*T, lambda x: 0, lambda x: (N-L)*T)[0]
            EV4_2 = dblquad(lambda w, x: (x + w + Dp)*fx(x)*fw(w)*Rh(w), L*T, M*T, lambda x: 0, lambda x: N*T-x)[0]
            EV4_3 = dblquad(lambda w, x: (M*T + w + Dp)*fx(x)*fw(w)*Rh(M*T + w - x), M*T, N*T, lambda x: x-M*T, lambda x: (N-M)*T)[0]
                
            EV4 = EV4_1 + EV4_2 + EV4_3
            
            ED4 = Dp*P4
        
        return (P4, EC4, EV4, ED4)
    
    def scenario_5():
        # Preventive replacement at N.T with system in defective state
        if (Y <= L):
            P5_1 = quad(lambda x: fx(x)*Rh(N*T-x)*Rw((N-L)*T), Y*T, L*T)[0]
            P5_2 = quad(lambda x: fx(x)*Rh(N*T-x)*Rw(N*T-x), L*T, M*T)[0]
            P5_3 = quad(lambda x: fx(x)*Rh(N*T-x)*Rw((N-M)*T), M*T, N*T)[0]
            
            P5 = P5_1 + P5_2 + P5_3
            
            EC5_1 = (Y*Ci + Cp)*P5_1
            EC5_2 = quad(lambda x: (Y*Ci + (x - L*T)*lbda*Coi + Cp)*fx(x)*Rh(N*T-x)*Rw(N*T-x), L*T, M*T)[0]
            EC5_3 = (Y*Ci + (M-L)*T*lbda*Coi + Cp)*P5_3
            
            EC5 = EC5_1 + EC5_2 + EC5_3
            
            EV5 = (N*T + Dp)*P5
            ED5 = Dp*P5
            
        if (L < Y) and (Y <= M):
            P5_1 = quad(lambda x: fx(x)*Rh(N*T-x)*Rw(N*T-x), Y*T, M*T)[0]
            P5_2 = quad(lambda x: fx(x)*Rh(N*T-x)*Rw((N-M)*T), M*T, N*T)[0]
            
            P5 = P5_1 + P5_2
            
            EC5_1 = quad(lambda x: (Y*Ci + (x - L*T)*lbda*Coi + Cp)*fx(x)*Rh(N*T-x)*Rw(N*T-x), Y*T, M*T)[0]
            EC5_2 = (Y*Ci + (M-L)*T*lbda*Coi + Cp)*P5_2
            
            EC5 = EC5_1 + EC5_2
            
            EV5 = (N*T + Dp)*P5
            ED5 = Dp*P5
            
        if (Y >= M):
            P5 = quad(lambda x: fx(x)*Rh(N*T-x)*Rw((N-M)*T), Y*T, N*T)[0]

            EC5 = (Y*Ci + (M-L)*T*lbda*Coi + Cp)*P5
            
            EV5 = (N*T + Dp)*P5
            ED5 = Dp*P5
            
        return(P5, EC5, EV5, ED5)
    
    def scenario_6():
        if (L >= 0) and (L < M) and (M < N) and (M < Y):
            P6_1 = 0; EC6_1 = 0; EV6_1 = 0
            P6_2 = 0; EC6_2 = 0; EV6_2 = 0
            for i in range(1, L+1):
                prob6_1 = dblquad(lambda h, x: fx(x)*fh(h), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                prob6_2 = dblquad(lambda h, x: fx(x)*fh(h)*Rw(x+h-i*T), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
                P6_1 = P6_1 + prob6_1
                P6_2 = P6_2 + prob6_2
                EC6_1 = EC6_1 + ((i-1)*Ci + Cf)*prob6_1
                EC6_2 = EC6_2 + (i*Ci + Cf)*prob6_2
                EV6_1 = EV6_1 + dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                EV6_2 = EV6_2 + dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(x+h-i*T), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
            
            P6_3 = sum(
                dblquad(lambda h, x: fx(x)*fh(h)*Rw(h), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                for i in range(L+1,M+1))
            P6_4 = sum(
                dblquad(lambda h, x: fx(x)*fh(h)*Rw(h), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
                for i in range(L+1,M+1))
            EC6_3 = sum(
                dblquad(lambda h, x: ((i-1)*Ci + (x- L*T)*lbda*Coi + Cf)*fx(x)*fh(h)*Rw(h), 
                        (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                for i in range(L+1,M+1))
            EC6_4 = sum(
                dblquad(lambda h, x: (i*Ci + (x- L*T)*lbda*Coi + Cf)*fx(x)*fh(h)*Rw(h), 
                        (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
                for i in range(L+1,M+1))
            EV6_3 = sum(
                dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(h), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                for i in range(L+1,M+1))
            EV6_4 = sum(
                dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(h), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
                for i in range(L+1,M+1))
            
            P6_5 = 0; EC6_5 = 0; EV6_5 = 0
            P6_6 = 0; EC6_6 = 0; EV6_6 = 0
            for i in range(M+1, Y+1):
                prob6_5 = dblquad(lambda h, x: fx(x)*fh(h)*Rw(x+h-M*T), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                prob6_6 = dblquad(lambda h, x: fx(x)*fh(h)*Rw(x+h-M*T), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
                P6_5 = P6_5 + prob6_5
                P6_6 = P6_6 + prob6_6
                EC6_5 = EC6_5 + ((i-1)*Ci + (M-L)*T*lbda*Coi + Cf)*prob6_5
                EC6_6 = EC6_6 + (i*Ci + (M-L)*T*lbda*Coi + Cf)*prob6_6
                EV6_5 = EV6_5 + dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(x+h-M*T), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                EV6_6 = EV6_6 + dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(x+h-M*T), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]

            P6_7 = dblquad(lambda h, x: fx(x)*fh(h)*Rw(x+h-M*T), Y*T, N*T, lambda x: 0, lambda x: N*T-x)[0]
            P6 = P6_1 + P6_2 + P6_3 + P6_4 + P6_5 + P6_6 + P6_7
            
            EC6_7 = (Y*Ci + (M-L)*T*lbda*Coi + Cf)*P6_7
            EC6 = EC6_1 + EC6_2 + EC6_3 + EC6_4 + EC6_5 + EC6_6 + EC6_7
            
            EV6_7 = dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(x+h-M*T), Y*T, N*T, lambda x: 0, lambda x: N*T-x)[0]
            EV6 = EV6_1 + EV6_2 + EV6_3 + EV6_4 + EV6_5 + EV6_6 + EV6_7
            
            ED6 = Df*P6
            
        if (L >= 0) and (L < M) and (M >= Y) and (Y > L):
            P6_1 = 0; EC6_1 = 0; EV6_1 = 0
            P6_2 = 0; EC6_2 = 0; EV6_2 = 0
            for i in range(1, L+1):
                prob6_1 = dblquad(lambda h, x: fx(x)*fh(h), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                prob6_2 = dblquad(lambda h, x: fx(x)*fh(h)*Rw(x+h-i*T), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
                P6_1 = P6_1 + prob6_1
                P6_2 = P6_2 + prob6_2
                EC6_1 = EC6_1 + ((i-1)*Ci + Cf)*prob6_1
                EC6_2 = EC6_2 + (i*Ci + Cf)*prob6_2
                EV6_1 = EV6_1 + dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                EV6_2 = EV6_2 + dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(x+h-i*T), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
            
            P6_3 = sum(
                dblquad(lambda h, x: fx(x)*fh(h)*Rw(h), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                for i in range(L+1,Y+1))
            P6_4 = sum(
                dblquad(lambda h, x: fx(x)*fh(h)*Rw(h), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
                for i in range(L+1,Y+1))
            P6_5 = dblquad(lambda h, x: fx(x)*fh(h)*Rw(h), Y*T, M*T, lambda x: 0, lambda x: N*T-x)[0]
            P6_6 = dblquad(lambda h, x: fx(x)*fh(h)*Rw(x+h-M*T), M*T, N*T, lambda x: 0, lambda x: N*T-x)[0]
            P6 = P6_1 + P6_2 + P6_3 + P6_4 + P6_5 + P6_6
            
            EC6_3 = sum(
                dblquad(lambda h, x: ((i-1)*Ci + (x- L*T)*lbda*Coi + Cf)*fx(x)*fh(h)*Rw(h), 
                        (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                for i in range(L+1,Y+1))
            EC6_4 = sum(
                dblquad(lambda h, x: (i*Ci + (x- L*T)*lbda*Coi + Cf)*fx(x)*fh(h)*Rw(h), 
                        (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
                for i in range(L+1,Y+1))
            EC6_5 = dblquad(lambda h, x: (Y*Ci + (x - L*T)*lbda*Coi + Cf)*fx(x)*fh(h)*Rw(h), Y*T, M*T, lambda x: 0, lambda x: N*T-x)[0]
            EC6_6 = (Y*Ci + (M-L)*T*lbda*Coi + Cf)*P6_6
            EC6 = EC6_1 + EC6_2 + EC6_3 + EC6_4 + EC6_5 + EC6_6
            
            EV6_3 = sum(
                dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(h), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                for i in range(L+1,Y+1))
            EV6_4 = sum(
                dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(h), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
                for i in range(L+1,Y+1))
            EV6_5 = dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(h), Y*T, M*T, lambda x: 0, lambda x: N*T-x)[0]
            EV6_6 = dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(x+h-M*T), M*T, N*T, lambda x: 0, lambda x: N*T-x)[0]
            EV6 = EV6_1 + EV6_2 + EV6_3 + EV6_4 + EV6_5 + EV6_6
            
            ED6 = Df*P6
            
        if (L >= 0) and (L == M) and (M < Y):
            P6_1 = 0; EC6_1 = 0; EV6_1 = 0
            P6_2 = 0; EC6_2 = 0; EV6_2 = 0
            for i in range(1, L+1):
                prob6_1 = dblquad(lambda h, x: fx(x)*fh(h), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                prob6_2 = dblquad(lambda h, x: fx(x)*fh(h)*Rw(x+h-i*T), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
                P6_1 = P6_1 + prob6_1
                P6_2 = P6_2 + prob6_2
                EC6_1 = EC6_1 + ((i-1)*Ci + Cf)*prob6_1
                EC6_2 = EC6_2 + (i*Ci + Cf)*prob6_2
                EV6_1 = EV6_1 + dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                EV6_2 = EV6_2 + dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(x+h-i*T), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
            
            P6_5 = 0; EC6_5 = 0; EV6_5 = 0
            P6_6 = 0; EC6_6 = 0; EV6_6 = 0
            for i in range(M+1, Y+1):
                prob6_5 = dblquad(lambda h, x: fx(x)*fh(h)*Rw(x+h-M*T), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                prob6_6 = dblquad(lambda h, x: fx(x)*fh(h)*Rw(x+h-M*T), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
                P6_5 = P6_5 + prob6_5
                P6_6 = P6_6 + prob6_6
                EC6_5 = EC6_5 + ((i-1)*Ci + (M-L)*T*lbda*Coi + Cf)*prob6_5
                EC6_6 = EC6_6 + (i*Ci + (M-L)*T*lbda*Coi + Cf)*prob6_6
                EV6_5 = EV6_5 + dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(x+h-M*T), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                EV6_6 = EV6_6 + dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(x+h-M*T), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]

            P6_7 = dblquad(lambda h, x: fx(x)*fh(h)*Rw(x+h-M*T), Y*T, N*T, lambda x: 0, lambda x: N*T-x)[0]
            P6 = P6_1 + P6_2 + P6_5 + P6_6 + P6_7
            
            EC6_7 = (Y*Ci + (M-L)*T*lbda*Coi + Cf)*P6_7
            EC6 = EC6_1 + EC6_2 + EC6_5 + EC6_6 + EC6_7
            
            EV6_7 = dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(x+h-M*T), Y*T, N*T, lambda x: 0, lambda x: N*T-x)[0]
            EV6 = EV6_1 + EV6_2 + EV6_5 + EV6_6 + EV6_7
            
            ED6 = Df*P6
            
        if (Y >= 1) and (Y <= L):
            P6_1 = 0; EC6_1 = 0; EV6_1 = 0
            P6_2 = 0; EC6_2 = 0; EV6_2 = 0
            for i in range(1,Y+1):
                prob6_1 = dblquad(lambda h, x: fx(x)*fh(h), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                prob6_2 = dblquad(lambda h, x: fx(x)*fh(h)*Rw(x+h-i*T), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]
                P6_1 = P6_1 + prob6_1
                P6_2 = P6_2 + prob6_2
                EC6_1 = EC6_1 + ((i-1)*Ci + Cf)*prob6_1
                EC6_2 = EC6_2 + (i*Ci + Cf)*prob6_2
                EV6_1 = EV6_1 + dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h), (i-1)*T, i*T, lambda x: 0, lambda x: i*T-x)[0]
                EV6_2 = EV6_2 + dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(x+h-i*T), (i-1)*T, i*T, lambda x: i*T-x, lambda x: i*T+delta-x)[0]

            P6_3 = dblquad(lambda h, x: fx(x)*fh(h), Y*T, L*T, lambda x: 0, lambda x: L*T-x)[0]
            P6_4 = dblquad(lambda h, x: fx(x)*fh(h)*Rw(x+h-L*T), Y*T, L*T, lambda x: L*T-x, lambda x: N*T-x)[0]
            P6_5 = dblquad(lambda h, x: fx(x)*fh(h)*Rw(h), L*T, M*T, lambda x: 0, lambda x: N*T-x)[0]
            P6_6 = dblquad(lambda h, x: fx(x)*fh(h)*Rw(x+h-M*T), M*T, N*T, lambda x: 0, lambda x: N*T-x)[0]
            P6 = P6_1 + P6_2 + P6_3 + P6_4 + P6_5 + P6_6
            
            EC6_3 = (Y*Ci + Cf)*P6_3
            EC6_4 = (Y*Ci + Cf)*P6_4
            EC6_5 = dblquad(lambda h, x: (Y*Ci + (x-L*T)*lbda*Coi + Cf)*fx(x)*fh(h)*Rw(h), L*T, M*T, lambda x: 0, lambda x: N*T-x)[0]
            EC6_6 = (Y*Ci + (M-L)*T*lbda*Coi + Cf)*P6_6
            EC6 = EC6_1 + EC6_2 + EC6_3 + EC6_4 + EC6_5 + EC6_6
            
            EV6_3 = dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h), Y*T, L*T, lambda x: 0, lambda x: L*T-x)[0]
            EV6_4 = dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(x+h-L*T), Y*T, L*T, lambda x: L*T-x, lambda x: N*T-x)[0]
            EV6_5 = dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(h), L*T, M*T, lambda x: 0, lambda x: N*T-x)[0]
            EV6_6 = dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(x+h-M*T), M*T, N*T, lambda x: 0, lambda x: N*T-x)[0]
            EV6 = EV6_1 + EV6_2 + EV6_3 + EV6_4 + EV6_5 + EV6_6
            
            ED6 = Df*P6
            
        if (Y == 0):
            P6_1 = dblquad(lambda h, x: fx(x)*fh(h), 0, L*T, lambda x: 0, lambda x: L*T-x)[0]
            P6_2 = dblquad(lambda h, x: fx(x)*fh(h)*Rw(x+h-L*T), 0, L*T, lambda x: L*T-x, lambda x: N*T-x)[0]
            P6_3 = dblquad(lambda h, x: fx(x)*fh(h)*Rw(h), L*T, M*T, lambda x: 0, lambda x: N*T-x)[0]
            P6_4 = dblquad(lambda h, x: fx(x)*fh(h)*Rw(x+h-M*T), M*T, N*T, lambda x: 0, lambda x: N*T-x)[0]
            P6 = P6_1 + P6_2 + P6_3 + P6_4
            
            EC6_1 = Cf*P6_1
            EC6_2 = Cf*P6_2
            EC6_3 = dblquad(lambda h, x: ((x-L*T)*lbda*Coi + Cf)*fx(x)*fh(h)*Rw(h), L*T, M*T, lambda x: 0, lambda x: N*T-x)[0]
            EC6_4 = ((M-L)*T*lbda*Coi + Cf)*P6_4
            EC6 = EC6_1 + EC6_2 + EC6_3 + EC6_4

            EV6_1 = dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h), 0, L*T, lambda x: 0, lambda x: L*T-x)[0]
            EV6_2 = dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(x+h-L*T), 0, L*T, lambda x: L*T-x, lambda x: N*T-x)[0]
            EV6_3 = dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(h), L*T, M*T, lambda x: 0, lambda x: N*T-x)[0]
            EV6_4 = dblquad(lambda h, x: (x+h+Df)*fx(x)*fh(h)*Rw(x+h-M*T), M*T, N*T, lambda x: 0, lambda x: N*T-x)[0]
            EV6 = EV6_1 + EV6_2 + EV6_3 + EV6_4
            
            ED6 = Df*P6
            
        return (P6, EC6, EV6, ED6)
    
    (P1, EC1, EV1, ED1) = scenario_1()
    (P2, EC2, EV2, ED2) = scenario_2()
    (P3, EC3, EV3, ED3) = scenario_3()        
    (P4, EC4, EV4, ED4) = scenario_4()        
    (P5, EC5, EV5, ED5) = scenario_5()        
    (P6, EC6, EV6, ED6) = scenario_6()
    
    P_total = P1 + P2 + P3 + P4 + P5 + P6
    EC = EC1 + EC2 + EC3 + EC4 + EC5 + EC6
    EV = EV1 + EV2 + EV3 + EV4 + EV5 + EV6
    ED = ED1 + ED2 + ED3 + ED4 + ED5 + ED6
    
    cost_rate = EC/EV
    MTBOF = EV/P6
    availability = 1 - (ED/EV)

    print(cost_rate, MTBOF, availability)
    
    return (P_total, EC, EV, ED, cost_rate, MTBOF, availability, P1, P2, P3, P4, P5, P6)

def ANS(L,M,N,T,delta):
    
    sample_cr = np.zeros(100)
    sample_mtbof = np.zeros(100)
    sample_av = np.zeros(100)
    
    for i in range(0,100):
        beta_xs = rd.uniform((1-var/100)*beta_x, (1+var/100)*beta_x)
        eta_xs = rd.uniform((1-var/100)*eta_x, (1+var/100)*eta_x)
        beta_hs = rd.uniform((1-var/100)*beta_h, (1+var/100)*beta_h)
        eta_hs = rd.uniform((1-var/100)*eta_h, (1+var/100)*eta_h)
        lbdas = rd.uniform((1-var/100)*lbda, (1+var/100)*lbda)
        Cps = rd.uniform((1-var/100)*Cp, (1+var/100)*Cp)
        Cops = rd.uniform((1-var/100)*Cop, (1+var/100)*Cop)
        Cis = rd.uniform((1-var/100)*Ci, (1+var/100)*Ci)
        Cois = rd.uniform((1-var/100)*Coi, (1+var/100)*Coi)
        Cfs = rd.uniform((1-var/100)*Cf, (1+var/100)*Cf)
        Cep_maxs = rd.uniform((1-var/100)*Cep_max, (1+var/100)*Cep_max)
        Dps = rd.uniform((1-var/100)*Dp, (1+var/100)*Dp)
        Dfs = rd.uniform((1-var/100)*Df, (1+var/100)*Df)
        print(beta_xs,eta_xs,beta_hs,eta_hs,lbdas,Cps,Cops,Cis,Cois,Cfs,Cep_maxs,delta_min,delta_lim,Dps,Dfs)
        P_total, EC, EV, ED, cost_rate, MTBOF, availability, P1, P2, P3, P4, P5, P6 = policy(L,M,N,T,delta,beta_xs,eta_xs,beta_hs,eta_hs,lbdas,Cps,Cops,Cis,Cois,Cfs,Cep_maxs,delta_min,delta_lim,Dps,Dfs)
        sample_cr[i] = cost_rate
        sample_mtbof[i] = MTBOF
        sample_av[i] = availability
    
    cr_mean = np.mean(sample_cr)
    cr_std = np.std(sample_cr)
    mtbof_mean = np.mean(sample_mtbof)
    mtbof_std = np.std(sample_mtbof)
    av_mean = np.mean(sample_av)
    av_std = np.std(sample_av)
    
    return(cr_mean, cr_std, mtbof_mean, mtbof_std, av_mean, av_std)

# Executar
if st.button("Calculate Expected Performance"):
    with st.spinner('⏳ Calculating...'):
        results = policy(L,M,N,T,delta,beta_x,eta_x,beta_h,eta_h,lbda,Cp,Cop,Ci,Coi,Cf,Cep_max,delta_min,delta_lim,Dp,Df)

    st.success("✅")
    st.markdown("**Maintenance policy performance metrics**")
    st.write({
        "Cost-rate": results[4],
        "MTBOF (mean time between failures after policy implementation)": results[5],
        "Availability": results[6]
    })

if st.button("Sensitivity Analysis"):
    with st.spinner('⏳ Running sensitivity analysis...'):
        
        results = ANS(L,M,N,T,delta)

    st.success("✅ Process concluded!")
    st.markdown("**Maintenance policy performance metrics**")
    st.write({
        "Mean cost-rate": results[0],
        "Cost-rate standard deviation": results[1],
        "Mean MTBOF (mean time between failures)": results[2],
        "MTBOF standard deviation": results[3],
        "Mean availability": results[4],
        "Availability standard deviation": results[5]
    })













