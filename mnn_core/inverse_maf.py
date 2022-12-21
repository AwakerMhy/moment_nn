from mnn_core.maf import *
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import fsolve

def eqs(X, *data):
    eq1 = maf.mean(X[0],X[1]) - data[0]
    eq2 = maf.std(X[0],X[1])[0] - data[1]
    return np.append(eq1,eq2)

def jacobian(X, *data):
    dudu, duds = maf.grad_mean(X[0],X[1])
    dsdu, dsds = maf.grad_std(X[0],X[1])    
    return np.array([[dudu,duds],[dsdu,dsds]])

#%%
if __name__=='__main__':
    maf = MomentActivation()
    
    #generate detailed contour plot for the 2d map
    n = 1000
    
    ubar = np.linspace(0.5,1.5,n)
    sbar = np.linspace(0,2,n)
    
    U = np.zeros((n,n))
    S = np.zeros((n,n))
    
    for i in range(n):
        u = maf.mean(ubar,sbar[i]*np.ones(n))
        s, _ = maf.std(ubar,sbar[i]*np.ones(n))
        U[i,:] = u
        S[i,:] = s
    
    
    plt.close('all')
    #plt.contour(ubar,sbar,U,20, antialiased = True, cmap='summer') #colors=['red'], 
    #plt.contour(ubar,sbar,S,20,   antialiased = True, cmap='spring')
    m = 15
    plt.contour(ubar,sbar,U,m, antialiased = True, colors=['orange']) 
    plt.contour(ubar,sbar,S,m, antialiased = True, colors=['gray'])
    plt.xlabel('Input mean')
    plt.ylabel('Input std')
    
    #%%
    #check pre-image error
    nn = 100
    #err2 = np.zeros((nn,nn))
    u_in = np.linspace(0.5,1.5,nn)
    s_in = np.linspace(0,2,nn)
    
    U0 = np.zeros((nn,nn))
    S0 = np.zeros((nn,nn))
    
    for i in range(nn): #vary mean
        print('Computing iteration {}/{}...'.format(i,nn))
        for j in range(nn): #vary std        
            dat = (maf.mean(u_in[i],s_in[j]), maf.std(u_in[i],s_in[j])[0])
            #without custom gradient
            #u0, s0 = fsolve(eqs, [1,1], args=dat)    
            #with custom gradient
            u0, s0 = fsolve(eqs, [1,1], args=dat, fprime = jacobian)
            
            U0[j,i] = u0
            S0[j,i] = s0
            
    #%%
    plt.close('all')
    
    err2_u = np.abs(U0-u_in.reshape(1,nn))#/np.abs(u_in.reshape(1,nn))
    err2_s = np.abs(S0-s_in.reshape(nn,1))#/np.abs(s_in.reshape(nn,1))
    
    #plt.subplot(1,2,1)
    plt.imshow(np.log10(err2_s+err2_u+1e-16), extent = [u_in[0],u_in[-1], s_in[0],s_in[-1]], aspect='auto', origin='lower')
    #plt.subplot(1,2,2)
    #plt.imshow(np.log10(err2_s+1e-16), extent = [u_in[0],u_in[-1], s_in[0],s_in[-1]], aspect='auto', origin='lower')
    
    #plt.imshow(np.log10(err), origin='lower', extent = [u_tar[0],u_tar[-1], s_tar[0],s_tar[-1]], aspect='auto')
    plt.colorbar()
    plt.title('Root finding error (log10)')
    plt.xlabel('Input current mean', fontsize = 12)
    plt.ylabel('Input current std', fontsize = 12)
    
    

 #%%
    # #systematically check the magnitude of error
    # #first, check image error
    # nn = 100
    # err = np.zeros((nn,nn))
    # u_tar = np.linspace(0.01,0.15,nn)
    # s_tar = np.linspace(0.01,0.4,nn)
    
    # for i in range(nn): #vary mean
    #     print('Computing iteration {}/{}...'.format(i,nn))
    #     for j in range(nn): #vary std        
    #         dat = (u_tar[i], s_tar[j])
    #         u0, s0 = fsolve(eqs, [1,1], args=dat)    
    #         err[j,i] = np.max(np.abs(eqs((u0,s0), *dat))/np.array(dat))
       
    # #print('Root found to be: ',(u0,s0))
    # #print('Error:', eqs((u0,s0), *dat) )
    # plt.close('all')
    # plt.imshow(np.log10(err), origin='lower', extent = [u_tar[0],u_tar[-1], s_tar[0],s_tar[-1]], aspect='auto')
    # plt.colorbar()
    # plt.title('Rel. error of root finding (log10)')
    # plt.xlabel('Mean firing rate (sp/ms)')
    # plt.ylabel('Firing variability (sp/ms^(1/2))')
    