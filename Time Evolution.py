import numpy as np 
from matplotlib import pyplot as plt
import scipy.io 
import pandas as pd
from scipy.integrate import quad

def y(t, N, V, f, Cav):

    ft = np.exp(-1j*N*t)
    
    ft = np.transpose(ft)

    f = np.transpose(f) 

    C = np.linalg.inv(V)

    y = np.matmul(V , np.multiply(ft, np.matmul(C,f)))
    
    return (np.abs(y[Cav]))**2


for i in range(1, 2):

    V = scipy.io.loadmat('/Users/bryn/Documents/ENPH 455/CurvedCROW/VCurved')
    N = scipy.io.loadmat('/Users/bryn/Documents/ENPH 455/CurvedCROW/NCurved')
    
    N = N['N_nu_curved']
    V = V['V_q_nu_curved']
    #V = scipy.io.loadmat('/Users/bryn/Documents/ENPH 455/Output/PCSout'+str(i)+'/V'+str(i))
    #N = scipy.io.loadmat('/Users/bryn/Documents/ENPH 455/Output/PCSout'+str(i)+'/N'+str(i))
    # N = N['N_nu']
    # V = V['V_q_nu']
    N = np.diag(N)

    aa = 480e-9     
    alpha = 0.2
    nn = 20
    hbar = 1.054571 * 10e-34
    c = 2.99792 * 10e8
    epsilon0 = 8.854187 * 10e-12

    numCavities = len(N)
    #numCavities = len(N) - 150

    OMEGA0 = 0.307674933843196 - 0.000008151192949j
    
    omega1 = 0.307483214025719 - 0.000008046645524j

    B1 = -0.04511 

    k1 = (1/aa) * np.arccos((1/B1)*(1-np.real(omega1)/np.real(OMEGA0)))
    
    omega2 = 0.307937102281673 - 0.000008461087745j
    
    k2 = (1/aa) * np.arccos((1/B1)*(1-np.real(omega2)/np.real(OMEGA0)))
    
    q0 = 25

    f = np.zeros(numCavities, dtype = 'complex')
    f1 = np.zeros(numCavities, dtype = 'complex')
    f2 = np.zeros(numCavities, dtype = 'complex')
    # f = np.zeros(150 + numCavities, dtype = 'complex')
    # f1 = np.zeros(150 + numCavities, dtype = 'complex')
    # f2 = np.zeros(150 + numCavities, dtype = 'complex')
    
    for q in range(0,50):
    
        f1[q] = alpha * np.exp(-(alpha**2  * (q-q0)**2))  * np.exp(-1j*aa*k1*(q-q0))
    
        f2[q] = alpha * np.exp(-(alpha**2 *  (q-q0)**2)) * np.exp(-1j*aa*k2*(q-q0))
    
        f[q] = f1[q] + f2[q]

    # q = np.linspace(0, 150 + numCavities, 150 + numCavities)

    # plt.plot(q, np.abs(f))
    # plt.show()
    yplot = np.zeros(10000)
    t= np.linspace(-2000,8000,10000)

    for j in range(0,10000):

        yplot[j] = y(t[j], N,V,f, 25)

    plt.plot(t, yplot)
    plt.show()
    
    # y1, err1 = scipy.integrate.quad(y, 25000, 50000, args=(N,V,f, numCavities + 49))
    # y2, err2 = scipy.integrate.quad(y, 25000, 50000, args=(N,V,f, numCavities + 35))
    # yref, errref = scipy.integrate.quad(y, 0, 25000, args=(N,V,f, 25))
    # print(y1, y2, yref)
    