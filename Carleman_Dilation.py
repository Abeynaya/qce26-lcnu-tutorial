#Generate the Carleman dilation method from Demirdjian 2025

import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt


##############################################################################
#PARAMETERS
##############################################################################

#DIRS
outdir = './'

def Carleman_Dilation_Matrix(params):
    alpha,nt,nx,dt,dx,nu = params['alpha'], params['nt'], params['nx'], params['dt'], params['dx'], params['nu']

    #F1 MATRIX
    main_diag = np.zeros(nx) - 2.0
    off_diag  = np.ones(nx)
    diagonals = [main_diag,off_diag,off_diag]
    F1        = sparse.diags(diagonals, [0, -1, 1], format="csr")
    F1        = F1.toarray()
    F1[0,nx-1] = 1.
    F1[nx-1,0] = 1.
    F1 = F1*nu/dx**2

    #F2 Matrix
    F2 = np.zeros((nx,nx**2))
    for i in range(0,nx):
        #i+1 index
        if (i<nx-1):
            F2[i,i*nx+1+i] = 1.
        else:
            F2[i,nx**2-nx] = 1.

        if (i==0):
            F2[i,nx-1] = -1.
        else:
            F2[i,i*nx-1+i] = -1.
    F2 = -F2/(2*dx)

    #BUILD THE LIU2020 SYSTEM OF EQUATIONS
    #EQUATIONS 3.1-3.4 of Liu et al. 2020
    nA = np.sum([nx**i for i in range(1,alpha+1)])
    nAe = 2*nx**alpha
    A = np.zeros((nA,nA))
    Ae = np.zeros((nAe,nAe))
    #LOOP OVER ROWS OF A
    for j in range(1,alpha+1):
        A1 = 0.
        A2 = 0.
        st = 0
        ed = 0
        for k in range(1,j+1):
            I1 = np.identity(nx**(k-1))
            I2 = np.identity(nx**(j-k))
            A1 += np.kron(np.kron(I1,F1),I2)
            A2 += np.kron(np.kron(I1,F2),I2)

            if (k>1):
                st += nx**(k-1) 
            ed += nx**(k)        

        #Save A21 padded with zeros    
        if (j==1):
            A21_pad = np.zeros((nx**2,nx**2))
            A21_pad[:nx,:] = F2

        #Create the full A matrix 
        A[st:ed,st:ed] = A1
        if (j<alpha):
            A[st:ed,ed:ed+nx**(j+1)] = A2
    #Embed the A matrix into the A_e matrix
    Ae[nAe-nA:,nAe-nA:] = A

    #Full Carleman matrix with zero padding following DH26
    Le = np.zeros((nAe*nt,nAe*nt))
    InAe   = np.identity(nAe)

    #Build the regular Carleman (L) and Carleman Dilation (Le)
    Me = InAe - dt*Ae
    for j in range(0,nt):
        if (j==0):
            #First diagonal block
            Le[j*nAe:(j+1)*nAe,j*nAe:(j+1)*nAe] = InAe
        else:
            #Diagonal blocks
            Le[j*nAe:(j+1)*nAe,j*nAe:(j+1)*nAe] = Me
            
            #Off diagonal blocks
            Le[j*nAe:(j+1)*nAe,(j-1)*nAe:j*nAe] = -InAe       
    return(Le)
