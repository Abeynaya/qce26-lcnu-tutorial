"""
Functions to create circuits fot the A_l matrices following
Demirdjian et al. (2025) "An Efficient Decomposition of the Carleman Linearized Burgers’ Equation"

These circuits are controlled to an ancilla for the hadamard test.
Note that the script "Bitstring_Generation_Lejj_nocontrol" is that same as this one
execpt it uses one less qubit since there is no control for the hadamard test.
"""

import numpy as np
import math
from qiskit import QuantumCircuit, transpile
from namelist import nx,Length,x,dx,nu,dt,tau,nt,nqubit,b_init
import Carleman_Dilation
from qiskit_aer import Aer, AerSimulator


##############################################################################
#Functions
##############################################################################

#Decimal -> binary bitstring
def dec2bin(x,tau):
    y = [int(d) for d in str(bin(x))[2:]]
    num = int(math.log2(tau)) - len(y)
    for i in range(0,num):
        y.insert(0,0)
    #map 1's to 3's
    for i,y0 in enumerate(y):
        if y0 == 1:
            y[i] = 3
    return(y)


#Decimal -> quaternary bitstring mapping function for A_j^j terms
def dec2quat_jj(x,tau):
    y = [int(d) for d in str(bin(x))[2:]]
    num = int(math.log2(tau)) - len(y)
    for i in range(0,num):
        y.insert(0,0)
    #map 1's to 3's
    for i,y0 in enumerate(y):
        if y0 == 1:
            y[i] = 3
    return(y)


#Decimal -> quaternary bitstring mapping function for A_{j+1}^j terms
def dec2quat_jp1j(x,tau):
    sigstr = []
    for q in range(0,int(np.log2(tau))):
        dq = []
        if (q==0):
            if (x%2==0):
                dq = 1
            else:
                dq = 2
        elif (q==1):
            if (x%4==0):
                dq = 0
            elif ((x+3)%4==0):
                dq = 1
            elif ((x+2)%4==0):
                dq = 3
            elif ((x+1)%4==0):
                dq = 2
        else:
            if ((x+1)%2**(q+1) in range(1,2**q)):
                dq = 0
            elif ((x+1)%2**(q+1) in range(2**q+1,2**(q+1))):
                dq = 3
            elif ((x+1)%2**(q+1) == 2**q):
                dq = 1
            elif ((x+1)%2**(q+1) == 0):
                dq = 2
        #Insert in reverse order
        sigstr.insert(0,dq)
        del[dq]
    return(sigstr)


#Quinary bitstring for L_1^(e) terms
def L1e_quinary_str(nx,tau,nt):
    #The values of the quinary bitstring represent the following:
    #0=\sigma_+\sigma_-, 1=\sigma_-\sigma_+, 2=\sigma_+, 3=\sigma_-, 4=I_2   

    #Static list
    tail = [4]*int(math.log2(tau*nx**tau)) #I_2^{\otimes log(\tau n_x^\tau)}

    #First two terms
    L1e = []
    L1e_coeff = []
    L1e.append([4]*int(math.log2(nt)) + tail)            #I_2^{int(math.log2(nt))} \otimes tail
    L1e_coeff.append(1.)
    L1e.append([4]*(int(math.log2(nt))-1) + [2] + tail)  #I_2^{int(math.log2(nt))-1}\otimes \sigm \otimes tail
    L1e_coeff.append(-1.)

    #All other terms
    for j in range(2,int(math.log2(nt))+1):
        L1e.append([4]*(j-2) + [2] + [1]*(int(math.log2(nt))-j+1) + tail) #I_2^{j-2} \otimes \sigm \otimes \sigp^{log(nt)-j+1}
        L1e_coeff.append(-1.)
    return(L1e,L1e_coeff)


#Quinary bitstring for L_j^{(e),j} terms
def Lejj_quinary_str(nx,tau,nt,nu,dt,dx):
    #The values of the quinary bitstring represent the following:
    #0=\sigma_+\sigma_-, 1=\sigma_-\sigma_+, 2=\sigma_+, 3=\sigma_-, 4=I_2   

    #Static lists
    t1 = [4]*int(int(math.log2(nt))) #I_2^{\otimes int(math.log2(nt))}
    t2 = [0]*int(int(math.log2(nt))) #(sig_+sig_-)^{\otimes int(math.log2(nt))}
    t1t2 = []
    t1t2.append(t1)
    t1t2.append(t2)

    #A11 terms
    A11 = []
    A11_coeff = []
    A11.append([4]*int(math.log2(nx)))            #I_2^{\otimes int(math.log2(nx))}
    A11_coeff.append([-2.])
    A11.append([4]*(int(math.log2(nx))-1)+[1])    #I_2^{\otimes int(math.log2(nx))-1}\otimes\sigma_+
    A11_coeff.append([1.])
    A11.append([4]*(int(math.log2(nx))-1)+[2])    #I_2^{\otimes int(math.log2(nx))-1}\otimes\sigma_-
    A11_coeff.append([1.])
    A11.append([1]*int(math.log2(nx)))            #\sigma_+^{\otimes int(math.log2(nx))}
    A11_coeff.append([1.])
    A11.append([2]*int(math.log2(nx)))            #\sigma_-^{\otimes int(math.log2(nx))}
    A11_coeff.append([1.])
    for m in range(2,int(math.log2(nx))+1):
        A11.append([4]*(m-2)+[2]+[1]*(int(math.log2(nx))-m+1))   #I_2^{\otimes m-2}\otimes\sigma_-\otimes\sigma_+^{\otimes int(math.log2(nx))-m+1}
        A11_coeff.append([1.])
        A11.append([4]*(m-2)+[1]+[2]*(int(math.log2(nx))-m+1))   #I_2^{\otimes m-2}\otimes\sigma_+\otimes\sigma_-^{\otimes int(math.log2(nx))-m+1}
        A11_coeff.append([1.])

    Lejj = []
    Lejj_coeff = []
    for t in range(0,2):
        for j in range(1,tau+1):
            qbs1 = []
            qbs1 += t1t2[t]
            qbs1 += dec2bin(j-1,tau)   #sig_{b(j-1)}
            qbs1 += [0]*int(int(math.log2(nx))*(tau-j)) #(sig_+sig_-)^{\otimes n_x^{tau-j}}
            for l in range(0,j):
                qbs2 = []
                qbs2 += qbs1
                qbs2 += [4]*(int(math.log2(nx))*l)       #I_2^{\otimes int(math.log2(nx))*l}
                for m in range(0,len(A11)):
                    qbs3 = []
                    qbs3 += qbs2
                    qbs3 += A11[m]           #The A_1^1 terms
                    qbs3 += [4]*(int(math.log2(nx))*(j-l-1))  #I_2^{\otimes int(math.log2(nx))*(j-l-1)}
                    #qbs3 = qbs3[::-1]   #Reverse the order so that Lejj[i][j] corresponds to the jth qubit
                    Lejj.append(qbs3)
                    Lejj_coeff.append(-1. * (-1.)**t * A11_coeff[m][0] * dt*nu/dx**2)
    return(Lejj,Lejj_coeff)
   

#Binary bitstring (bbs) for \bar{Le}
def binary_str(Le):
    #The values of the binary bitstring represent the following:
    #0=Pauli-X, 1=I_2
    #The mapping follows from Gnanasekaran and Surana (2024) Theorem 2.

    Le_bar = []
    for i in range(0,len(Le)):
        bbs = []
        for j in range(0,len(Le[i])):
            if (Le[i][j] in [0,3,4]):
                bbs += [1]
            elif (Le[i][j] in [1,2]):
                bbs += [0]
        Le_bar.append(bbs)
    return (Le_bar)

    
#Tertiary bitstring (tbs) for matmul(Le,Le^T)
def tertiary_str(Le):
    #The values of the tertiary bitstring represent the following:
    #0=open control, 1=closed control, 2=no control

    #The mapping follows from the proof of Theorem 3 in GS24 combined
    #with the following:
    #(\sigma_+\sigma_-)(\sigma_+\sigma_-)^T = (\sigma_+\sigma_-)
    #(\sigma_-\sigma_+)(\sigma_-\sigma_+)^T = (\sigma_-\sigma_+)
    #(\sigma_+)(\sigma_+)^T = (\sigma_+\sigma_-)
    #(\sigma_-)(\sigma_-)^T = (\sigma_-\sigma_+)
    #(I_2)(I_2)^T = I_2

    Le_LeT = []
    for i in range(0,len(Le)):
        tbs = []
        for j in range(0,len(Le[i])):
            if (Le[i][j] in [0,1]):
                tbs += [0]
            elif (Le[i][j] in [2,3]):
                tbs += [1]
            elif (Le[i][j] in [4]):
                tbs += [2]
        Le_LeT.append(tbs)
    return(Le_LeT)


#The U1 multi control not 
def U1(qc,qbs):
    #U1
    #Get bitstring for AA^T
    qbs_qbsT = tertiary_str([qbs])[0]
    len_qbs = len(qbs)

    Xlist = []
    Clist = []
    Clist.append(nqubit)
    for r in range(0,len_qbs):
        if (qbs_qbsT[r] == 0):
            Xlist.append(nqubit-r-2)
            Clist.append(nqubit-r-2)
        elif (qbs_qbsT[r] == 1):
            Clist.append(nqubit-r-2)
    if (Xlist):
        for k in range(0,len(Xlist)):
            qc.x(Xlist[k])
    if (Clist):
        qc.mcx(Clist,nqubit-1,mode='noancilla') #Target is on the 2nd ancilla
    else:
        qc.cx(nqubit,nqubit-1) #For the all identity case
    if (Xlist):
        for k in range(0,len(Xlist)):
            qc.x(Xlist[k])
    return(qc)


#U_1 and U_2 circuits for pure sigma terms
def U1U2_circ_pure_sigma(Le_bar,Le_LeT,nqubit):
    #Note that bitstrings are written in tensor product order s.t 
    #b1,b2,...,bn = sig_{b1} \otimes sig_{b2} \otimes ... \otimes sig_{bn}
    #so the order must be reversed for the circuits
    circs = []
    #Quantum Circuits for U_2=I \otimes \bar{A}
    for i in range(0,len(Le_bar)):
        qc = QuantumCircuit(nqubit+1)
        for j in range(0,len(Le_bar[i])):
            if (Le_bar[i][nqubit-2-j] == 0):
                qc.cx(nqubit,j)
        #qc.barrier(list(range(0,nqubit)),label='U_2(Lejj)')
        circs.append(qc)
        del(qc)

    #Quantum Circuits for U_1
    for i in range(0,len(Le_LeT)):
        Xlist = [] 
        Clist = []
        Clist.append(nqubit) #Had test
        for j in range(0,len(Le_LeT[i])):
            if (Le_LeT[i][nqubit-2-j] == 0):
                Xlist.append(j)
                Clist.append(j)
            elif (Le_LeT[i][nqubit-2-j] == 1):
                Clist.append(j)
        if (Xlist):
            for k in range(0,len(Xlist)):
                circs[i].cx(nqubit,Xlist[k])
        if (Clist):
            circs[i].mcx(Clist,nqubit-1,mode='noancilla') #Target is the ancilla
        else:
            circs[i].cx(nqubit,nqubit-1) #For the all identity case
        if (Xlist):
            for k in range(0,len(Xlist)):
                circs[i].cx(nqubit,Xlist[k])
        #circs[i].barrier(list(range(0,nqubit)),label='U_1(Lejj)')
        #circs[i].save_unitary()   #Use save_unitary in order to print the matrix later
    return(circs)


#U_1 and U_2 circuits for L_1^{(e)} = (I_{n_t} - S_{-1} + \rho_1^{\log n_t}) \ otimes I
#where S_{-1} is the decrementer
def U1U2_circ_L1e(circs):
    t1 = [4]*int(math.log2(tau*nt*nx**tau))
    t2 = []
    t3 = [1]*int(math.log2(nt)) + [4]*int(math.log2(tau*nx**tau))
    ts = [t1,t2,t3]
    L1e_coeff = [1.,-1.,1.]

    for t in range(0,3):
        qbs = []
        qbs += ts[t]
        qbs_bar = binary_str([qbs])[0] #Complement for the sigma terms

        qc = QuantumCircuit(nqubit+1)
        #U2 for the sigma terms 
        for k in range(0,len(qbs_bar)):
            if (qbs_bar[k] == 0):
                qc.cx(nqubit,nqubit-2-k)

        if (t == 1): 
            Decrementer(int(math.log2(nt)),qc,int(math.log2(tau*nx**tau)))

        #U1 circuit
        qc = U1(qc,qbs)
        circs.append(qc)

        del([qbs,qbs_bar])
    return(circs,L1e_coeff)


#U_1 and U_2 circuits for L_j^{(e),j} terms that are not pure sigma
def U1U2_circ_Lejj(circs):
    #Term 1: Pure sigma terms
    #Static lists
    t1 = [4]*int(int(math.log2(nt))) #I_2^{\otimes int(math.log2(nt))}
    t2 = [0]*int(int(math.log2(nt))) #(sig_+sig_-)^{\otimes int(math.log2(nt))}
    t1t2 = []
    t1t2.append(t1)
    t1t2.append(t2)

    Lejj_coeff = []
    for t in range(0,2):
        for j in range(1,tau+1):
            qbs = []
            qbs += t1t2[t]
            qbs += dec2quat_jj(j-1,tau) #sig_{q(j)}
            qbs += [0]*int(int(math.log2(nx))*(tau-j)) #(sig_+sig_-)^{\otimes n_x^{tau-j}}
            qbs_bar = binary_str([qbs])[0] #Complement for the sigma terms

            for l in range(0,j):
                for pm in range(0,3):
                    qc = QuantumCircuit(nqubit+1)
                    #U2 for the sigma terms 
                    for k in range(0,len(qbs_bar)):
                        if (qbs_bar[k] == 0):
                            qc.cx(nqubit,nqubit-2-k)
                    #qc.barrier(list(range(0,nqubit)),label='U_2(sigs)')
                                        
                    #U2 for I_{nx}^l \otimes P \otimes I_{nx}{j-l-1}
                    #F_1 = Incrementer + -2*I + Decrementer
                    #pm==0 is incrementer, pm==1 is I, pm==2 is decrementer
                    #The pm order above is chosen to make coefficient calculation simpler
                    if (pm == 0):
                        Incrementer(int(math.log2(nx)),qc,int(math.log2(nx))*(j-l-1))
                    if (pm == 2): 
                        Decrementer(int(math.log2(nx)),qc,int(math.log2(nx))*(j-l-1))

                    #U1
                    #Get bitstring for AA^T
                    qbs_qbsT = tertiary_str([qbs])[0]
                    len_qbs = len(qbs)

                    Xlist = []
                    Clist = []
                    Clist.append(nqubit)
                    for r in range(0,len_qbs):
                        if (qbs_qbsT[r] == 0):
                            Xlist.append(nqubit-r-2)
                            Clist.append(nqubit-r-2)
                        elif (qbs_qbsT[r] == 1):
                            Clist.append(nqubit-r-2)
                    if (Xlist):
                        for k in range(0,len(Xlist)):
                            qc.x(Xlist[k])
                    if (Clist):
                        qc.mcx(Clist,nqubit-1,mode='noancilla') #Target is on the 2nd ancilla
                    else:
                        qc.cx(nqubit,nqubit-1) #For the all identity case
                    if (Xlist):
                        for k in range(0,len(Xlist)):
                            qc.x(Xlist[k])
                    #qc.barrier(list(range(0,nqubit)),label='U_1(Lejp1j)')
                    #qc.save_unitary()   #Use save_unitary in order to print the matrix later                    
                    circs.append(qc)
                    
                    #Coefficients
                    factor = 2 if pm==1 else 1
                    Lejj_coeff.append(-1. * (-1)**t * (-1)**pm * factor * dt*nu/dx**2)
                    del([qc,Xlist,Clist,qbs_qbsT])          
        
            del([qbs,qbs_bar])
    return(circs,Lejj_coeff)


#U_1 and U_2 circuits for L_{j+1}^{(e),j} terms
def U1U2_circ_Lejp1j(circs):
    #Term 1: Pure sigma terms
    #Static lists
    t1 = [4]*int(int(math.log2(nt))) #I_2^{\otimes int(math.log2(nt))}
    t2 = [0]*int(int(math.log2(nt))) #(sig_+sig_-)^{\otimes int(math.log2(nt))}
    t1t2 = []
    t1t2.append(t1)
    t1t2.append(t2)

    Lejp1j_coeff = []
    for t in range(0,2):
        for j in range(1,tau):
            qbs = []
            qbs += t1t2[t]
            qbs += dec2quat_jp1j(j-1,tau) #sig_{q(j)}
            qbs += [0]*int(int(math.log2(nx))*(tau-j-1)) #(sig_+sig_-)^{\otimes n_x^{tau-j-1}}
            qbs_bar = binary_str([qbs])[0] #Complement for the sigma terms

            for l in range(0,j):
                for pm in range(0,2):
                    qc = QuantumCircuit(nqubit+1)
                    #U2 for the sigma terms 
                    for k in range(0,len(qbs_bar)):
                        if (qbs_bar[k] == 0):
                            qc.cx(nqubit,nqubit-2-k)
                    #qc.barrier(list(range(0,nqubit)),label='U_2(sigs)')

                    #U2 for K^{n_x^2,n_x^l}
                    commutation_circ(2*int(math.log2(nx)),l*int(math.log2(nx)),qc,int(math.log2(nx))*(j-l-1))
                    #qc.barrier(list(range(0,nqubit)),label='U_2(K^{n_x^2,n_x^l})')
                                        
                    #U2 for D1*P \otimes I^{int(math.log2(nx))*l}
                    if (pm == 0):
                        Pmatrix_plus(int(math.log2(nx)),qc,int(math.log2(nx))*(j-1))
                    elif (pm == 1):
                        Pmatrix_minus(int(math.log2(nx)),qc,int(math.log2(nx))*(j-1))

                    #U2 for sigpm^{\otimes int(math.log2(nx))} \otimes K^{n_x^l,n_x}
                    commutation_circ(int(math.log2(nx))*l,int(math.log2(nx)),qc,int(math.log2(nx))*(j-l-1))
                    #qc.barrier(list(range(0,nqubit)),label='U_2(K^{n_x^l,n_x})')
                                            
                    #U1
                    #Get bitstring for AA^T
                    qbs_qbsT = tertiary_str([qbs])[0]
                    qbs_qbsT += [0]*int(math.log2(nx)) + [2]*j*int(math.log2(nx))
                    #qbs_qbsT += [2]*int(math.log2(nx)) + [2]*j*int(math.log2(nx))   ####FLAG: for T3 testing
                    len_qbs = len(qbs_qbsT)
                    
                    Xlist = []
                    Clist = []
                    Clist.append(nqubit)
                    for r in range(0,len_qbs):
                        if (qbs_qbsT[r] == 0):
                            Xlist.append(nqubit-r-2)
                            Clist.append(nqubit-r-2)
                        elif (qbs_qbsT[r] == 1):
                            Clist.append(nqubit-r-2)
                    if (Xlist):
                        for k in range(0,len(Xlist)):
                            qc.x(Xlist[k])
                    if (Clist):
                        qc.mcx(Clist,nqubit-1,mode='noancilla') #Target is on the 2nd ancilla
                    else:
                        qc.cx(nqubit,nqubit-1) #For the all identity case
                    if (Xlist):
                        for k in range(0,len(Xlist)):
                            qc.x(Xlist[k])
                    #qc.barrier(list(range(0,nqubit)),label='U_1(Lejp1j)')
                    #qc.save_unitary()   #Use save_unitary in order to print the matrix later                    
                    circs.append(qc)  
                    
                    #Coefficients
                    Lejp1j_coeff.append(-1. * -1. * (-1)**t * (-1)**pm * dt/(2.*dx))
                    del([qc,Xlist,Clist,qbs_qbsT])
                    
            del([qbs,qbs_bar])
    return(circs,Lejp1j_coeff)


#Basic quantum adder ####FLAG: The depth can be improved
def Incrementer(nq,qc,offset):
    qc.cx(nqubit,0+offset)
    qc.mcx([nqubit,0+offset],1+offset)
    
    for q in range(nq-3,-1,-1):
        controls = list(range(0+offset,nq-q-1+offset))
        controls.append(nqubit)        
        qc.mcx(controls,nq-q-1+offset,mode='noancilla')
        del(controls)


#Basic quantum subtractor ####FLAG: The depth can be improved
def Decrementer(nq,qc,offset):
    for q in range(0,nq-2):
        controls = list(range(0+offset,nq-q-1+offset))
        controls.append(nqubit)
        qc.mcx(controls,nq-q-1+offset,mode='noancilla')
        del(controls)
    #qc.barrier(list(range(0,nqubit)),label='U_2(P_2^+)')
    if (nq>1):
        qc.mcx([nqubit,0+offset],1+offset)
    qc.cx(nqubit,0+offset)


#Circuit for commutation matrix K^{2**m,2**n}
def commutation_circ(m,n,qc,offset):
    for r in range(n-1,-1,-1):
        for q in range(m-1,-1,-1):
            qc = cswap(qc,nqubit,n-r+q-1+offset,n-r+q+offset)
    return(qc)


#Circuit for P^+ matrix 
def Pmatrix_minus(nq,qc,offset):
    #P_2^-: Decrementer circuit
    for q in range(0,nq-2):
        controls = list(range(0+offset,nq-q-1+offset))
        controls.append(nqubit)
        qc.mcx(controls,nq-q-1+offset,mode='noancilla')
        del(controls)
    #qc.barrier(list(range(0,nqubit)),label='U_2(P_2^+)')
    qc.mcx([nqubit,0+offset],1+offset)
    qc.cx(nqubit,0+offset)
    
    #P_1: Modular multiplication by n_x+1
    for q in range(0,nq):
        qc.mcx([nqubit,nq-q-1+offset],2*nq-q-1+offset)
    #qc.barrier(list(range(0,nqubit)),label='U_2(P_1)')
   

#Circuit for P^+ matrix
def Pmatrix_plus(nq,qc,offset):
    #P_2^+: Incrementer circuit
    qc.cx(nqubit,0+offset)
    qc.mcx([nqubit,0+offset],1+offset)
    for q in range(nq-3,-1,-1):
        controls = list(range(0+offset,nq-q-1+offset))
        controls.append(nqubit)        
        qc.mcx(controls,nq-q-1+offset,mode='noancilla')
        del(controls)
    #qc.barrier(list(range(0,nqubit)),label='U_2(P_2^+)')
    
    #P_1: Modular multiplication by n_x+1
    for q in range(0,nq):
        qc.mcx([nqubit,nq-q-1+offset],2*nq-q-1+offset)
    #qc.barrier(list(range(0,nqubit)),label='U_2(P_1)')
    
    
#Validate pure sigma circuits
def validate_sig(Le,circs):
    #Remake the A matrix
    for i in range(0,len(Le)):
        #Run the circuit and extract the unitary matrix
        simulator = Aer.get_backend('aer_simulator')      # Transpile for simulator
        # simulator = AerSimulator(method = 'unitary')    # Another option to create the simulator
        circ = transpile(circs[i], simulator)
        result = simulator.run(circ).result()             # Run
        unitary = result.get_unitary(circ)                # Get unitary
        U_circ = np.asarray(unitary)      
        A_circ = coeffs[i] * U_circ[:int(2**(nqubit-1)),int(2**(nqubit-1)):]
        
        A_real = bitstr2matrix(Le[i])
        print(np.max(np.abs(A_real-A_circ)))
        del([A_real,A_circ,circ,U_circ1,U_circ2])


#Validate Lejp1j circuits
def validate_Lejp1j(circs):
    #Term 1: Pure sigma terms 
    #Static lists
    t1 = [4]*int(int(math.log2(nt))) #I_2^{\otimes int(math.log2(nt))}
    t2 = [0]*int(int(math.log2(nt))) #(sig_+sig_-)^{\otimes int(math.log2(nt))}
    t1t2 = []
    t1t2.append(t1)
    t1t2.append(t2)
    temp_sig = np.zeros((nx,nx))
    temp_sig[0,0] = 1.

    m = -1
    for t in range(0,2):
        for j in range(1,tau):
            qbs = []
            qbs += t1t2[t]
            qbs += dec2quat_jp1j(j-1,tau) #sig_{q(j)}
            qbs += [0]*int(int(math.log2(nx))*(tau-j-1)) #(sig_+sig_-)^{\otimes n_x^{tau-j-1}}
            qbs_bar = binary_str([qbs])[0] #Complement for the sigma terms
            
            #Get the matrix from bitstring
            mat1 = bitstr2matrix(qbs)

            for l in range(0,j):               
                for pm in range(0,2):
                    m += 1

                    #U2 for sigpm^{\otimes int(math.log2(nx))} \otimes K^{n_x^l,n_x}
                    T1 = np.kron(temp_sig,comm_mat(int(math.log2(nx))*l,int(math.log2(nx))))
                    
                    #U2 for D1*P \otimes I^{int(math.log2(nx))*l}
                    if (pm == 0):
                        T2 = np.kron(Ap(nx),np.identity(2**(l*int(math.log2(nx)))))
                    elif (pm == 1):
                        T2 = np.kron(Am(nx),np.identity(2**(l*int(math.log2(nx)))))
                        
                    #U2 for K^{n_x^2,n_x^l}
                    T3 = comm_mat(2*int(math.log2(nx)),l*int(math.log2(nx)))
                    
                    #Lejp1j matrix
                    mat2 = np.kron(np.kron(mat1,T1@T2@T3),np.identity(2**(int(math.log2(nx))*(j-l-1))))                   
                    #mat2 = np.kron(np.kron(mat1,T1),np.identity(2**(int(math.log2(nx))*(j-l-1))))   ####FLAG: testing T1
                    #mat2 = np.kron(np.kron(mat1,T2),np.identity(2**(int(math.log2(nx))*(j-l-1))))   ####FLAG: testing T2
                    #mat2 = np.kron(np.kron(mat1,T3),np.identity(2**(int(math.log2(nx))*(j-l-1))))   ####FLAG: testing T3                    
                    #np.savetxt('matrix.txt',mat2,fmt='%.0f')

                       
                    #Difference between matrix and circuit
                    #Run the circuit and extract the unitary matrix
                    #print(circs[m])
                    simulator = Aer.get_backend('aer_simulator')      # Transpile for simulator
                    circ = transpile(circs[m], simulator)
                    result = simulator.run(circ).result()             # Run
                    unitary = result.get_unitary(circ)                # Get unitary
                    U_circ = np.asarray(unitary)       
                    A_circ = coeffs[i] * U_circ[:int(2**(nqubit-1)),int(2**(nqubit-1)):]
                    print(np.max(np.abs(mat2-A_circ)))
                    #print('')
                    
                    del([circ,unitary,result])
                    del([T1,T2,T3,mat2,U_circ])

            del([qbs,qbs_bar,mat1])
    return(circs,Lejp1j_coeff)


#Add all matrices together to ensure that they exactly match the Carleman dilated matrix
def validate_CarlemanDilated_Matrix(circs,coeffs,tau,nt,nx,dt,dx,nu):
    A_circ = 0.
    for i in range(len(circs)):
        print('Working on', i)
        #Run the circuit and extract the unitary matrix
        simulator = Aer.get_backend('aer_simulator')      # Transpile for simulator
        circ = transpile(circs[i], simulator)
        circ.save_unitary()                               # Must save unitary to "get_unitary" later
        result = simulator.run(circ).result()             # Run
        unitary = result.get_unitary(circ)                # Get unitary
        U_circ = np.asarray(unitary)
        A_circ += coeffs[i] * U_circ[:int(2**(nqubit-1)),int(2**(nqubit-1)):]

        del([circ,unitary,result,U_circ])

    #Carleman dilated matrix
    L, Le = Carleman_Dilation.Carleman_Dilation_Matrix(tau,nt,nx,dt,dx,nu)
    
    #Check Matrices against eachother
    if (np.max(np.abs(A_circ-Le)) < 1e10):
        print('Successful validation')
    else:
        print('Unsuccessful validation')


#Use a quainary bitstring to form a matrix
def bitstr2matrix(qbs):
    #Make the real A matrix for verification
    mat = 1.
    for j in range(0,len(qbs)):
         if (qbs[j] == 0):
            mat = np.kron(mat,sigpm)
         elif (qbs[j] == 1):
            mat = np.kron(mat,sigp)
         elif (qbs[j] == 2):
            mat = np.kron(mat,sigm)
         elif (qbs[j] == 3):
            mat = np.kron(mat,sigmp)
         elif (qbs[j] == 4):
            mat = np.kron(mat,I2)
    return(mat)


#Commutation matrix K^{2**m,2**n}
def comm_mat(m, n):
    a = 2**m
    b = 2**n
    # determine permutation applied by K
    w = np.arange(a * b).reshape((a, b), order="F").T.ravel(order="F")

    # apply this permutation to the rows (i.e. to each column) of identity matrix and return result
    return np.eye(a * b)[w, :]


#A^- matrix 
def Am(nx):
    Ap = np.zeros((nx**2,nx**2))
    for i in range(0,nx):
        if (i==0):
            Ap[i,nx-1] = 1.
        else:
            Ap[i,i*nx-1+i] = 1.
    return(Ap)


#A^+ matrix:
def Ap(nx):
    Am = np.zeros((nx**2,nx**2))
    for i in range(0,nx):
        #i+1 index
        if (i<nx-1):
            Am[i,i*nx+1+i] = 1.
        else:
            Am[i,nx**2-nx] = 1.
    return(Am)


#Swap gate using CNOT gates. Something wrong with qiskits swap
def swap(qc,q1,q2):
    qc.cx(q1,q2)
    qc.cx(q2,q1)
    qc.cx(q1,q2)
    return(qc)


#Swap gate using CNOT gates. Something wrong with qiskits swap
def cswap(qc,q1,q2,q3):
    qc.mcx([q1,q3],q2)
    qc.mcx([q2,q3],q1)
    qc.mcx([q1,q3],q2)
    return(qc)


#Print full matrix
def print_matrix(a):
    for row in a:
         for elem in row:
             print(elem, end=",")
         print()                           
    return()
    

#Cost function: delta_{ll'}^q
def cost_delta(Al,Alp,xt,Ub,q):
    #Al = A_l, Alp=A_l',xt=x(theta), Ub=U_b
    
    qc_l = Al.control(1)
    qc_lp = Alp.control(1)
    
    circ = QuantumCircuit(nqubit+1,2)
    circ.h(nqubit)
    circ.compose(xt, list(range(0,nqubit-1)), inplace=True)
    circ.compose(qc_l, [nqubit] + list(range(0,nqubit)), inplace=True)
    circ.compose(Ub.inverse(), list(range(0,nqubit-1)), inplace=True)
    circ.h(q) #CCZ
    circ.mcx([nqubit,nqubit-1],q,mode='noancilla') #CCZ
    circ.h(q) #CCZ
    circ.compose(Ub, list(range(0,nqubit-1)), inplace=True)
    circ.x(nqubit) #open control
    circ.compose(qc_lp, [nqubit] + list(range(0,nqubit)), inplace=True)
    circ.x(nqubit) #open control    
    circ.h(nqubit)
    circ.measure(nqubit-1,0)
    circ.measure(nqubit,1)    
    return(circ)


#Cost function: beta_{ll'}
def cost_beta(Al,Alp,xt):
    #Al = A_l, Alp=A_l',xt=x(theta)
    
    qc_l = Al.control(1)
    qc_lp = Alp.control(1)
    
    circ = QuantumCircuit(nqubit+1,2)
    circ.h(nqubit)
    circ.compose(xt, list(range(0,nqubit-1)), inplace=True)
    circ.compose(qc_l, [nqubit] + list(range(0,nqubit)), inplace=True)
    circ.x(nqubit) #open control
    circ.compose(qc_lp, [nqubit] + list(range(0,nqubit)), inplace=True)
    circ.x(nqubit) #open control     
    circ.h(nqubit)
    circ.measure(nqubit-1,0)
    circ.measure(nqubit,1)    
    return(circ)


#Create all parameterized circuits for VQLS
def create_all_circs(nx,tau,nt,nlayer,nqubit,ltest_VQLS,b_init,nu,dt,dx):
    
    #Create the block encoded Carleman dilated matrices    
    L1e,L1e_coeff = L1e_quinary_str(nx,tau,nt)     #Create the quinary bitstring for L1e terms
    Lejj,Lejj_coeff = Lejj_quinary_str(nx,tau,nt,nu,dt,dx)     #Create the quinary bitstring for Lejj terms
    Le = L1e + Lejj     #Combine quinary bitstring
    Le_bar = binary_str(Le)                #quinary to binary bitstring mapping
    Le_LeT = tertiary_str(Le)              #quinary to tertiary bitstring mapping
    circs1 = U1U2_circ_pure_sigma(Le_bar,Le_LeT,nqubit)
    circs2 = []
    circs2,Lejp1j_coeff = U1U2_circ_Lejp1j(circs2)

    #Combine all circuits and coefficients    
    circs = circs1 + circs2
    coeffs = L1e_coeff + Lejj_coeff + Lejp1j_coeff
  
    if (ltest_VQLS):
        #Create identity in two parts: I=circs[0]+circs[1]
        del([circs,coeffs])
        circs = []        
        qc = QuantumCircuit(nqubit)
        qc.x(nqubit-2)
        qc.cx(nqubit-2,nqubit-1)
        qc.x(nqubit-2)         
        circs.append(qc)
        del(qc)
        qc = QuantumCircuit(nqubit)        
        qc.cx(nqubit-2,nqubit-1)    
        circs.append(qc)  
        coeffs = [1.,1.]

    start_time = time.process_time()    
    #Ansatz x(theta)
    #xt,params_ob = Ansatz_Sim9_Modified()

    #b-vector
    y0 = b_prep(b_init,nx,tau)
    Ub = initial_condition_circ(y0)

    if (ltest_VQLS):
        Ub = QuantumCircuit(nqubit-1)
         
    circs_beta = []
    circs_delta = []
    coeffs_sqrd_beta = []
    coeffs_sqrd_delta = []
    for l in range(0,len(circs)):
        for lp in range(0,len(circs)):
            # ####FLAG: This seems like it does worse compared to non-symmetry case
            # #Account for the symmetry: beta_{llq}=beta_{lql}
            # #All circuits have a symmetric pair except when lq=l
            # if (lp >= l):
                # circs_beta.append(cost_beta(circs[l],circs[lp],xt))
                # if (lp==l):
                    # factor = 1.
                # else: 
                    # factor = 2.
                # coeffs_sqrd_beta.append(factor*coeffs[l]*coeffs[lp])
                
            #Non-symmetry case
            ####FLAG: I am getting some negative cost function evaluations
            #         does this indicate that something is wrong?
            circs_beta.append(cost_beta(circs[l],circs[lp],xt))
            coeffs_sqrd_beta.append(coeffs[l]*coeffs[lp])
            for q in range(0,nqubit-1):
                circs_delta.append(cost_delta(circs[l],circs[lp],xt,Ub,q))
                coeffs_sqrd_delta.append(coeffs[l]*coeffs[lp])
        
    # circs_dic = {}
    # circs_dic['circs_beta'] = circs_beta
    # circs_dic['circs_delta'] = circs_delta
    # circs_dic['coeffs_sqrd_beta'] = coeffs_sqrd_beta
    # circs_dic['coeffs_sqrd_delta'] = coeffs_sqrd_delta
    # circs_dic['params_ob'] = params_ob    
    return(circs_beta,circs_delta,coeffs_sqrd_beta,coeffs_sqrd_delta)

   
#Ansatz circuit. Modified version of circuit 9 from Sims et al. (2019)
#Same one used in my 2022 paper
def Ansatz_Sim9_Modified():    
    circ = QuantumCircuit(nqubit-1)
    params_ob = ParameterVector('params',nparam)
    m = -1
    for ly in range(0,nlayer):
        for iz in range(0,nqubit-1):
            circ.h(iz)
        for iz in range(0,nqubit-2):
            circ.cz (iz,iz+1)
        for iz in range(0,nqubit-1):
            m += 1
            circ.ry(params_ob[m],iz)
    return(circ,params_ob)


#Create a circuit for the initial condition
def initial_condition_circ(y0):
    qc0 = QuantumCircuit(nqubit-1)   
    #qc0.initialize(y0) #Cannot use the .inverse() method
    qc0.prepare_state(y0) #Can use the .inverse() method
    qc1 = qc0.decompose().decompose().decompose().decompose().decompose()
    qc2 = QuantumCircuit(nqubit+1)
    qc2.compose(qc1, list(range(0,nqubit-1)), inplace=True)
    return(qc2)


#Sine wave for initial condition
def b_prep(b_init,nx,tau):
    #If u0 is the initial condition, then the output is given by:
    #y0 = (u0, 0_1, u0 \otimes u0, 0_2 , ... , u0^{\otimes tau})
    #where 0_j = (u0, 0_1, u0 \otimes u0, 0_2 , ... , u0^{\otimes tau})

    y0   = np.zeros((int(tau * nt * nx**tau)))
    temp = 1.
    for i in range(0,tau):
        temp = np.kron(temp,b_init)
        st = i*nx**tau
        ed = i*nx**tau + nx**(i+1)
        y0[st:ed] = temp
    y0 = y0/np.linalg.norm(y0)
    return(y0)


#Sigma coordinates
sigpm = np.array([[1,0],[0,0]])
sigp = np.array([[0,1],[0,0]])
sigm = np.array([[0,0],[1,0]])
sigmp = np.array([[0,0],[0,1]])
I2 = np.identity(2)

#Logs
log_nx  = int(math.log2(nx))
log_nt  = int(math.log2(nt))
log_tau = int(math.log2(tau))


