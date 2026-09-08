#Utilities to encode the matrix from the Carleman linearized Burgers' equation.
#The LCNU method is used, where each nonunitary is encoded into a unitary matrix of the form U=U_1U_2

#The general procedure is taken from DH26, with specifics also taken from DQ26.
#For example the F_2^{(e)} matrix comes from DQ26

#Definitions & Acronyms
#sigma_0=|0><0|, sigma_1=|0><1|, sigma_2=|1><0|, sigma_3=|1><1|, sigma_4=I 
#qbs = quinary bit string elements. Values in {0,1,2,3,4} = {sigma_0,...,sigma_4}
#qbs_bar = the unitary complement of each element in qbs. Values in {0,1}={Pauli-X,I}
#qbs_qbsT = each qbs element times its transpose. Values in {0,1,2}={open contrl,closed control,no control}
#GS24 = Gnanasekaran & Surana 2024. "Efficient variational quantum linear solver for structured sparse matrices"
#DH26 = Demirdjian & Hogancamp et al. 2026. "Quantum Data Loading for Carleman Linearized Systems: Application to the Lattice-Boltzmann Equation"
#DQ26 = Demirdjian & Quinn et al. 2026. "A Scalable Approach to Solve the Carleman Linearized Burgers' Equation on a Quantum Computer"
#Demi22 = ...

import numpy as np
from qiskit import QuantumCircuit, transpile
import Carleman_Dilation
import circuit_utils as cu
from qiskit_aer import AerSimulator


def binary_str(qbs):
    """Calculates the unitary complement of a tensor product of sigma basis elements following Theorem 2 of GS24.

    Args: A qbs representing a tensor product of sigma matrices

    Returns: The corresponding bbs representing the untitary complement
    """
    qbs_bar = []
    for i in range(0,len(qbs)):
        bbs = []
        for j in range(0,len(qbs[i])):
            if (qbs[i][j] in [0,3,4]):
                bbs += [1]
            elif (qbs[i][j] in [1,2]):
                bbs += [0]
        qbs_bar.append(bbs)
    return (qbs_bar)


def tertiary_str(qbs):
    """The values of each element of qbs times itself. Values represent the following:
    #0=open control, 1=closed control, 2=no control.
    #See proof of Theorem 3 in GS24 for details.

    Args: A qbs representing a tensor product of sigma matrices

    Returns: String of integers corresponding to open, closed, or no control 
             Ex: Consider A=sigma_1 otimes sigma_2. Then AA^T=sigma_0 otimes sigma_3.
             From Theorem 3 of GS24, this is represented by the transformation qbs=[1,2] => qbs_qbsT=[0,1]
    """
    qbs_qbsT = []
    for i in range(0,len(qbs)):
        tbs0 = []
        for j in range(0,len(qbs[i])):
            if (qbs[i][j] in [0,1]):
                tbs0 += [0]
            elif (qbs[i][j] in [2,3]):
                tbs0 += [1]
            elif (qbs[i][j] in [4]):
                tbs0 += [2]
        qbs_qbsT.append(tbs0)
    return(qbs_qbsT)


def U1(qc,qbs,params):
    """Controlled U_1 circuit

    Args: qc: circuit to append to 
          qbs: a qbs representing a tensor product of sigma matrices

    Returns: Controlled-U_1 circuit
    """
    nqubit = params['nqubit']
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
        qc.mcx(Clist,nqubit-1,mode='noancilla')
    else:
        qc.cx(nqubit,nqubit-1)
    if (Xlist):
        for k in range(0,len(Xlist)):
            qc.x(Xlist[k])
    return(qc)


def U1U2_circ_L1e(circs,coeffs,params):
    """Following DH26, embed the three non-untary L_1^{(e)} terms into a unitary, where
    L_1^{(e)} = (I_{n_t} - S_{+1} + rho_1^{log n_t}) otimes I,
    and S_{+1} is the incrementer.

    Args: circs=list of circuits to append to, coeffs=list of coefficients to append to

    Returns: (1) The circuit for Ctrl-U with control on the ancilla qubit for the Hadamard Test. 
             (2) The coefficients for each embedded circuit.
    """
    alpha,nx,nt,nqubit = params['alpha'], params['nx'], params['nt'], params['nqubit']
    t1 = [4]*int(np.log2(2 * nt * nx**alpha))
    t2 = []
    t3 = [1]*int(np.log2(nt)) + [4]*int(np.log2(2 * nx**alpha))
    qbs = [t1,t2,t3]
    coeffs += [1.,-1.,1.]

    for t in range(0,3):
        qc = QuantumCircuit(nqubit+1)
        
        #Controlled U_2 circuit
        qc.cx(nqubit,nqubit-1) #X \otimes \overline{L}
        qbs_bar = binary_str([qbs[t]])[0] #Unitary complement for the sigma terms
        for k in range(0,len(qbs_bar)):
            if (qbs_bar[k] == 0):
                qc.cx(nqubit,nqubit-2-k)
        if (t == 1): 
            cu.Incrementer(int(np.log2(nt)),qc,int(np.log2(2 * nx**alpha)),params)

        #Controlled U_1 circuit
        qc = U1(qc,qbs[t],params)
        circs.append(qc)
    return(circs,coeffs)


def U1U2_circ_Lejj(circs,coeffs,params):
    """Following DH26, embed the linear non-untary L_{2}^{(e)} terms into the unitary U=(U_1)(U_2)

    Args: circs=list of circuits to append to, coeffs=list of coefficients to append to

    Returns: (1) The circuit for Ctrl-U with control on the ancilla qubit for the Hadamard Test. 
             (2) The coefficients for each embedded circuit.
    """
    alpha, nx, nt, nqubit, dt, dx, nu = params['alpha'], params['nx'], params['nt'], params['nqubit'], params['dt'], params['dx'], params['nu']

    #Static lists
    t1 = [4]*int(int(np.log2(nt))) #I_2^{\otimes int(np.log2(nt))}
    t2 = [0]*int(int(np.log2(nt))) #(sig_+sig_-)^{\otimes int(np.log2(nt))}
    t1t2 = []
    t1t2.append(t1)
    t1t2.append(t2)
    for t in range(0,2):
        for j in range(1,alpha+1):
            qbs = []
            qbs += t1t2[t]
            qbs += ([0] + [3]*int(np.log2(nx)-1))*(alpha-j) + [3] #(sig0 otimes sig3^{log(nx)-1})^{\otimes (alpha-j)} otimes sig3
            qbs_bar = binary_str([qbs])[0] #Complement for the sigma terms

            for l in range(0,j):
                for pm in range(0,3):
                    #Skip the identity term for l>0 since they are combined
                    if pm==1 and l>0: continue

                    qc = QuantumCircuit(nqubit+1)
                    qc.cx(nqubit,nqubit-1) #X \otimes \overline{L}
                    #U2 for the sigma terms 
                    for k in range(0,len(qbs_bar)):
                        if (qbs_bar[k] == 0):
                            qc.cx(nqubit,nqubit-2-k)
                                        
                    #F_1 = Incrementer + -2*I + Decrementer
                    #pm==0 is incrementer, pm==1 is I, pm==2 is decrementer
                    #The pm order above is chosen to make coefficient calculation simpler
                    if (pm == 0):
                        cu.Decrementer(int(np.log2(nx)),qc,int(np.log2(nx))*(j-l-1),params)
                    if (pm == 2): 
                        cu.Incrementer(int(np.log2(nx)),qc,int(np.log2(nx))*(j-l-1),params)

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
                    circs.append(qc)

                    #Coefficients
                    factor = 2*j if pm==1 else 1
                    coeffs.append(-1. * (-1)**t * (-1)**pm * factor * dt*nu/dx**2)
                    del([qc,Xlist,Clist,qbs_qbsT])
            del([qbs,qbs_bar])
    return(circs,coeffs)


def U1U2_circ_Lejp1j(circs,coeffs,params):
    """Following DH26, embed the nonlinear (quadratic) non-untary L_{2}^{(e)} terms into the unitary U=(U_1)(U_2)
 
    Args: circs=list of circuits to append to, coeffs=list of coefficients to append to

    Returns: (1) The circuit for Ctrl-U with control on the ancilla qubit for the Hadamard Test. 
             (2) The coefficients for each embedded circuit.
    """
    alpha, nx, nt, nqubit, dt, dx, nu = params['alpha'], params['nx'], params['nt'], params['nqubit'], params['dt'], params['dx'], params['nu']

    #Static lists
    t1 = [4]*int(int(np.log2(nt))) #I_2^{\otimes int(np.log2(nt))}
    t2 = [0]*int(int(np.log2(nt))) #(sig_+sig_-)^{\otimes int(np.log2(nt))}
    t1t2 = []
    t1t2.append(t1)
    t1t2.append(t2)
    for t in range(0,2):
        for j in range(1,alpha):
            qbs = []
            qbs += t1t2[t]
            qbs += ([0] + [3]*int(np.log2(nx)-1))*(alpha-j-1) + [1] #(sig0 otimes sig3^{log(nx)-1})^{\otimes (alpha-j-1)} otimes sig1
            qbs_bar = binary_str([qbs])[0] #Complement for the sigma terms

            for l in range(0,j):
                for pm in range(0,2):
                    qc = QuantumCircuit(nqubit+1)

                    #sigma contributions
                    qc.cx(nqubit,nqubit-1) #X \otimes \overline{L}
                    for k in range(0,len(qbs_bar)):
                        if (qbs_bar[k] == 0):
                            qc.cx(nqubit,nqubit-2-k)

                    #K^{n_x^2,n_x^l}
                    cu.commutation_circ(2*int(np.log2(nx)),l*int(np.log2(nx)),qc,int(np.log2(nx))*(j-l-1),nqubit)
                
                    #The Incrementer/Decrementer term from equation 21 of DQ26
                    if (pm == 0):
                        cu.P_plus(int(np.log2(nx)),qc,int(np.log2(nx)*(j-1)),params)
                    elif (pm == 1):
                        cu.P_minus(int(np.log2(nx)),qc,int(np.log2(nx)*(j-1)),params)

                    #sig0^{\otimes int(np.log2(nx))} \otimes K^{n_x^l,n_x}
                    cu.commutation_circ(int(np.log2(nx))*l,int(np.log2(nx)),qc,int(np.log2(nx))*(j-l-1),nqubit)

                    #P_k matrix from DH26 with k=2 
                    cu.P_k(nx,nx-1,qc,int(j*np.log2(nx)),params)

                    #U1
                    qbs_qbsT = tertiary_str([qbs])[0] #Get bitstring for AA^T
                    qbs_qbsT += [1]*int(np.log2(nx)) + [2]*j*int(np.log2(nx))
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
                    circs.append(qc)  
                                  
                    #Coefficients
                    coeffs.append(-1. * (-1)**t * (-1)**pm * dt/(2.*dx))
                    del([qc,Xlist,Clist,qbs_qbsT])
            del([qbs,qbs_bar])
    return(circs,coeffs)


def create_circs(params):
    """Create the circuits for the Carleman dilated matrix. This function is a wrapper for the U1U2_circ_L1e, U1U2_circ_Lejj, and U1U2_circ_Lejp1j functions.

    Args: params=parameters for the Burgers' equation

    Returns: (1) The circuit for Ctrl-U with control on the ancilla qubit for the Hadamard Test. 
             (2) The coefficients for each embedded circuit.
    """
    circs = []
    coeffs = []
    circs,coeffs = U1U2_circ_L1e(circs,coeffs,params)
    circs,coeffs = U1U2_circ_Lejj(circs,coeffs,params)
    circs,coeffs = U1U2_circ_Lejp1j(circs,coeffs,params)

    #Transpile all circuits
    simulator = AerSimulator()
    for i,circ in enumerate(circs):
        circs[i] = transpile(circ, simulator, optimization_level=3)

    return(circs,coeffs)


def validate_CarlemanDilated_Matrix(circs,coeffs,params):
    """Validate the circuits from create_circs() with the actual Carleman linearized matrix.

    Args: circs,coeffs = output from create_circs

    Returns: Le_real = the actual Carleman linearized matrix
             Le_circ = the matrix derived from the circuits
    """
    nqubit,coeff_thresh,error_thresh = params['nqubit'],params['coeff_thresh'],params['error_thresh']
    Le_circ = 0.
    for i in range(len(circs)):
        print('Working on', i)
        #Run the circuit and extract the unitary matrix
        simulator = AerSimulator()
        circ = transpile(circs[i], simulator)
        circ.save_unitary()                               # Must save unitary to "get_unitary" later
        result = simulator.run(circ).result()             # Run
        unitary = result.get_unitary(circ).to_matrix()    # Get unitary
        U_circ = unitary[int(2**(nqubit)):,int(2**(nqubit)):]**2  # Post-select on the last ancilla=1
        if (np.abs(coeffs[i])>coeff_thresh): #Only sum terms with sufficient magnitude
            Le_circ += coeffs[i] * U_circ[:int(2**(nqubit-1)),:int(2**(nqubit-1))] # Extract the embedded non-untiary
        del([circ,unitary,result,U_circ])
    
    #Carleman dilated matrix
    Le_real = Carleman_Dilation.Carleman_Dilation_Matrix(params)
    
    #Check Matrices against eachother
    #error = np.linalg.norm(Le_circ-Le_real)#/np.linalg.norm(Le_real)
    error = np.max(np.abs(Le_circ-Le_real))
    if (error < error_thresh):
        print('Successful validation, max error is ', error)
    else:
        print('Unsuccessful validation, max error is ', error)

    return(Le_real,Le_circ)


def extract_Aj_from_Uj(circs,params):

    nqubit = params['nqubit']

    simulator = AerSimulator()
    Aj = []
    for j in range(len(circs)):
        circ = transpile(circs[j], simulator)
        circ.save_unitary()                               # Must save unitary to "get_unitary" later
        result = simulator.run(circ).result()             # Run
        unitary = result.get_unitary(circ).to_matrix()    # Get unitary
        U_circ = unitary[int(2**(nqubit)):,int(2**(nqubit)):]**2  # Post-select on the last ancilla=1
        Aj.append(U_circ[:int(2**(nqubit-1)),:int(2**(nqubit-1))]) # Extract the embedded non-untiary
        
    return(Aj)


def extract_UZUd(qc_init,params):
    nqubit = params['nqubit']

    simulator = AerSimulator()
    UZUd = []
    for k in range(nqubit-1):
        circ = QuantumCircuit(nqubit-1)
        circ.compose(qc_init.inverse(),list(range(0,nqubit-1)),inplace=True) #U_b^dagger
        circ.z(k)
        circ.compose(qc_init,list(range(0,nqubit-1)),inplace=True) #U_b

        circ = transpile(circ, simulator)
        circ.save_unitary()
        result = simulator.run(circ).result()
        UZUd.append(result.get_unitary(circ).to_matrix())

    return(UZUd)

