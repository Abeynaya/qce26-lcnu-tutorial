#Various circuits used for the LCNU of the Carleman linearized Burgers' equation 
#DH26 = Demirdjian & Hogancamp et al. 2026. "Quantum Data Loading for Carleman Linearized Systems: Application to the Lattice-Boltzmann Equation"
#DQ26 = Demirdjian & Quinn et al. 2026. "A Scalable Approach to Solve the Carleman Linearized Burgers' Equation on a Quantum Computer"
#Demi22 = ...


import numpy as np
from qiskit import QuantumCircuit, transpile, qpy
from qiskit.circuit import ParameterVector
from qiskit_aer import AerSimulator

def Ansatz_Sim9_Modified(params):
    """#Ansatz circuit. Modified version of circuit 9 from Sims et al. (2019)
    #Same one used in Demi22

    Args: nqubit=number of qubits in ansatz circuit, ntheta=number of variational parameters, nlayer=number of ansatz layers

    Returns: A circuit representing the parameterized ansatz
    """
    nqubit,ntheta,nlayer = params['nqubit'],params['ntheta'],params['nlayer']    
    circ = QuantumCircuit(nqubit-1)
    thetas = ParameterVector('params',ntheta)
    m = -1
    for ly in range(0,nlayer):
        for iz in range(0,nqubit-1):
            circ.h(iz)
        for iz in range(0,nqubit-2):
            circ.cz (iz,iz+1)
        for iz in range(0,nqubit-1):
            m += 1
            circ.ry(thetas[m],iz)
    return(circ)


def Ansatz_Sim18Mod(params):
    """#Ansatz circuit. Modified version of circuit 18 from Sims et al. (2019)

    Args: nqubit=number of qubits in ansatz circuit, ntheta=number of variational parameters, nlayer=number of ansatz layers

    Returns: A circuit representing the parameterized ansatz
    """
    nqubit,ntheta,nlayer = params['nqubit'],params['ntheta'],params['nlayer']
    circ = QuantumCircuit(nqubit-1)
    thetas = ParameterVector('params',ntheta)
    m = -1    
    for ly in range(0,nlayer):
        for iz in range(0,nqubit-1):
            m += 1
            circ.ry(thetas[m],iz)
        m += 1
        circ.cry(thetas[m],nqubit-2,0)
        for iz in range(nqubit-3,-1,-1):
            m += 1
            circ.cry(thetas[m],iz,iz+1)
    return(circ)


def Incrementer(nq,qc,offset,params):
    """The quantum adder circuit. Note, there are known improvements to this circuit.
    
    Args: nq=number of qubits for incrementer, qc=existing quantum circuit to append, offest=starting qubit

    Returns: Quantum circuit appended with the incrementer
    """
    nqubit = params['nqubit'] 
    for q in range(0,nq-2):
        controls = list(range(0+offset,nq-q-1+offset))
        controls.append(nqubit)
        qc.mcx(controls,nq-q-1+offset,mode='noancilla')
        del(controls)
    if (nq>1):
        qc.mcx([nqubit,0+offset],1+offset)
    qc.cx(nqubit,0+offset)


def Decrementer(nq,qc,offset,params):
    """The quantum subtractor circuit. Note, there are known improvements to this circuit.
    
    Args: nq=number of qubits for decrementer, qc=existing quantum circuit to append, offest=starting qubit

    Returns: Quantum circuit appended with the decrementer
    """
    nqubit = params['nqubit']
    qc.cx(nqubit,0+offset)
    qc.mcx([nqubit,0+offset],1+offset)
    for q in range(nq-3,-1,-1):
        controls = list(range(0+offset,nq-q-1+offset))
        controls.append(nqubit)        
        qc.mcx(controls,nq-q-1+offset,mode='noancilla')
        del(controls)


def commutation_circ(m,n,qc,offset,nqubit):
    """Circuit for the commutation matrix K^{2^m,2^n}. 
    See: https://en.wikipedia.org/wiki/Commutation_matrix
    
    Args: m=exponent for the first dimension, n=exponent for the second dimension, qc=existing quantum circuit to append, offset=starting qubit, nqubit=number of qubits

    Return: Quantum circuit appended with the commutation matrix
    """
    for r in range(n-1,-1,-1):
        for q in range(m-1,-1,-1):
            qc.cswap(nqubit,n-r+q-1+offset,n-r+q+offset)
    return(qc)


def P_k(n,b,qc,offset,params):
    """The P_k matrix from DH26
    
    Args: n=size of matrix, b=decimal number, qc=existing quantum circuit to append, offest=starting qubit 

    Returns: Quantum circuit appended with the P_k matrix
    """
    nqubit = params['nqubit']
    b_str = format(b,'0'+str(int(np.log2(n)))+'b') #Convert decimal number to binary string
    for q in reversed(range(0,len(b_str))):
        if (b_str[q] == '1'):
            qc.cx(nqubit,offset + q)


def P_minus(nq,qc,offset,params):
    """The Decrementer term from equation 21 of DQ26
    
    Args: nq=number of qubits for subroutine, qc=existing quantum circuit to append, offest=starting qubit

    Returns: Quantum circuit appended with the P^- matrix
    """
    nx,nqubit = params['nx'], params['nqubit']
    Decrementer(nq,qc,offset,params)
    for q in range(0,nq):
        qc.mcx([nqubit,int(np.log2(nx)-q-1+offset)],int(2*np.log2(nx)-q-1+offset))


def P_plus(nq,qc,offset,params):
    """The Incrementer term from equation 21 of DQ26
    
    Args: nq=number of qubits for subroutine, qc=existing quantum circuit to append, offest=starting qubit

    Returns: Quantum circuit appended with the P^+ matrix
    """
    nx,nqubit = params['nx'], params['nqubit']
    Incrementer(nq,qc,offset,params)
    for q in range(0,nq):
        qc.mcx([nqubit,int(np.log2(nx)-q-1+offset)],int(2*np.log2(nx)-q-1+offset))


def create_beta_ij(i,j,qc_carl,qc_anz,params):
    """Create the beta_ij circuits from the VQLS cost function

    Args: i,j=looping indices, qc_carl=circuits for LCNU of L^e, qc_anz=parameterized ansatz circuit

    Returns: beta_ij
    """

    nqubit = params['nqubit']
    beta_ij = QuantumCircuit(nqubit+1)
    #Ansatz=V(theta)
    beta_ij.compose(qc_anz,qubits=list(range(0,nqubit-1)),inplace=True)
    #Had test
    beta_ij.h(nqubit)
    #C-U_i    
    beta_ij.compose(qc_carl[i],list(range(0,nqubit+1)),inplace=True) 
    #OC-U_j
    beta_ij.x(nqubit) #convert qc_carl[j] to open control
    beta_ij.compose(qc_carl[j],list(range(0,nqubit+1)),inplace=True)
    beta_ij.x(nqubit) #convert qc_carl[j] to open control
    #Had test
    beta_ij.h(nqubit) #for Had test
    beta_ij.save_statevector() #exact statevector
    return(beta_ij)


def measure_beta_ij(circs_beta,coeffs_ij,thetas):
    """Measure ancilla's in beta circuit to find expectation value. See GS2024 for details.

    Args: circs_beta=beta circuits, coeffs_ij=coefficients from LCNU, thetas=values for variational parameters

    Returns: beta_ijk
    """
    #Assign updated parameters
    circs = [qc.assign_parameters(thetas) for qc in circs_beta]

    # Batch all parameter sets into one run call
    backend = AerSimulator(method="statevector")
    job = backend.run(circs)
    result = job.result()
    probs = [np.abs(result.get_statevector(i).data)**2 for i in range(len(circs))]

    n = int(np.log2(len(probs[0])))
    beta = 0
    for i in range(0,len(probs)):
        P00 = np.sum(probs[i][:2**(n-2)])
        #P01 = np.sum(probs[i][2**(n-2):2**(n-1)])
        P10 = np.sum(probs[i][2**(n-1):3*2**(n-2)])
        #P11 = np.sum(probs[i][3*2**(n-2):])
        beta += coeffs_ij[i] * (P00-P10)
    return(beta)


def create_delta_ijk(i,j,k,qc_carl,qc_anz,qc_init,params):
    """Create the delta_ijk circuits from the VQLS cost function

    Args: i,j,k=looping indices, qc_carl=circuits for LCNU of L^e, qc_anz=parameterized ansatz circuit

    Returns: bdelta_ijk
    """
    #C-U=closed control U gate, OC-U=open control U gate
    nqubit= params['nqubit']
    delta_ijk = QuantumCircuit(nqubit+1)
    #Ansatz=V(theta)
    delta_ijk.compose(qc_anz,qubits=list(range(0,nqubit-1)),inplace=True) 
    #Had test
    delta_ijk.h(nqubit) 

    #C-U_i
    delta_ijk.compose(qc_carl[i],list(range(0,nqubit+1)),inplace=True) 

    #OC-U_j
    delta_ijk.x(nqubit) #convert qc_carl[j] to open control
    delta_ijk.compose(qc_carl[j],list(range(0,nqubit+1)),inplace=True) #U_j
    delta_ijk.x(nqubit) #convert qc_carl[j] to open control

    #U_b^dagger
    delta_ijk.compose(qc_init.inverse(),list(range(0,nqubit-1)),inplace=True)

    #C-OC-Z
    delta_ijk.x(nqubit-1) #convert c-c-z to c-oc-z, i.e. open control
    delta_ijk.ccz(nqubit-1,nqubit,k)
    delta_ijk.x(nqubit-1) #convert c-c-z to c-oc-z, i.e. open control

    #U_b
    delta_ijk.compose(qc_init,list(range(0,nqubit-1)),inplace=True) #U_b

    #Had test
    delta_ijk.h(nqubit)
    delta_ijk.save_statevector() #exact statevector
    return(delta_ijk)


def measure_delta_ijk(circs_delta,coeffs_ijk,thetas):
    """Measure ancilla's in beta circuit to find expectation value. See GS2024 for details.

    Args: circs_beta=beta circuits, coeffs_ij=coefficients from LCNU, thetas=values for variational parameters

    Returns: beta_ijk
    """
    #Assign updated parameters
    circs = [qc.assign_parameters(thetas) for qc in circs_delta]

    # Batch all parameter sets into one run call
    backend = AerSimulator(method="statevector")
    job = backend.run(circs)
    result = job.result()
    probs = [np.abs(result.get_statevector(i).data)**2 for i in range(len(circs))]

    n = int(np.log2(len(probs[0])))
    delta = 0
    for i in range(0,len(probs)):
        P00 = np.sum(probs[i][:2**(n-2)])
        #P01 = np.sum(probs[i][2**(n-2):2**(n-1)])
        P10 = np.sum(probs[i][2**(n-1):3*2**(n-2)])
        #P11 = np.sum(probs[i][3*2**(n-2):])
        delta += coeffs_ijk[i] * (P00-P10)
    return(delta)


import pennylane as qml
import pennylane.estimator as qre

from qiskit import QuantumCircuit
from qiskit.circuit.library import MCXGate


def is_mcx_instruction(instruction):
    """
    Return True if a Qiskit instruction is an MCX gate.

    This handles MCX gates whose names may be:
        mcx
        mcx_gray
        mcx_recursive
        mcx_v_chain
        etc.
    """

    operation = instruction.operation

    return (
        isinstance(operation, MCXGate)
        or operation.name.lower().startswith("mcx")
    )


def get_mcx_t_count(qiskit_circuit):
    """
    Compute the T-gate count contributed by MCX gates.

    Assumed formula:

        T_count(MCX with k controls) = 8*k - 12
    """

    total_mcx_t_count = 0
    mcx_data = []

    for instruction in qiskit_circuit.data:
        if is_mcx_instruction(instruction):
            operation = instruction.operation

            # An MCX gate has k control qubits and one target qubit.
            num_controls = getattr(
                operation,
                "num_ctrl_qubits",
                len(instruction.qubits) - 1,
            )

            t_count = 8 * num_controls - 12

            total_mcx_t_count += t_count

            mcx_data.append({
                "controls": num_controls,
                "t_count": t_count,
            })

    return total_mcx_t_count, mcx_data


def get_t_count_from_resources(resources):
    """
    Extract the T-gate count from PennyLane's Resources object.

    PennyLane stores CompressedResourceOp objects as dictionary keys,
    rather than using strings such as "T".
    """

    for resource_op, count in resources.gate_types.items():
        if resource_op.op_type.__name__ == "T":
            return count

    return 0


def estimate_remaining_gates(qiskit_circuit):
    """
    Estimate the T-gate count of all non-MCX gates using PennyLane.

    MCX gates are removed from a copy of the circuit before conversion.
    The original Qiskit circuit is not modified.

    No transpilation is performed.
    """

    # Make an empty circuit with the same registers as the input circuit.
    circuit_without_mcx = qiskit_circuit.copy_empty_like()

    # Copy every instruction except MCX gates.
    for instruction in qiskit_circuit.data:
        if not is_mcx_instruction(instruction):
            circuit_without_mcx.append(
                instruction.operation,
                instruction.qubits,
                instruction.clbits,
            )

    # Convert only the non-MCX circuit to PennyLane.
    pennylane_circuit = qml.from_qiskit(circuit_without_mcx)

    # Estimate resources using the Clifford+T gate set.
    estimator = qre.estimate(
        pennylane_circuit,
        gate_set={
            "Hadamard",
            "S",
            "CNOT",
            "T",
        },
    )

    resources = estimator()

    remaining_t_count = get_t_count_from_resources(resources)

    return remaining_t_count, resources


def estimate_total_t_count(qiskit_circuit):
    """
    Compute the total T-gate count of a Qiskit circuit.

    The total is:

        MCX T count + PennyLane-estimated remaining T count
    """

    mcx_t_count, mcx_data = get_mcx_t_count(qiskit_circuit)

    remaining_t_count, resources = estimate_remaining_gates(
        qiskit_circuit
    )

    total_t_count = mcx_t_count + remaining_t_count

    return {
        "total_t_count": total_t_count,
        "mcx_t_count": mcx_t_count,
        "remaining_t_count": remaining_t_count,
        "mcx_gates": mcx_data,
        "resources": resources,
    }


def estimate_circuits(circuits):
    """
    Estimate the total T-gate count for every circuit in a list.

    Returns a list of dictionaries containing the estimate for each
    circuit.
    """

    results = []

    for index, circuit in enumerate(circuits):
        estimate = estimate_total_t_count(circuit)

        results.append({
            "circuit_index": index,
            **estimate,
        })

    return results
