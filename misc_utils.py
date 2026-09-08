import numpy as np

def create_Be(u_0,params):

    nx,nt,alpha = params['nx'],params['nt'],params['alpha']

    ye0 = np.zeros(int(2*nx**alpha))
    sizes = [int(nx**j) for j in range(1,alpha+1)]
    sizes.insert(0,0)
    delta = np.sum(sizes)
    offset = int(2*nx**alpha-delta)
    tmp = np.array([1])
    for j in range(1,alpha+1):
        tmp = np.kron(tmp,u_0)
        ye0[offset+np.sum(sizes[:j]):offset+np.sum(sizes[:j+1])] = tmp

    Be = np.zeros(int(2*nt*nx**alpha))
    Be[:int(2*nx**alpha)] = ye0
    Be = Be/np.linalg.norm(Be) #Normalize to make things simpler 
    return(Be)


def normalized_initial_condition(params):
    """Creates the initial condition vector for the zero padded Carleman linear system
    Specifically, craetes B^{(e)} of eq. 19 in DH26.

    Args:

    Returns:    
    """
    nx,nt,alpha,Length,ic_type = params['nx'],params['nt'],params['alpha'],params['Length'],params['ic_type']
    sig  = 0.5
    mu   = np.pi

    if (ic_type=='Gaussian'):
        if (nx<2**10 and nt<2**10 and alpha<4): #slow for large circuits
            x    = np.linspace(0,Length,nx)
            u_0 = 1.0 / (np.sqrt(2.0 * np.pi) * sig**2) * np.exp(-(x - mu)**2/(2.*sig**2))
            #u_0 = u_0/np.linalg.norm(u_0) #Normalize to make things simpler
    elif (ic_type=='Impulse'):
        u_0 = np.zeros(nx)
        u_0[0] = 1.

    Be = create_Be(u_0,params)     
    return(Be)


def solve_sys(Le_real,Be,params):

    nx,nt,alpha = params['nx'],params['nt'],params['alpha']

    x_class_full = np.linalg.solve(Le_real,Be)
    x_class_full = x_class_full/np.linalg.norm(x_class_full) #Normalize 
    x_class = np.zeros(nt*nx)
    delta = np.sum([int(nx**j) for j in range(1,alpha+1)])
    for i in range(nt):
        offset = int((i+1)*2*nx**alpha - delta)
        x_class[i*nx:(i+1)*nx] = x_class_full[offset:nx+offset]
    return(x_class)


