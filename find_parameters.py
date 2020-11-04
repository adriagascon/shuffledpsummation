from math import log,sqrt,exp,ceil,pi,floor,inf
from tabulate import tabulate
from scipy.optimize import basinhopping,golden
from shuffleddp.amplification_bounds import BennettExact
from shuffleddp.mechanisms import RRMechanism

class Parameters:
    """Class to encapsulate privacy parameters for an algorithm
    
    Attributes:
        eps: The epsilon for dp
        delta: The delta for dp
        n: The number of clients providing data
    """
    
    def __init__(self,eps,delta,n):
        """ Initiates a parameters object

        Args:
            eps: the parameter epsilon of dp
            delta: the parameter delta of dp
            n: the number of clients
        """
        
        assert(delta<0.1)
        self.eps=eps
        self.delta=delta
        self.n=n

class Results:
    """Class to contains results of a mechanism

    Attributes:
        algo: A string indicating which mechanism was used
        params: A Parameters type object specifying the guarantee
        messages: The number of messages that are required (per client)
        mse: The mean squared error of the resulting estimate
    """
    
    def __init__(self,algo,params,messages,mse,choices=None):
        """Creates a Results type object

        Args:
            algo: A string indicating which mechanism was used
            params: A Parameters type object specifying the guarantee
            messages: The number of messages that are required (per client)
            mse: The mean squared error of the resulting estimate
        """
        
        self.algo=algo
        self.params=params
        self.messages=messages
        self.mse=mse

    def __str__(self):
        """Creates a string listing the attributes"""
        
        return self.algo+":"+str(self.params)+"\n"+"messages:"+str(self.messages)+"\n"+"mse:"+str(self.mse)

    def asdict(self):
        """Returns Results as a dictionary

        The dictionary has keys, "PROTOCOL", "clients", "epsilon",
        "delta", "MESSAGES", "MSE" and "MSE/n^2"

        Each is the key for an attribute of the REsults object or
        of its Parameters attribute.

        "PROTOCOL" and "clients" are the keys for algo and params.n respectively
        """
        
        return {
            "PROTOCOL" : self.algo,
            "clients" : self.params.n,
            "epsilon" : self.params.eps,
            "delta" : self.params.delta,
            "MESSAGES" : self.messages,
            "MSE" : "{0:.1f}".format(self.mse),
            "MSE/n^2" : "{0:.9f}".format(self.mse/self.params.n**2)
        }

def get_gamma(eps, delta, domain_size, n):
    """Get gamma (lying rate) for Randomized Response in the shuffle model

    Args:
        eps: epsilon to be achieved for dp
        delta: delta to be achieved for dp
        domain_size: number of possible responses
        n: number of clients

    Returns:
        A floating point value gamma
    """
    
    bound = BennettExact(RRMechanism(k=domain_size), tol=1e-15)
    krr = RRMechanism(k=domain_size)
    krr.set_eps0(bound.get_eps0(eps, n, delta))
    return krr.get_gamma()[0]


def errorforrecursive(n,m,p,e,d):
    """Computes error of recursive method

    Args:
        n: number of clients
        m: number of messages per client
        p_list: list of precisions for each message
            (note that p[0] is ignored to make the indexing match the one indexing of the paper)
        eps_list: list of epsilons for each message (again e[0] is ignored)
        delts_list: total delta (the code assumes this is to be split equally accross messages)#

    Returns:
        Mean squared error as a floating point value
    """
    
    q_list=[1]*(m+1) #Will store cumulative products of p_list
    B=[0]*(m+1)
    for j in range(1,m+1):
        if eps_list[j]<=0:
            return 10**10
        q_list[j]=q_list[j-1]*p_list[j]
        islast=1 if j==m else 0
        gamma=get_gamma(eps_list[j],d/m,p_list[j]+islast,n)
        B[j]=n*(gamma*(p_list[j]**2-1)/12+(p_list[j]-1)**2*gamma*(1-gamma)/4)/(1-gamma)**2
    mse=n/(4*q_list[m]**2)+sum(B[j]/q_list[j]**2 for j in range(1,m+1))
    return mse


def recursive(params):
    """Results of recursive method, with parameters from the paper

    Args:
        params: the privacy parameters to be met (of type Parameters)

    Returns:
        A Results object containing the number of messages and mse
    """
    
    m=round(log(log(params.n,2),3))
    epsj=params.eps/m
    maxp=(params.n-1)*epsj/max(14*log(2*m/params.delta)/epsj,27)
    if maxp<1:
        return Results("Recursive",params,m,inf)
    a=((3**m-1)*params.n*params.eps**2/m**3)**(3**(-m-1))
    p_list=[min(ceil(a**(3**t)),floor(maxp)) for t in range(m+1)]
    eps_list=[epsj]*(m+1)
    return Results("Recursive",params,m,errorforrecursive(params.n,m,p_list,eps_list,params.delta))

def maxpned(n,eps,delta):
    """Computes the maximum valid p for fixed parameters

    Args:
        n: number of clients
        eps: the epsilon of the privacy guarantee
        delta: the delta of the privacy guarantee

    Returns:
        The maximum value of p for these parameters that the 
        amplificationbyshuffling module can handle
    """
    
    if eps<=0:
        return 0
    return floor((n-1)*eps/max(14*log(2/delta)/eps,27))

def maxp(eps,params,m):
    """Gives an upper bound on the p that could be valid with a given epsilon
    
    Args:
        eps: The parameter epsilon
        m: Number of messages (to split the delta between)
        params: A container for the other params

    Returns:
        An upper bound on the p for a single message with that epsilon
    """
    p = maxpned(params.n,eps,params.delta/m)
    if p <= 2:
      return 4
    return p

def get_mineps(params,m):
    """Provides a lookup table of the inverse of maxp as a function of eps
    
    Args:
        params: The parameters n, delta and epsilon. The latter is treated as an upper bound.
        m: Number of messages the delta is being split between

    Returns:
        A list to look up mineps for a give p
    """
    mineps=[0]*(maxp(params.eps,params,m)+1)
    for p in range(2,maxp(params.eps,params,m)+1):
        l=0
        u=params.eps
        while u-l>0.0000000001:
            mid=(u+l)/2
            if p<=maxp(mid,params,m):
                u=mid
            else:
                l=mid
        mineps[p]=u
    return mineps

def recursive2(params):
    """Results of recursive method, with two messages and numerically optimized parameters

    Args:
        params: The privacy parameters to be met (of type Parameters)

    Returns:
        A Results object containing the number of messages and mse
    """
    m=2
    choices=None
    selected_params = ([None, None], [None, None])
    mse=inf
    mineps=get_mineps(params,m)
    # Search through the possible values of p1 and p2
    for p1 in range(2,maxp(params.eps,params,m)+1):
        if p1==2:
            upperboundonp2=maxp(params.eps,params,m)
        else:
            upperboundonp2=lastgoodp2
        lastgoodp2=0
        for p2 in range(2,upperboundonp2+1):
            # p2 can't work if there is no known mineps for that p or the sum would be too big
            try:
              if mineps[p1]+mineps[p2]>params.eps:
                break
            except:
              print(p1)
              print(p2)
              print(mineps)
              break
            # Otherwise we perform a golden section search to choose how to split the epsilon
            def err(eps):
                """Calls the errorforrecursive function to get the error for a given epsilon split
                
                Args:
                    eps: The epsilon budget for the first message
                
                Return:
                    
                """
                if p1>maxp(eps,params,m) or p2>maxp(params.eps-eps,params,m):
                    return inf
                return errorforrecursive(params.n,m,[0,p1,p2],[0,eps,params.eps-eps],params.delta)
            res=golden(err,brack=(mineps[p1],params.eps-mineps[p2]))
            newmse=err(res)
            # If this is a record small error we record it
            if newmse<mse:
                mse=newmse
                lastgoodp2=p2
                choices="{} {} {} {}".format(p1,p2,res,params.eps-res)
                print('Best so far: {}'.format(choices))
                selected_params = ([p1, p2], [res, params.eps-res])
    print("parameters chosen for 2msgs:",choices)
    print("Analytical mse:",mse)
    return selected_params


def recursive3(params):
    """Results of recursive method, with three messages and numerically optimized parameters

    Args:
        params: The privacy parameters to be met (of type Parameters)

    Returns:
        A Results object containing the number of messages and mse
    """
    m=3
    selected_params = ([None, None, None], [None, None, None])
    choices=None
    mse=inf
    mineps=get_mineps(params,m)
    # Loop over possible p1,p2 and p3 in turn
    for p1 in range(2,maxp(params.eps,params,m)+1):
        mseforp1=inf
        if p1==2:
            upperboundonp2=maxp(params.eps,params,m)
        else:
            upperboundonp2=lastgoodp2
        lastgoodp2=0
        for p2 in range(p1,upperboundonp2+1):
            if p2==p1:
                upperboundonp3=maxp(params.eps,params,m)
            else:
                upperboundonp3=lastgoodp3
            lastgoodp3=0
            record=False
            for p3 in range(p2,upperboundonp3+1):
                if mineps[p1]+mineps[p2]+mineps[p3]>params.eps:
                    break
                # We now choose eps1 with a golden section search and for each function evaluation
                # we choose eps2 by golden section search inside it
                def err1(eps1):
                    def err2(eps2):
                        return errorforrecursive(params.n,m,[0,p1,p2,p3],[0,eps1,eps2,params.eps-eps1-eps2],params.delta)
                    opteps2=golden(err2,brack=(mineps[p2],params.eps-mineps[p1]-mineps[p3]), tol=10**(-4))
                    opteps2=max(0.00000001,min(params.eps-eps1-0.00000001,opteps2))
                    return (err2(opteps2),opteps2)
                res=golden(lambda e:err1(e)[0],brack=(mineps[p1],params.eps-mineps[p2]-mineps[p3]), tol=10**(-4))
                
                newmse=err1(res)[0]
                # If the MSE is a new record for this p1 record that fact so we can record
                # the last good p3 as an upper bound for p3 for the next value of p2.
                if newmse<mseforp1:
                    lastgoodp3=p3
                    mseforp1=newmse
                    record=True
                else:
                    if record:
                        break
                #If the MSE is a new record record that
                if newmse<mse:
                    mse=newmse
                    lastgoodp2=p2
                    choices="{} {} {} {} {} {}".format(p1,p2,p3,res,err1(res)[1],params.eps-res-err1(res)[1])
                    print(choices)
                    print('Best so far: {}'.format(choices))
                    selected_params = ([p1, p2, p3], [res, err1(res)[1], params.eps-res-err1(res)[1]])

    print("parameters chosen for 3msgs:",choices)
    return selected_params

