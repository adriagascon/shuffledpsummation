import numpy as np
import math
import scipy.stats
import find_parameters as fp
from shuffleddp.amplification_bounds import BennettExact
from shuffleddp.mechanisms import RRMechanism

def get_optimized_parameters(n, epsilon, delta, num_messages):
    assert(num_messages in [2, 3])
    try:
        multimessage_precomputed_parameters = np.load(
        'multimessage_precomputed_parameters.npy', allow_pickle='TRUE').item()
    except FileNotFoundError:
        multimessage_precomputed_parameters = {}
    if not (n, epsilon, delta, num_messages) in multimessage_precomputed_parameters:
        param_finder = fp.recursive2 if num_messages == 2 else fp.recursive3
        (precisions, epsilons) = param_finder(fp.Parameters(epsilon, delta, n))
        multimessage_precomputed_parameters[
            (n, epsilon, delta, num_messages)] = (precisions, epsilons)
        np.save('multimessage_precomputed_parameters.npy',
            multimessage_precomputed_parameters)
    if multimessage_precomputed_parameters[
        (n, epsilon, delta, num_messages)][0][0] == None:
        del multimessage_precomputed_parameters[
        (n, epsilon, delta, num_messages)]
        np.save('multimessage_precomputed_parameters.npy',
            multimessage_precomputed_parameters)
    return multimessage_precomputed_parameters[
        (n, epsilon, delta, num_messages)]

def get_gamma(eps, delta, domain_size, n):
    # (borja) Reducing the tolerance to avoid numerical issues
    bound = BennettExact(RRMechanism(k=domain_size), tol=1e-15)
    krr = RRMechanism(k=domain_size)
    krr.set_eps0(bound.get_eps0(eps, n, delta))
    return krr.get_gamma()[0]

def quantize(data, p, randomized_rounding = True, return_remainder = False):
    if randomized_rounding and return_remainder:
        raise NotImplementedError
    if randomized_rounding:
        return [int(x*p) + int(np.random.binomial(size=1, n=1, p=x*p-int(x*p)))
            for x in data]
    if return_remainder:
        return [int(x*p) for x in data], [x*p-int(x*p) for x in data]

class RealSumEstimator(object):
    epsilon_ = None
    delta_ = None

    def __init__(self, eps, d, n = None):
        self.epsilon_ = eps
        self.delta_ = d

    """
    Returns the estimate of the sum given a numpy array [data]
    containing floats in [0,1]
    """
    def estimate(self, data):
        raise NotImplementedError

class NonPrivateRealSum(RealSumEstimator):
    def estimate(self, data):
        return sum(data)

class CentralLaplaceRealSum(NonPrivateRealSum):
    def estimate(self, data):
        return super().estimate(data) + np.random.laplace(scale = 1./self.epsilon_)

class LocalLaplaceRealSum(NonPrivateRealSum):
    def estimate(self, data):
        return super().estimate(
            data + np.random.laplace(
                scale = 1./self.epsilon_, size = len(data))
        )

class CentralGeomRealSum(NonPrivateRealSum):
    def estimate(self, data):
        def sample_two_sided_geom_from_polya(n, r, p, n_samples=1):
            """
            Returns [n_samples] samples of two-sided geometric
            generated as the difference between n samples of polya(r, p)
            """
            res = []
            for _ in range(n_samples):
                r1 = scipy.stats.nbinom.rvs(r, p, size=n)
                r2 = scipy.stats.nbinom.rvs(r, p, size=n)
            res.append(sum(r1) - sum(r2))
            if n_samples == 1:
                return res[0]
            return res
        precision = np.sqrt(len(data))
        data_quant = quantize(data, precision)
        return (super().estimate(data_quant) + sample_two_sided_geom_from_polya(
        len(data), 1./len(data), np.exp(-self.epsilon_/precision))) / precision

class SingleMessageDiscreteSum(NonPrivateRealSum):
    def __init__(self, domain_size, n, **kwargs):
        super(self.__class__, self).__init__(**kwargs)
        self.n_ = n
        self.domain_size_ = domain_size
        self.gamma_ = get_gamma(
            self.epsilon_, self.delta_, self.domain_size_, self.n_)

        def lr(x):
            t = np.random.binomial(size=1, n=1, p = self.gamma_)
            if t == 0:
                return x
            return np.random.randint(0, self.domain_size_+1)
        self.local_randomizer_ = lr

    def estimate(self, data):
        def debias(x):
            return (x - self.n_*self.gamma_*self.domain_size_/2) / (1 - self.gamma_)
        return debias(sum(map(self.local_randomizer_, data)))

class SingleMessageRealSum(NonPrivateRealSum):
    def estimate(self, data):
        precision = int(len(data)**(1./3))
        estimator = SingleMessageDiscreteSum(
            domain_size=precision,
            n=len(data),
            eps=self.epsilon_,
            d=self.delta_)
        return estimator.estimate(quantize(data,precision)) / precision

class ManyMessageRealSum(NonPrivateRealSum):
    def __init__(self, eps, d, n, generic_params = True, num_messages = 2):
        self.n_ = n
        self.epsilon_ = eps
        self.delta_ = d
        self.num_messages_ = num_messages
        assert(np.log(1./self.delta_) >= 2*self.epsilon_)
        if generic_params:
            self.epsilons_ = [
                self.epsilon_ / self.num_messages_ for _ in range(self.num_messages_)
                ]
            epsj = self.epsilon_ / self.num_messages_
            maxp = (self.n_ - 1) * epsj / max(
                14 * np.log( 2 * self.num_messages_ / self.delta_) / epsj, 27
                )
            a = ((3**self.num_messages_-1) * self.n_ * self.epsilon_ **2 /
                self.num_messages_ **3 ) ** (3**(-self.num_messages_-1))
            # self.precisions_ = [
            #     float(int(self.n_**(3**(j -self.num_messages_-1)))) \
            #         for j in range(1, self.num_messages_+1)]
            self.precisions_ = [min(math.ceil(a**(3**t)),math.floor(maxp)) for t in range(1, self.num_messages_ + 1)]
            self.epsilons_ = [epsj] * (self.num_messages_)
        else :
            (self.precisions_, self.epsilons_) =  get_optimized_parameters(
                self.n_, self.epsilon_, self.delta_, self.num_messages_)
        self.deltas_ = [self.delta_ / self.num_messages_ for _ in range(self.num_messages_)]
        self.global_precisions_ = [self.precisions_[0]] * self.num_messages_
        for i in range(1, self.num_messages_):
            self.global_precisions_[i] = self.global_precisions_[i-1] * self.precisions_[i]
        self.estimators_ = [
            SingleMessageDiscreteSum(
                domain_size=self.precisions_[i] + (1 if i == self.num_messages_ - 1 else 0),
                n=self.n_,
                eps=self.epsilons_[i],
                d=self.deltas_[i])
            for i in range(self.num_messages_)]
    def estimate(self, data):
        print('\t -> Estimating with n = {}, epsilons = {}, precisions = {}'.format(
            self.n_, self.epsilons_, self.precisions_))
        result = 0
        for i in range(self.num_messages_):
            if i == self.num_messages_ - 1:
                data_quant = quantize(
                    data,
                    self.precisions_[i],
                    randomized_rounding = True,
                    return_remainder = False)
            else:
                data_quant, data = quantize(
                    data,
                    self.precisions_[i],
                    randomized_rounding = False,
                    return_remainder = True)
            result += self.estimators_[i].estimate(data_quant) / self.global_precisions_[i]
        return result

class TwoMessageRealSum(ManyMessageRealSum):
    def __init__(self, eps, d, n):
        super(self.__class__, self).__init__(eps, d, n, generic_params = True)

class TwoMessageRealSumOptimized(ManyMessageRealSum):
    def __init__(self, eps, d, n):
        super(self.__class__, self).__init__(eps, d, n, generic_params = False)

class ThreeMessageRealSum(ManyMessageRealSum):
    def __init__(self, eps, d, n):
        super(self.__class__, self).__init__(eps, d, n, generic_params = True,
            num_messages = 3)

class ThreeMessageRealSumOptimized(ManyMessageRealSum):
    def __init__(self, eps, d, n):
        super(self.__class__, self).__init__(eps, d, n, generic_params = False,
            num_messages = 3)
