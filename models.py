import math
import torch
import scipy.stats as stats
import numpy as np
import hdmedians as hd


class HDMoM(object):
    def __init__(self, delta, geometric_median=True):
        self.delta = delta
        self.k = math.ceil(8 * np.log(1 / self.delta))
        self.geometric_median = geometric_median

    def estimate(self, X, device='cpu'):
        '''
        Arguments
            X: shape = [N, D]
        '''
        X = X.clone().to(device)
        N, D = X.shape
        block_size = N // self.k
        assert block_size >= 2

        mean_values = []
        for i in range(self.k):
            if i != self.k - 1:
                x_mean_i = torch.mean(X[i * block_size: (i+1) * block_size], axis=0)
            else:
                x_mean_i = torch.mean(X[i * block_size:], axis=0)
            mean_values.append(x_mean_i)

        mean_values = torch.stack(mean_values, axis=0)  # shape = [k, D]
        if self.geometric_median:
            mean_values = mean_values.cpu().numpy()
            x_mean = hd.geomedian(mean_values, axis=0)  # shape = [D]
            x_mean = torch.tensor(x_mean).to(device)
        else:
            x_mean, _ = torch.median(mean_values, axis=0)  # shape = [D]
        return x_mean


class CoordTruncMeans(object):
    def __init__(self, delta):
        self.delta = delta

    def estimate(self, X, device='cpu'):
        X = X.clone().to(device)
        N, D = X.shape
        
        epsilon = 16 * np.log(8/self.delta) / (3 * N)
        assert self.delta >= 8 * np.exp(-3 * N / 16)
        
        ## split the dataset into two subsets
        X = X[torch.randperm(N)]
        N0 = N//2
        X0 = X[:N0] # shape = [N0, D]
        X1 = X[N0:] # shape = [N1, D]

        ## use X0 to find alpha and beta
        sorted_X0, _ = torch.sort(X0, dim=0) # shape = [N0, D]
        alpha = sorted_X0[int(epsilon * N0), :].unsqueeze(0) # shape = [1, D]
        beta = sorted_X0[int((1-epsilon) * N0), :].unsqueeze(0) # shape = [1, D]
        
        ## use X1 to find mean
        clipped_X1 = torch.max(torch.min(X1, beta), alpha) # shape = [D]
        x_mean = torch.mean(clipped_X1, dim=0) # shape = [D]
        return x_mean


class CatoniGiulini(object):
    def __init__(self, delta, two_phase):
        self.delta = delta
        self.mu = 1 / 4
        self.two_phase = two_phase

    def estimate(self, X, device='cpu'):
        '''
        Arguments
            X: shape = [N, D]
        '''
        X = X.clone().to(device)
        N, D = X.shape
        if self.two_phase:
            '''
            A two phase method 
            1) apply MoM on a portion of the data to find a rough mean value, then center the rest of the data
            2) apply Catoni-Giulini for mean estimation
            '''
            ## randomly split the data into two subsets
            N0 = int(np.sqrt(N))
            X = X[torch.randperm(N)]
            X0 = X[:N0]
            X1 = X[N0:]
            
            ## apply Median of Mean to find a rough mean value, and center the other subset
            estimator = HDMoM(self.delta, geometric_median=True)
            x_mean_0 = estimator.estimate(X0, device=device) # shape = [D]
            X1 = X1 - x_mean_0.unsqueeze(0) # shape = [N, D]
            
            ## apply Catoni-Giulini
            x_mean_1 = self.catoni_giulini(X1)
            x_mean = x_mean_0 + x_mean_1
        else:
            '''
            directly apply Catoni-Giulini
            '''
            x_mean = self.catoni_giulini(X)
        return x_mean

    def catoni_giulini(self, X):
        '''
        Arguments
            X: shape = [N, D]
        '''
        N, D = X.shape

        ## find the spectral norm of the data
        v = self._spectral_norm(X)

        lambda_ = self._lambda(N, v)
        
        x_norm = torch.norm(X, p=2, dim=-1) # shape = [N]
        tmp_coefs = self._psi(x_norm * lambda_) / (x_norm * lambda_) # shape = [N]
        Y = tmp_coefs.unsqueeze(-1) * X # shape = [N, D]
        
        y_mean = torch.mean(Y, dim=0) # shape = [D]
        return y_mean

    def _spectral_norm(self, X):
        '''
        return the largest eigenvalue of X.T @ X
        '''
        x_mean = torch.mean(X, dim=0, keepdim=True) # shape = [1, D]
        X = X - x_mean # shape = [N, D]
        eig_vals, _ = torch.eig(X.T @ X)
        eig_vals = eig_vals[:, 0]
        v = float(max(eig_vals))
        return v

    def _lambda(self, N, v):
        a = self._g2(2 * self.mu)
        lambda_ = 1 / self.mu * np.sqrt(2 * np.log(1 / self.delta) / (a * v * N))
        return lambda_

    def _g2(self, x):
        return 2 / (x ** 2) * (np.exp(x) - 1 - x)

    def _psi(self, x):
        return torch.clamp(x, min=None, max=1)


class LogNormal(object):
    '''
    https://en.wikipedia.org/wiki/Log-normal_distribution
    '''
    def __init__(self, mu, sigma, random_state=0):
        self.mu = mu
        self.sigma = sigma
        self.mean = np.exp(mu + sigma ** 2 / 2)
        self.variance = (np.exp(sigma ** 2) - 1) * np.exp(2 * mu + sigma ** 2)
        self.random_state = random_state
        self._random_state = random_state

    def reset(self,):
        self._random_state = self.random_state

    def sample(self, N, D):
        np.random.seed(self._random_state)
        self._random_state += 1
        return np.random.lognormal(mean=self.mu, sigma=self.sigma, size=(N, D))


class Burr(object):
    '''
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.burr.html#scipy.stats.burr
    '''
    def __init__(self, c, d, random_state=0):
        self.c = c
        self.d = d
        self.mean = float(stats.burr.stats(c, d, moments='m'))
        self.random_state = random_state
        self._random_state = random_state

    def reset(self,):
        self._random_state = self.random_state

    def sample(self, N, D):
        np.random.seed(self._random_state)
        self._random_state += 1
        samples = np.random.uniform(0, 1, size=(N, D))
        return stats.burr.ppf(samples, c=self.c, d=self.d)


def euclidean_distance(x, y):
    return torch.sqrt(torch.sum((x - y) ** 2))


def evaluate(data_generator, estimator, N, D, verbose=False):
    '''
    use the average of the Euclidean distance as the evaluation metric
    '''
    device = 'cuda:1'
    epoch = 100  # number of experiments
    mu = torch.ones((D), device=device) * data_generator.mean
    
    if verbose:
        print('True Mean %.5f' % data_generator.mean)

    acc_euc_dis = 0
    for _ in range(epoch):
        data = data_generator.sample(N, D)
        data = torch.from_numpy(data)
        mu_hat = estimator.estimate(data, device=device)
        error = euclidean_distance(mu, mu_hat)
        acc_euc_dis += float(error)

    acc_euc_dis /= epoch

    if verbose:
        print('Average ||mu_hat - mu|| (%d runs) %.5f' % (epoch, acc_euc_dis))
    return acc_euc_dis


if __name__ == '__main__':
    # np.random.seed(1000)
    data_generator = LogNormal(3, 1, random_state=1001)
    print(data_generator.mean)
    data_generator = Burr(1.2, 10, random_state=1001)
    print(data_generator.mean)
    estimator = HDMoM(0.01, False)
    # estimator = CoordTruncMeans(0.01)
    # estimator = CatoniGiulini(0.01, two_phase=True)
    # evaluate(data_generator, estimator, 20000, 20)
