import numpy as np
import scipy.stats as stats


class LogNormal(object):
    '''
    https://en.wikipedia.org/wiki/Log-normal_distribution
    '''
    def __init__(self, mu, sigma, contamination_level=-1, random_state=0):
        self.mu = mu
        self.sigma = sigma
        self.mean = np.exp(mu + sigma ** 2 / 2)
        self.variance = (np.exp(sigma ** 2) - 1) * np.exp(2 * mu + sigma ** 2)
        self.random_state = random_state
        self._random_state = random_state
        self.contamination_level = contamination_level
        if self.contamination_level > 0:
            self.upper_tail = stats.lognorm.ppf((1-contamination_level/2), s=sigma, scale=np.exp(mu))
            self.lower_tail = stats.lognorm.ppf(contamination_level/2, s=sigma, scale=np.exp(mu))

    def reset(self,):
        self._random_state = self.random_state

    def sample(self, N, D):
        np.random.seed(self._random_state)
        self._random_state += 1
        samples = np.random.lognormal(mean=self.mu, sigma=self.sigma, size=(N, D))
        
        if self.contamination_level <= 0:
            return samples

        contamination_type = np.random.randint(low=0, high=3, size=1)[0]
        if contamination_type == 0:
            samples = np.clip(samples, a_min=self.lower_tail, a_max=None)
        else:
            samples = np.clip(samples, a_max=self.upper_tail, a_min=None)
        return samples
        


class Burr(object):
    '''
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.burr.html#scipy.stats.burr
    '''
    def __init__(self, c, d, contamination_level=-1, random_state=0):
        self.c = c
        self.d = d
        self.mean = float(stats.burr.stats(c, d, moments='m'))
        self.random_state = random_state
        self._random_state = random_state
        self.contamination_level = contamination_level
        if self.contamination_level > 0:
            self.upper_tail = stats.burr.ppf((1-contamination_level/2), c=self.c, d=self.d)
            self.lower_tail = stats.burr.ppf(contamination_level/2, c=self.c, d=self.d)

    def reset(self,):
        self._random_state = self.random_state

    def sample(self, N, D):
        np.random.seed(self._random_state)
        self._random_state += 1
        samples = np.random.uniform(0, 1, size=(N, D))
        samples = stats.burr.ppf(samples, c=self.c, d=self.d)

        if self.contamination_level <= 0:
            return samples
        
        contamination_type = np.random.randint(low=0, high=3, size=1)[0]
        if contamination_type == 0:
            samples = np.clip(samples, a_min=self.lower_tail, a_max=None)
        else:
            samples = np.clip(samples, a_max=self.upper_tail, a_min=None)
        return samples

