import torch
import numpy as np
import json
import scipy.stats as stats
from tqdm import tqdm
import matplotlib.pyplot as plt
from models import evaluate, CatoniGiulini, CoordTruncMeans, HDMoM, LogNormal, Burr, EmpiricalMean
import matplotlib.cm as cm


@torch.no_grad()
def high_dimensional_mean_estimation(distribution_name):
    #### set up constants, parameters, and models
    delta = 0.05
    N = 20000
    dimensions = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    data_generators = {'LogNormal': LogNormal(3, 1, random_state=1001), 'Burr': Burr(1.2, 10, random_state=1001)}

    model_names = ['EmpiricalMean', 'CatoniGiulini two-phase', 'CatoniGiulini one-phase', 'Coordinate-wise truncated mean', 'HDMoM geometric median', 'HDMoM coordinative-wise median']
    models = [EmpiricalMean(), CatoniGiulini(delta, True), CatoniGiulini(delta, False), CoordTruncMeans(delta), HDMoM(delta, True), HDMoM(delta, False)]
    model2markers = {'EmpiricalMean':'o', 'CatoniGiulini two-phase':'+', 'CatoniGiulini one-phase':'+', 'Coordinate-wise truncated mean':'+', 'HDMoM geometric median':'x', 'HDMoM coordinative-wise median':'x'}
    # markers = ['.', '+', '+', '+', 'x', 'x']
    experimental_results = {model: [] for model in model_names}

    #### conduct experiments
    if False:
        data_generator = data_generators[distribution_name]
        for model, model_name in tqdm(zip(models, model_names), total=len(models)):
            for D in dimensions:
                data_generator.reset()
                error = evaluate(data_generator, model, N, D)
                experimental_results[model_name].append(error)
        with open('./%sOut.json' % distribution_name, 'w') as f:
            json.dump(experimental_results, f)
    else:
        with open('./%sOut.json' % distribution_name, 'r') as f:
            experimental_results = json.load(f)
    
    #### plots
    ## plot all methods
    plt.rcParams["font.family"] = "Times New Roman"
    plt.xscale('log', base=2)
    colors = cm.rainbow(np.linspace(0, 1, len(models)))
    
    marker_size = 20

    scatters = []
    for i, model_name in enumerate(model_names):
        l_i = plt.scatter(dimensions, experimental_results[model_name], color=colors[i], s=marker_size, marker=model2markers[model_name])
        scatters.append(l_i)
    plt.legend(scatters, model_names)
    plt.xlabel('Dimension')
    plt.ylabel('Error')
    if distribution_name == 'LogNormal':
        plt.ylim(0, 50)
    plt.savefig('./figures/%sAll.pdf' % distribution_name, dpi=300)
    plt.cla()


    ## plot only trimmed mean based methods
    plt.rcParams["font.family"] = "Times New Roman"
    plt.xscale('log', base=2)
    scatters = []
    TrimmedMeanMethods = ['EmpiricalMean', 'CatoniGiulini two-phase', 'CatoniGiulini one-phase', 'Coordinate-wise truncated mean']
    for i, model_name in enumerate(TrimmedMeanMethods):
        l_i = plt.scatter(dimensions, experimental_results[model_name], color=colors[i], s=marker_size, marker=model2markers[model_name])
        scatters.append(l_i)
        
    plt.legend(scatters, TrimmedMeanMethods)
    plt.xlabel('Dimension')
    plt.ylabel('Error')
    if distribution_name == 'LogNormal':
        plt.ylim(0, 50)
    plt.savefig('./figures/%sTrimmed.pdf' % distribution_name, dpi=300)
    plt.cla()

    ## plot only MoM based methods
    plt.rcParams["font.family"] = "Times New Roman"
    plt.xscale('log', base=2)
    scatters = []
    HDMoMMethods = ['EmpiricalMean', 'HDMoM geometric median', 'HDMoM coordinative-wise median']
    for i, model_name in enumerate(HDMoMMethods):
        i += 3
        l_i = plt.scatter(dimensions, experimental_results[model_name], color=colors[i], s=marker_size, marker=model2markers[model_name])
        scatters.append(l_i)
        
    plt.legend(scatters, HDMoMMethods)
    plt.xlabel('Dimension')
    plt.ylabel('Error')
    if distribution_name == 'LogNormal':
        plt.ylim(0, 50)
    plt.savefig('./figures/%sMoM.pdf' % distribution_name, dpi=300)
    plt.cla()

def plot_distributions():
    ## plot burr
    burr = Burr(1.2, 10, random_state=1001)
    fig, ax = plt.subplots()
    rnge = (0, 40)
    x = np.linspace(rnge[0], rnge[1], 1001)
    ax.plot(x, stats.burr.pdf(x, c=burr.c,d=burr.d) / stats.burr.cdf(rnge[1], c=burr.c,d=burr.d), 'r-', lw=2, alpha=0.6)
    plt.xlabel('x')
    plt.ylabel('y')
    fig.savefig('./figures/burr.pdf', dpi=300)
    ax.cla()

    log_normal = LogNormal(3, 1, random_state=1001)
    fig, ax = plt.subplots()
    rnge = (0, 80)
    x = np.linspace(rnge[0], rnge[1], 1001)
    ax.plot(x, stats.lognorm.pdf(x, s=log_normal.sigma, scale=np.exp(log_normal.mu)) / stats.lognorm.cdf(rnge[1], s=log_normal.sigma,scale=np.exp(log_normal.mu)), 'r-', lw=2, alpha=0.6)
    plt.xlabel('x')
    plt.ylabel('y')
    fig.savefig('./figures/lognormal.pdf', dpi=300)
    ax.cla()
    print(burr.mean, log_normal.mean)


if __name__ == '__main__':
    high_dimensional_mean_estimation('Burr')
    # plot_distributions()