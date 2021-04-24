import torch
import numpy as np
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from models import evaluate, CatoniGiulini, CoordTruncMeans, HDMoM, LogNormal, Burr
import matplotlib.cm as cm


@torch.no_grad()
def high_dimensional_mean_estimation(distribution_name):
    #### set up constants, parameters, and models
    delta = 0.05
    N = 20000
    dimensions = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    data_generators = {'LogNormal': LogNormal(3, 1, random_state=1001), 'Burr': Burr(1.2, 10, random_state=1001)}

    model_names = ['CatoniGiulini two-phase', 'CatoniGiulini one-phase', 'Coordinate-wise Truncated Mean', 'HDMoM geometric median', 'HDMoM coordinative-wise median']
    models = [CatoniGiulini(delta, True), CatoniGiulini(delta, False), CoordTruncMeans(delta), HDMoM(delta, True), HDMoM(delta, False)]
    experimental_results = {model: [] for model in model_names}

    #### conduct experiments
    if True:
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
    
    scatters = []
    for i, model_name in enumerate(model_names):
        l_i = plt.scatter(dimensions, experimental_results[model_name], color=colors[i])
        scatters.append(l_i)
    plt.legend(scatters, model_names)
    plt.xlabel('Dimension')
    plt.ylabel('Error')
    plt.savefig('./figures/%sAll.pdf' % distribution_name, dpi=300)
    plt.cla()


    ## plot only trimmed mean based methods
    plt.rcParams["font.family"] = "Times New Roman"
    plt.xscale('log', base=2)
    scatters = []
    for i, model_name in enumerate(model_names[:3]):
        l_i = plt.scatter(dimensions, experimental_results[model_name], color=colors[i])
        scatters.append(l_i)
        
    plt.legend(scatters, model_names[:3])
    plt.xlabel('Dimension')
    plt.ylabel('Error')
    plt.savefig('./figures/%sTrimmed.pdf' % distribution_name, dpi=300)
    plt.cla()

    ## plot only MoM based methods
    plt.rcParams["font.family"] = "Times New Roman"
    plt.xscale('log', base=2)
    scatters = []
    for i, model_name in enumerate(model_names[3:]):
        i += 3
        l_i = plt.scatter(dimensions, experimental_results[model_name], color=colors[i])
        scatters.append(l_i)
        
    plt.legend(scatters, model_names[3:])
    plt.xlabel('Dimension')
    plt.ylabel('Error')
    plt.savefig('./figures/%sMoM.pdf' % distribution_name, dpi=300)
    plt.cla()

if __name__ == '__main__':
    high_dimensional_mean_estimation('Burr')