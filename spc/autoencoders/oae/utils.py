import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import torch


def estimate_upper_control_limit(data, alpha=0.05, band_width="scott"):
    """
    Estimate upper control limit (UCL) using kernel density estimation (KDE) with a gaussian kernel
    :param data: input numpy array of statistics
    :param alpha: significance level for finding the (1-alpha) percentile
    :param band_width: if "standard" uses elbow rule, otherwise can be a float value to be used as bandwidth
    :return: value of the empirical cumulative distribution function corresponding to (1-alpha)-upper percentile
    """
    bw = sp.stats.gaussian_kde(data, bw_method=band_width).covariance_factor() * data.std()
    return sp.optimize.brentq(lambda x: sum(sp.stats.norm.cdf((x - data) / bw)) / len(data) - (1 - alpha), 0, 100000)


def get_grads(x, trained_model):
    inp = x.unsqueeze(0).requires_grad_(True)
    output = trained_model.encoder(inp)  # trained model should have the encoding layers saved as "encoder"
    gradient = torch.autograd.grad(output.mean(), inp)[0][0].data.numpy()
    return gradient


def integrated_gradients(model, inp, baseline, approximation_method="monte_carlo", steps=20, samples=1000):
    # If baseline is not provided, start with an array of zeros
    if baseline is None:
        baseline = np.zeros(len(inp)).astype(np.float32)
    else:
        baseline = baseline

    baseline = np.array(baseline).reshape(-1, len(inp))
    inp = np.array(inp).reshape(-1, len(inp))

    if approximation_method == "trapezoidal":
        path_inputs = [baseline + (i / steps) * (inp - baseline) for i in range(steps + 1)]
        grads = []
        for elem in path_inputs:
            grad = get_grads(torch.Tensor(elem[0]), model)
            grads.append(np.array(grad))
        delta = 1 / steps
        integrated_grads = delta * np.sum((inp - baseline) * grads, axis=0)

    elif approximation_method == "monte_carlo":
        random_samples = np.random.uniform(0, 1, (samples, len(inp)))
        random_vectors = baseline + (inp - baseline) * random_samples
        grads = []
        for i in range(samples):
            grad = get_grads(torch.Tensor(random_vectors[i, :]), model)
            grads.append(np.array(grad))
        integrated_grads = (inp - baseline) * np.mean(grads, axis=0)

    else:
        raise ValueError(
            "Invalid approximation method: {}, use either monte_carlo or trapezoidal".format(approximation_method))

    return integrated_grads


def plot_multivariate_control_charts(t2_scores, t2_ucl, spe_scores, spe_ucl, log_scale=False):
    if log_scale:
        t2_scores = np.log(t2_scores)
        t2_ucl = np.log(t2_ucl)
        spe_scores = np.log(spe_scores)
        spe_ucl = np.log(spe_ucl)

    fig, axs = plt.subplots(2, 1, figsize=(8, 8))
    fig.subplots_adjust(hspace=0.5, bottom=0.1)
    # T^2 chart
    axs[0].plot(t2_scores, marker='', color='black', linewidth=0.8, alpha=1)
    axs[0].axhline(y=t2_ucl, color='red', label="UCL", alpha=0.6)
    axs[0].grid(False)
    if log_scale:
        axs[0].set_ylabel("$T^2$ statistic (log)")
    else:
        axs[0].set_ylabel("$T^2$ statistic")
    axs[0].set_xlabel("Observations")
    axs[0].legend(loc="upper left")
    axs[0].set_title("$T^2$ Control Chart", loc='center', fontweight=1)

    # SPE chart
    axs[1].plot(spe_scores, marker='', color='black', linewidth=0.8, alpha=1)
    axs[1].axhline(y=spe_ucl, color='red', label="UCL", alpha=0.6)
    axs[1].grid(False)
    if log_scale:
        axs[1].set_ylabel("SPE statistic (log)")
    else:
        axs[1].set_ylabel("SPE statistic")
    axs[1].set_xlabel("Observations")
    axs[1].legend(loc="upper left")
    axs[1].set_title("SPE Control Chart", loc='center', fontweight=1)
    plt.show()
