import numpy as np
import matplotlib.pyplot as plt


def plot_learning_curve(x, scores, filename, lines=None):
    maddpg_scores = scores

    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")

    N = len(maddpg_scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(
                maddpg_scores[max(0, t-100):(t+1)])

    ax.plot(x, running_avg, color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("MADDPG Score", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)