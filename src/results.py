
import numpy as np

# def results_training_graph(stats):
def results_training_graph(stats,opts, outputfile:str):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 1, sharex='col')
    fig.suptitle(opts)
    fig.set_figwidth(10)
    fig.set_figheight(10)
    print(stats["nb-epoch"])
    x = list(range(0, stats["nb-epoch"]))
    axs[0].set_yscale('log')
    axs[0].set_title("loss")
    axs[0].plot(x, [stats[i]["train-loss"] for i in x], label="train loss", linewidth=1)
    # axs[0].plot(x, [stats[i]["test-loss"] for i in x], label="test loss", linewidth=1)
    axs[0].grid()
    axs[0].legend()
    axs[1].set_title("train metrics")
    axs[1].set_ylim([-0.01, 1.01])
    axs[1].set_yticks(np.arange(0, 1.1, step=0.1))
    axs[1].plot(x, [stats[i]["train"]["accuracy"] for i in x], label="accuracy", linewidth=1)
    axs[1].plot(x, [stats[i]["train"]["balanced_accuracy"] for i in x], label="balanced accuracy", linewidth=1)
    axs[1].plot(x, [stats[i]["train"]["recall"] for i in x], label="recall", linewidth=1)
    axs[1].plot(x, [stats[i]["train"]["tn"] for i in x], label="tn", linewidth=1)
    axs[1].plot(x, [stats[i]["train"]["f1"] for i in x], label="f1", linewidth=1)
    axs[1].plot(x, [stats[i]["train"]["precision"] for i in x], label="precision", linewidth=1)
    axs[1].grid()
    axs[1].legend()
    axs[2].set_title("test metrics")
    axs[2].set_ylim([-0.01, 1.01])
    axs[2].set_yticks(np.arange(0, 1.1, step=0.1))
    axs[2].plot(x, [stats[i]["test"]["accuracy"] for i in x], label="accuracy", linewidth=1)
    axs[2].plot(x, [stats[i]["test"]["balanced_accuracy"] for i in x], label="balanced accuracy", linewidth=1)
    axs[2].plot(x, [stats[i]["test"]["recall"] for i in x], label="recall", linewidth=1)
    axs[2].plot(x, [stats[i]["test"]["tn"] for i in x], label="tn", linewidth=1)
    axs[2].plot(x, [stats[i]["test"]["f1"] for i in x], label="f1", linewidth=1)
    axs[2].plot(x, [stats[i]["test"]["precision"] for i in x], label="precision", linewidth=1)
    axs[2].grid()
    axs[2].legend()

    plt.savefig(outputfile, bbox_inches='tight')

