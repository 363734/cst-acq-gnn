from setuptools.command.rotate import rotate

from src.benchmark.load import load_benchmark
from src.log_analysis.load_evaluation_log import load_evaluation_log
from src.log_analysis.load_training_log import load_training_log


def plot_loss(train_stat, outputfile):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 1, sharey='row')
    fig.set_figwidth(10)
    fig.set_figheight(10)

    axs.set_title("loss")
    axs.plot([i for i in range(len(train_stat["loss"]))], train_stat["loss"], label="train loss", linewidth=1)

    plt.savefig(outputfile, bbox_inches='tight')


def plot_loss_multi(train_stat, outputfile):  # TODO finish here
    import matplotlib.pyplot as plt
    nb_runs = len(train_stat)
    fig, axs = plt.subplots(nb_runs, 1, sharey='row', sharex='col')
    fig.set_figwidth(3)
    fig.set_figheight(3 * nb_runs)
    for ax, row in zip(axs, list(range(nb_runs))):
        pad = 45
        ax.annotate(f"Run {row}", xy=(0, 0.5), xytext=(-pad, 0),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline', rotation=90)
    for i in range(nb_runs):
        train = train_stat[i]
        if len(train) > 0:
            axs[i].plot([i for i in range(len(train_stat[i]["loss"]))], train_stat[i]["loss"], label="train loss", linewidth=1)
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(outputfile, bbox_inches='tight')


def plot_train(train_stat, outputfile):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 1, sharey='row')
    fig.set_figwidth(10)
    fig.set_figheight(10)

    axs.set_title("train metrics")
    axs.set_ylim([0, 1])
    axs.bar(["acc", "b_acc", "rec", "tn", "prec", "f1"],
            [train_stat["accuracy"], train_stat["balanced_accuracy"], train_stat["recall"], train_stat["tn"],
             train_stat["precision"], train_stat["f1"]])
    plt.savefig(outputfile, bbox_inches='tight')


def plot_train_multi(train_stat, outputfile):
    import matplotlib.pyplot as plt
    nb_runs = len(train_stat)
    fig, axs = plt.subplots(nb_runs + 1, 1, sharey='row', sharex='col')
    fig.set_figwidth(3)
    fig.set_figheight(3 * (nb_runs + 1))
    for ax, row in zip(axs, list(range(nb_runs)) + ["all"]):
        pad = 30
        ax.annotate(f"Run {row}", xy=(0, 0.5), xytext=(-pad, 0),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline', rotation=90)
    d = {"accuracy": 0, "balanced_accuracy": 0, "recall": 0, "tn": 0, "precision": 0, "f1": 0}
    n = 0
    for i in range(nb_runs):
        train = train_stat[i]
        if len(train) > 0:
            n += 1
            d["accuracy"] += train["accuracy"]
            d["balanced_accuracy"] += train["balanced_accuracy"]
            d["recall"] += train["recall"]
            d["tn"] += train["tn"]
            d["precision"] += train["precision"]
            d["f1"] += train["f1"]
            axs[i].set_ylim([0, 1])
            axs[i].bar(["acc", "b_acc", "rec", "tn", "prec", "f1"],
                       [train["accuracy"], train["balanced_accuracy"], train["recall"], train["tn"],
                        train["precision"], train["f1"]])
    axs[nb_runs].set_ylim([0, 1])
    axs[nb_runs].bar(["acc", "b_acc", "rec", "tn", "prec", "f1"],
                     [d["accuracy"] / n, d["balanced_accuracy"] / n, d["recall"] / n,
                      d["tn"] / n, d["precision"] / n, d["f1"] / n])
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(outputfile, bbox_inches='tight')


def plot_eval(test_stat, outputfile):
    listofbench = [l for l in test_stat if type(test_stat[l]) is dict if l != "all"]
    listofbench = sorted(listofbench)
    import matplotlib.pyplot as plt
    alllist = ["all"] + listofbench
    nb_bench = len(alllist)
    fig, axs = plt.subplots(1, nb_bench, sharey='row')
    fig.set_figwidth(5 * nb_bench)
    fig.set_figheight(5)
    for ax, col in zip(axs, alllist):
        pad = 20
        ax.annotate(f"Bench {col}", xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')
    for i in range(nb_bench):
        b = alllist[i]
        test = test_stat[b]
        print(b)
        axs[i].set_ylim([0, 1])
        axs[i].bar(["acc", "b_acc", "rec", "tn", "prec", "f1"],
                   [test["accuracy"], test["balanced_accuracy"], test["recall"],
                    test["tn"], test["precision"], test["f1"]])

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(outputfile, bbox_inches='tight')


def plot_eval_multi(test_stat, outputfile):
    listofbench = list(set([l for t_s in test_stat for l in t_s if type(t_s[l]) is dict if l != "all"]))
    listofbench = sorted(listofbench)
    import matplotlib.pyplot as plt
    alllist = ["all"] + listofbench
    nb_bench = len(alllist)
    nb_runs = len(test_stat)
    fig, axs = plt.subplots(nb_runs + 1, nb_bench, sharey='row', sharex='col')
    fig.set_figwidth(3 * nb_bench)
    fig.set_figheight(3 * (nb_runs + 1))
    for ax, col in zip(axs[0], alllist):
        pad = 20
        ax.annotate(f"Bench {col}", xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')
    for ax, row in zip([axs[i, 0] for i in range(len(axs))], list(range(nb_runs)) + ["all"]):
        pad = 30
        ax.annotate(f"Run {row}", xy=(0, 0.5), xytext=(-pad, 0),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline', rotation=90)
    for i in range(nb_bench):
        b = alllist[i]
        d = {"accuracy": 0, "balanced_accuracy": 0, "recall": 0, "tn": 0, "precision": 0, "f1": 0}
        n = 0
        for j in range(nb_runs):
            if b in test_stat[j]:
                test = test_stat[j][b]
                n += 1
                d["accuracy"] += test["accuracy"]
                d["balanced_accuracy"] += test["balanced_accuracy"]
                d["recall"] += test["recall"]
                d["tn"] += test["tn"]
                d["precision"] += test["precision"]
                d["f1"] += test["f1"]
                axs[j, i].set_ylim([0, 1])
                axs[j, i].bar(["acc", "b_acc", "rec", "tn", "prec", "f1"],
                              [test["accuracy"], test["balanced_accuracy"], test["recall"],
                               test["tn"], test["precision"], test["f1"]])
        axs[nb_runs, i].set_ylim([0, 1])
        axs[nb_runs, i].bar(["acc", "b_acc", "rec", "tn", "prec", "f1"],
                            [d["accuracy"] / n, d["balanced_accuracy"] / n, d["recall"] / n,
                             d["tn"] / n, d["precision"] / n, d["f1"] / n])
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(outputfile, bbox_inches='tight')


def plot_allbut(allbutlist, train_stat, test_stat, outputfile):
    import matplotlib.pyplot as plt
    withloss = 'loss' in train_stat[0]
    nb_bench = len(allbutlist)
    fig, axs = plt.subplots(2 + withloss, nb_bench, sharey='row')
    fig.set_figwidth(5 * nb_bench)
    fig.set_figheight(5 * (2 + withloss))
    for ax, col in zip(axs[0], allbutlist):
        pad = 20
        ax.annotate(f"All but {col}", xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')
    for i in range(nb_bench):
        b = allbutlist[i]
        train = train_stat[i]
        test = test_stat[i]
        print(b)
        j = 0
        if withloss:
            axs[j, i].set_title("loss")
            axs[j, i].plot([i for i in range(len(train["loss"]))], train["loss"], label="train loss", linewidth=1)
            j += 1

        axs[j, i].set_title("train metrics")
        axs[j, i].set_ylim([0, 1])
        axs[j, i].bar(["acc", "b_acc", "rec", "tn", "prec", "f1"],
                      [train["accuracy"], train["balanced_accuracy"], train["recall"], train["tn"], train["precision"],
                       train["f1"]])

        j += 1
        axs[j, i].set_title("test metrics")
        axs[j, i].set_ylim([0, 1])
        axs[j, i].bar(["acc", "b_acc", "rec", "tn", "prec", "f1"],
                      [test['all']["accuracy"], test['all']["balanced_accuracy"], test['all']["recall"],
                       test['all']["tn"], test['all']["precision"], test['all']["f1"]])

    plt.subplots_adjust(wspace=0)
    plt.savefig(outputfile, bbox_inches='tight')


if __name__ == "__main__":
    algo = "nn"

    # bench = load_benchmark("../../target/benchmarks/training_sets/classical_CA/training_set_classical_ca.txt")
    # train_pattern = "../../target/logs/experimentallbutone_nn/training_model_nn_allbut_{}.txt"
    # test_pattern = "../../target/logs/experimentallbutone_nn/testing_model_nn_allbut_{}.txt"
    # path_image = "../../target/logs/experimentallbutone_nn/plot.pdf"
    # train_stat = [load_training_log(train_pattern.format(b[1])) for b in bench]
    # test_stat = [load_evaluation_log(test_pattern.format(b[1])) for b in bench]
    # print(train_stat)
    # print(test_stat)
    # plot_allbut([b[1] for b in bench], train_stat, test_stat, path_image)
