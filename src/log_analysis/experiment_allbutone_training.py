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

    plt.subplots_adjust(wspace=0)
    plt.savefig(outputfile, bbox_inches='tight')


def plot_allbut(allbutlist, train_stat, test_stat, outputfile):
    import matplotlib.pyplot as plt
    withloss = 'loss' in train_stat[0]
    nb_bench = len(allbutlist)
    fig, axs = plt.subplots(2+withloss, nb_bench, sharey='row')
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
