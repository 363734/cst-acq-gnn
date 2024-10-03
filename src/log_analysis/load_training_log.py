import re


def load_training_log(filename):
    d = {}
    with open(filename) as file:
        lines = list(file.readlines())
        for l in lines:
            if " accuracy score" in l:
                d["accuracy"] = float(re.findall(r'\[(.*?)\]', l)[0])
            if "balanced_accuracy score" in l:
                d["balanced_accuracy"] = float(re.findall(r'\[(.*?)\]', l)[0])
            if "recall score" in l:
                d["recall"] = float(re.findall(r'\[(.*?)\]', l)[0])
            if " tn score" in l:
                d["tn"] = float(re.findall(r'\[(.*?)\]', l)[0])
            if "precision score" in l:
                d["precision"] = float(re.findall(r'\[(.*?)\]', l)[0])
            if "f1 score" in l:
                d["f1"] = float(re.findall(r'\[(.*?)\]', l)[0])
            if "nb_pos score" in l:
                d["nb_pos"] = int(re.findall(r'\[(.*?)\]', l)[0])
            if "nb_tp score" in l:
                d["nb_tp"] = int(re.findall(r'\[(.*?)\]', l)[0])
            if "nb_fn score" in l:
                d["nb_fn"] = int(re.findall(r'\[(.*?)\]', l)[0])
            if "nb_neg score" in l:
                d["nb_neg"] = int(re.findall(r'\[(.*?)\]', l)[0])
            if "nb_tn score" in l:
                d["nb_tn"] = int(re.findall(r'\[(.*?)\]', l)[0])
            if "nb_fp score" in l:
                d["nb_fp"] = int(re.findall(r'\[(.*?)\]', l)[0])
            if "\tLoss:" in l:
                d["loss"] = eval(l.strip().split(":")[-1])
            if "training time" in l:
                d["training time"] = float(re.findall(r'\[(.*?)\]', l)[0])
    return d


if __name__ == "__main__":
    d = load_training_log("../../target/logs/experimentallbutone_nn/training_model_nn_allbut_sudoku_9.txt")
    print(d)
