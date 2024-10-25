import os
import re


def load_evaluation_log(filename):
    d = {}
    if os.path.exists(filename):
        with open(filename) as file:
            lines = list(file.readlines())
            b = ""
            for l in lines:
                if "::- Evaluation All Benchmarks" in l:
                    b = "all"
                    d[b] = {}
                if "::- Evaluation benchmark " in l:
                    b = l.replace("::- Evaluation benchmark ", "").strip()
                    d[b] = {}
                if " accuracy score" in l:
                    d[b]["accuracy"] = float(re.findall(r'\[(.*?)\]', l)[0])
                if "balanced_accuracy score" in l:
                    d[b]["balanced_accuracy"] = float(re.findall(r'\[(.*?)\]', l)[0])
                if "recall score" in l:
                    d[b]["recall"] = float(re.findall(r'\[(.*?)\]', l)[0])
                if " tn score" in l:
                    d[b]["tn"] = float(re.findall(r'\[(.*?)\]', l)[0])
                if "precision score" in l:
                    d[b]["precision"] = float(re.findall(r'\[(.*?)\]', l)[0])
                if "f1 score" in l:
                    d[b]["f1"] = float(re.findall(r'\[(.*?)\]', l)[0])
                if "nb_pos score" in l:
                    d[b]["nb_pos"] = int(re.findall(r'\[(.*?)\]', l)[0])
                if "nb_tp score" in l:
                    d[b]["nb_tp"] = int(re.findall(r'\[(.*?)\]', l)[0])
                if "nb_fn score" in l:
                    d[b]["nb_fn"] = int(re.findall(r'\[(.*?)\]', l)[0])
                if "nb_neg score" in l:
                    d[b]["nb_neg"] = int(re.findall(r'\[(.*?)\]', l)[0])
                if "nb_tn score" in l:
                    d[b]["nb_tn"] = int(re.findall(r'\[(.*?)\]', l)[0])
                if "nb_fp score" in l:
                    d[b]["nb_fp"] = int(re.findall(r'\[(.*?)\]', l)[0])
                if "Total evaluation time" in l:
                    d["total evaluation time"] = float(re.findall(r'\[(.*?)\]', l)[0])
                if "Mean evaluation time" in l:
                    d["mean evaluation time"] = float(re.findall(r'\[(.*?)\]', l)[0])
    else:
        print("no file: {}".format(filename))
    return d


if __name__ == "__main__":
    d = load_evaluation_log("../../target/logs/experimentallbutone_nn/testing_model_nn_allbut_sudoku_9.txt")
    print(d)
