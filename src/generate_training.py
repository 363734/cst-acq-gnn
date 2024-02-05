import random


# output a list of evaluated or not for each bias constraint
def generate_senarios_random(nb_senarios, bias_length):
    return [[random.randint(0, 1) for i in range(bias_length)] for j in range(nb_senarios)]


def generate_senarios_pourcent(nb_senarios, bias_length, pourcent):
    stop = bias_length * pourcent
    l = [[1 if i < stop else 0 for i in range(bias_length)] for j in range(nb_senarios)]
    for sl in l:
        random.shuffle(sl)
    return l


# return the unknown/yes/no features
def get_indicator(ground_truth, senario):
    unknown = [1 - i for i in senario]
    yes = [senario[i] * ground_truth[i] for i in range(len(ground_truth))]
    no = [senario[i] * (1 - ground_truth[i]) for i in range(len(ground_truth))]
    return unknown, yes, no


if __name__ == "__main__":
    print(generate_senarios_random(10, 10))
    print(generate_senarios_pourcent(10, 10, 0.1))
    s = generate_senarios_pourcent(1, 10, 0.4)[0]
    gt = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    print(get_indicator(gt, s))
