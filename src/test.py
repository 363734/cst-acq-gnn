from src.bench_loader import load_bench
from src.generate_training import generate_senarios_pourcent, get_indicator
from src.graph import create_graph_unknown, switch_senario, choose_feats
from src.opt import parser
from src.results import results_training_graph
from src.training import training

if __name__ == "__main__":
    (opts, args) = parser.parse_args()
    print('test')
    print(args)
    print(opts)

    bench = "time"

    gamma = ["var1 == var2", "var1 != var2", "var1 < var2", "var1 > var2", "var1 >= var2", "var1 <= var2"]

    f, gt = load_bench(bench)
    # for f in [f1b, f2b, f3b]:
    print('----------')
    print(f)
    print(len(f))
    print(len(gt))
    # print(f)
    ct= 0
    for i in range(len(f)):
        if gt[i] == 1:
            print(f[i])
            ct+=1
    print(ct)

    g = create_graph_unknown(gamma, f)
    print(g)
    print(g.nodes['cst'].data)

    senarios = generate_senarios_pourcent(9, len(gt), 0.1) \
               + generate_senarios_pourcent(10, len(gt), 0.3) \
               + generate_senarios_pourcent(10, len(gt), 0.5) \
               + generate_senarios_pourcent(10, len(gt), 0.75)
    indicators = [get_indicator(gt, s) for s in senarios]
    # print(indicators)

    updated_g = [switch_senario(g, indicators[i]) for i in range(len(senarios))]

    # print(updated_g)

    batch = [(g, gt)] + [(u_g, gt) for u_g in updated_g]
    for (i, j) in batch:
        choose_feats(i)

    senarios_test = generate_senarios_pourcent(1, len(gt), 0.1) \
               + generate_senarios_pourcent(1, len(gt), 0.2) \
               + generate_senarios_pourcent(1, len(gt), 0.3) \
               + generate_senarios_pourcent(1, len(gt), 0.4)\
               + generate_senarios_pourcent(1, len(gt), 0.5)\
               + generate_senarios_pourcent(1, len(gt), 0.6)\
               + generate_senarios_pourcent(1, len(gt), 0.7)\
               + generate_senarios_pourcent(1, len(gt), 0.8)\
               + generate_senarios_pourcent(1, len(gt), 0.9)
    indicators_test = [get_indicator(gt, s) for s in senarios_test]
    updated_g_test = [switch_senario(g, indicators_test[i]) for i in range(len(senarios_test))]

    test_graph = [batch[0]] + [(u_g, gt) for u_g in updated_g_test]
    test_graph = [test_graph[0]]

    stats = training(opts, batch, test_graph)

    results_training_graph(stats, opts, "out_{}.pdf".format(bench))
