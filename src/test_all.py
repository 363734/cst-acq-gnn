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

    # benchs = ["sudoku9", "sudoku4"]
    # test_bench = "sudoku16"
    benchs = ["nurse_2_7_15_5", "nurse_3_7_15_5"]
    test_bench = "nurse_4_7_20_5"

    gamma = ["var1 == var2", "var1 != var2", "var1 < var2", "var1 > var2", "var1 >= var2", "var1 <= var2"]

    batch = []
    for bench in benchs:
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

        senarios = generate_senarios_pourcent(2, len(gt), 0.1) \
                   + generate_senarios_pourcent(3, len(gt), 0.3) \
                   + generate_senarios_pourcent(3, len(gt), 0.5) \
                   + generate_senarios_pourcent(3, len(gt), 0.75)
        indicators = [get_indicator(gt, s) for s in senarios]
        # print(indicators)

        updated_g = [switch_senario(g, indicators[i]) for i in range(len(senarios))]

        # print(updated_g)

        sub_batch = [(g, gt)] #+ [(u_g, gt) for u_g in updated_g]
        for (i, j) in sub_batch:
            choose_feats(i, opts)
        batch += sub_batch

    f_test, gt_test = load_bench(test_bench)
    g_test = create_graph_unknown(gamma, f_test)
    senarios_test = generate_senarios_pourcent(1, len(gt_test), 0.1) \
               + generate_senarios_pourcent(1, len(gt_test), 0.2) \
               + generate_senarios_pourcent(1, len(gt_test), 0.3) \
               + generate_senarios_pourcent(1, len(gt_test), 0.4)\
               + generate_senarios_pourcent(1, len(gt_test), 0.5)\
               + generate_senarios_pourcent(1, len(gt_test), 0.6)\
               + generate_senarios_pourcent(1, len(gt_test), 0.7)\
               + generate_senarios_pourcent(1, len(gt_test), 0.8)\
               + generate_senarios_pourcent(1, len(gt_test), 0.9)
    indicators_test = [get_indicator(gt_test, s) for s in senarios_test]
    updated_g_test = [switch_senario(g_test, indicators_test[i]) for i in range(len(senarios_test))]

    test_graph = [(g, gt)] #+ [(u_g, gt) for u_g in updated_g_test]
    for (i, j) in test_graph:
        choose_feats(i, opts)
    test_graph = [test_graph[0]]

    stats = training(opts, batch, test_graph)

    results_training_graph(stats, opts, "out_all_{}.pdf".format(test_bench))
