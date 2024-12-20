from src.cstAcqAlgos.ConAcq import ConAcq
from src.cstAcqAlgos.MQuAcq import MQuAcq
from src.cstAcqAlgos.MQuAcq2 import MQuAcq2
from src.cstAcqAlgos.QuAcq import QuAcq
from src.cstAcqAlgos.utils import construct_bias_for_var, get_con_subset


class GrowAcq(ConAcq):
    def __init__(self, gamma, grid, ct=list(), bias=list(), X=set(), C_l=set(), qg="pqgen", gqg= False, gfs=False, gfc=False,
                 obj="proba", classifier=None, classifier_name=None, time_limit=None, findscope_version=4, findc_version=1,
                 tqgen_t=None, qgen_blimit=5000, algorithm="mquacq2"):
        super().__init__(gamma, grid, ct, bias, X, C_l, qg, gqg, gfs, gfc, obj, classifier, classifier_name, time_limit, findscope_version,
                    findc_version, tqgen_t, qgen_blimit)
        self.algorithm = algorithm

    def learn(self):

        Y = []

        while True:

            x = self.X.pop()
            B = construct_bias_for_var(Y, self.gamma, x)
            if self.debug_mode:
                print(f"\nAdding variable {x} in GrowAcq")
                print("\nsize of B in growacq: ", len(B))
            Y.append(x)
            # B = get_consubset(self.B, Y)
            C_T = get_con_subset(self.C_T, Y)

            if self.algorithm == "quacq":
                ca = QuAcq(self.gamma, self.grid, C_T, B.copy(), set(Y), self.C_l.constraints, qg=self.qg,
                           gqg=self.gqg, gfs=self.gfs, gfc=self.gfc,
                           obj=self.obj,
                           classifier=self.classifier, classifier_name=self.classifier_name,
                           time_limit=self.time_limit,
                           findscope_version=self.fs, findc_version=self.fc)
            elif self.algorithm == "mquacq":
                ca = MQuAcq(self.gamma, self.grid, C_T, B.copy(), set(Y), self.C_l.constraints, qg=self.qg,
                            gqg=self.gqg, gfs=self.gfs, gfc=self.gfc,
                            obj=self.obj,
                            classifier=self.classifier, classifier_name=self.classifier_name,
                            time_limit=self.time_limit,
                            findscope_version=self.fs, findc_version=self.fc)
            elif self.algorithm == "mquacq2":
                ca = MQuAcq2(self.gamma, self.grid, C_T, B.copy(), set(Y), self.C_l.constraints, qg=self.qg,
                             gqg=self.gqg, gfs=self.gfs, gfc=self.gfc,
                             obj=self.obj,
                             classifier=self.classifier, classifier_name=self.classifier_name,
                             time_limit=self.time_limit,
                             findscope_version=self.fs, findc_version=self.fc,
                             perform_analyzeAndLearn=False)
            elif self.algorithm == "mquacq2-a":
                ca = MQuAcq2(self.gamma, self.grid, C_T, B.copy(), set(Y), self.C_l.constraints, qg=self.qg,
                             gqg=self.gqg, gfs=self.gfs, gfc=self.gfc,
                             obj=self.obj,
                             classifier=self.classifier, classifier_name=self.classifier_name,
                             time_limit=self.time_limit,
                             findscope_version=self.fs, findc_version=self.fc,
                             perform_analyzeAndLearn=True)

            counts, countsB = self.get_counts()
            ca.set_counts(counts, countsB)
            ca.set_dataset(self.dataset_X, self.dataset_Y)
            # TODO pass on the prior to the internal system
            ca.prior = self.prior
            ca.prior_use = self.prior_use
            ca.prior_param = self.prior_param

            if self.classify and len(self.C_l.constraints) > 0:
                ca.train_classifier()

            ca.learn()

            self.metrics += ca.metrics

            counts, countsB = ca.get_counts()
            self.set_counts(counts, countsB)
            self.dataset_X, self.dataset_Y = ca.get_dataset()
            self.C_l = ca.C_l

            if len(self.X) == 0:
                break

 #           if self.debug_mode:
            print("C_L: ", len(self.C_l.constraints))
            print("B: ", len(self.B))
            print("Number of queries: ", self.metrics.queries_count)
            print("Top level Queries: ", self.metrics.top_lvl_queries)
            print("FindScope Queries: ", self.metrics.findscope_queries)
            print("FindC Queries: ", self.metrics.findc_queries)

#        if self.debug_mode:
        print("Converged ------------------------------------")
        print("Number of queries: ", self.metrics.queries_count)
        print("Top level Queries: ", self.metrics.top_lvl_queries)
        print("FindScope Queries: ", self.metrics.findscope_queries)
        print("FindC Queries: ", self.metrics.findc_queries)