import time

from src.cstAcqAlgos.ConAcq import ConAcq
from src.cstAcqAlgos.utils import construct_bias, get_kappa



class QuAcq(ConAcq):
    def __init__(self, gamma, grid, ct, bias, X, C_l, qg="pqgen", gqg= False, gfs=False, gfc=False, obj="proba", classifier=None,
                 classifier_name=None, time_limit=None, findscope_version=4, findc_version=1, tqgen_t=None,
                 qgen_blimit=5000):
        super().__init__(gamma, grid, ct, bias, X, C_l, qg, gqg, gfs, gfc, obj, classifier, classifier_name, time_limit, findscope_version,
                    findc_version, tqgen_t, qgen_blimit)

    def learn(self):

        answer = True
        if len(self.B) == 0:
            self.B = construct_bias(self.X, self.gamma)

        while True:
            if self.debug_mode:
                print("Size of CL: ", len(self.C_l.constraints))
                print("Size of B: ", len(self.B))
                print("n of Queries: ", self.queries_count)

            gen_start = time.time()

            gen_flag, Y = self.call_query_generation(answer)

            gen_end = time.time()

            if not gen_flag:
                # if no query can be generated it means we have converged to the target network -----
                break

            self.metrics.increase_generation_time(gen_end - gen_start)
            self.metrics.increase_generated_queries()
            self.metrics.increase_top_queries()
            kappaB = get_kappa(self.B, Y)

            answer = self.ask_query(Y)
            if answer:
                # it is a solution, so all candidates violated must go
                # B <- B \setminus K_B(e)
                #        print("Removing the following constraints 1:", [c for c in B if check_value(c) is not False] )
                self.remove_from_bias(kappaB)
                if self.debug_mode:
                    print("B:", len(self.B))

            else:  # user says UNSAT

                scope = self.call_findscope(Y, kappaB)
                self.call_findc(scope)