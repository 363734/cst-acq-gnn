from src.prior.prior_opt import parse_args
from src.prior.sk.prior_sk import eval_prior_sk

if __name__ == "__main__":
    print(f"===== EVALUATION =====")
    args = parse_args()
    print(f"\tparameters: {args}")

    if args.architecture_gnn:
        print()
    else:
        eval_prior_sk(args)
