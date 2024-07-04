from src.prior.nn.prior_nn import eval_prior_nn

from src.prior.prior_opt import parse_args

if __name__ == "__main__":
    print(f"===== EVALUATION =====")
    args = parse_args()
    print(f"\tparameters: {args}")

    if args.architecture_nn:

        eval_prior_nn(args)

    elif args.architecture_gnn:
        print()
