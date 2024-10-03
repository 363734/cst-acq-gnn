from src.prior.sk.prior_sk import train_prior_sk

from src.prior.prior_opt import parse_args

if __name__ == "__main__":
    print(f"===== TRAINING =====")
    args = parse_args()
    print(f"\tparameters: {args}")

    if args.architecture_gnn:
        print()
    else:
        train_prior_sk(args)
