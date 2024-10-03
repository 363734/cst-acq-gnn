from src.bench_loader import load_bench

if __name__ == "__main__":
    bench = "nurse"
    f, gt = load_bench(bench)
    ct = 0
    for i in range(len(f)):
        if gt[i] == 1:
            print(f[i])
            ct+=1
    print(ct)
