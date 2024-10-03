import os.path


def load_benchmark(filename: str):
    with open(filename) as file:
        return [line.strip().split() for line in file.readlines()]


def store_benchmark(filename: str, benchmark):
    with open(filename, "w") as file:
        for inst in benchmark:
            file.write(f"{inst[0]}\t{inst[1]}\n")


def create_all_but_one_benchmark(filenameAll: str, dirButOne:str):
    d = os.path.dirname(filenameAll)
    f = str(os.path.basename(filenameAll))
    print(d)
    print(f)
    bench = load_benchmark(filenameAll)
    for i in range(len(bench)):
        file = f.replace(".", f"_allbut_{bench[i][1]}.")
        store_benchmark(os.path.join(d, file), bench[:i] + bench[i + 1:])
        file = f.replace(".", f"_only_{bench[i][1]}.")
        store_benchmark(os.path.join(dirButOne, file), [bench[i]])


if __name__ == "__main__":
    filename = "../../target/benchmarks/training_sets/classical_CA/training_set_classical_ca.txt"
    dir_other = "../../target/benchmarks/testing_sets/classical_CA"
    print(load_benchmark(filename))
    create_all_but_one_benchmark(filename,dir_other)
