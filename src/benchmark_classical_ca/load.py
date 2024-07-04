import os.path


def load_benchmark(filename: str):
    with open(filename) as file:
        return [line.strip().split() for line in file.readlines()]


def store_benchmark(filename: str, benchmark):
    with open(filename, "w") as file:
        for inst in benchmark:
            file.write(f"{inst[0]}\t{inst[1]}\n")


def create_all_but_one_benchmark(filename: str):
    d = os.path.dirname(filename)
    f = str(os.path.basename(filename))
    print(d)
    print(f)
    bench = load_benchmark(filename)
    for i in range(len(bench)):
        file = f.replace(".", f"_allbut_{bench[i][1]}.")
        store_benchmark(os.path.join(d, file), bench[:i] + bench[i + 1:])


if __name__ == "__main__":
    filename = "../../target/training_sets/training_set_sudoku.txt"
    print(load_benchmark(filename))
    create_all_but_one_benchmark(filename)
