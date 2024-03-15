from src.memoization import p_load

locations = {
    "sudoku4": ('../data/sudoku/sudoku_56_C.pickle', '../data/sudoku/sudoku_56_CY.pickle'),
    "sudoku9": ('../data/sudoku/dataset_C.pickle', '../data/sudoku/dataset_CY.pickle'),
    "sudoku16": ('../data/sudoku/sudoku_4992_C.pickle', '../data/sudoku/sudoku_4992_CY.pickle'),
    "jsudoku": ('../data/jsudoku/jsudoku_811_C.pickle', '../data/jsudoku/jsudoku_811_CY.pickle'),
    "nurse": ('../data/nurse_rostering/dataset_C.pickle', '../data/nurse_rostering/dataset_CY.pickle'),
    "timetabling": ('../data/exam_timetabling/dataset_C.pickle', '../data/exam_timetabling/dataset_CY.pickle')
}


def load_bench(name):
    files = locations[name]
    return p_load(files[0]), p_load(files[1])
