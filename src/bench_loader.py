from src.utils.memoization import p_load

locations = {
    "sudoku4": ('../data/sudoku/sudoku_56_C.pickle', '../data/sudoku/sudoku_56_CY.pickle'),
    "sudoku9": ('../data/sudoku/dataset_C.pickle', '../data/sudoku/dataset_CY.pickle'),
    "sudoku16": ('../data/sudoku/sudoku_4992_C.pickle', '../data/sudoku/sudoku_4992_CY.pickle'),
    "jsudoku": ('../data/jsudoku/jsudoku_811_C.pickle', '../data/jsudoku/jsudoku_811_CY.pickle'),
    "nurse": ('../data/nurse_rostering/dataset_C.pickle', '../data/nurse_rostering/dataset_CY.pickle'),
    "nurse_2_7_15_5": ('../data/nurse_rostering/nurse_rostering_advanced_465_C_2_7_15_5.pickle', '../data/nurse_rostering/nurse_rostering_advanced_465_CY_2_7_15_5.pickle'),
    "nurse_3_7_15_5": ('../data/nurse_rostering/nurse_rostering_advanced_885_C_3_7_15_5.pickle', '../data/nurse_rostering/nurse_rostering_advanced_885_CY_3_7_15_5.pickle'),
    "nurse_4_7_20_5": ('../data/nurse_rostering/nurse_rostering_advanced_1480_C_4_7_20_5.pickle', '../data/nurse_rostering/nurse_rostering_advanced_1480_CY_4_7_20_5.pickle'),
    "timetabling": ('../data/exam_timetabling/dataset_C.pickle', '../data/exam_timetabling/dataset_CY.pickle')
}


def load_bench(name):
    files = locations[name]
    return p_load(files[0]), p_load(files[1])


if __name__ == "__main__":
    k = load_bench('sudoku4')
    print(k[0])
    print(k[1])
