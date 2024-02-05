from src.generate_training import generate_senarios_pourcent, get_indicator
from src.graph import create_graph_unknown, switch_senario
from src.memoization import p_load
from src.training import training

f1a = p_load('../data/sudoku/dataset_C.pickle')
f1b = p_load('../data/sudoku/dataset_CY.pickle')
f2a = p_load('../data/jsudoku/dataset_C.pickle')
f2b = p_load('../data/jsudoku/dataset_CY.pickle')
f3a = p_load('../data/nurse_rostering/dataset_C.pickle')
f3b = p_load('../data/nurse_rostering/dataset_CY.pickle')
f4a = p_load('../data/exam_timetabling/dataset_C.pickle')
f4b = p_load('../data/exam_timetabling/dataset_CY.pickle')

f = f1a
gt = f1b
# for f in [f1b, f2b, f3b]:
print('----------')
print(f)
print(len(f))
# print(f)
print(set([type(e) for e in f]))

g = create_graph_unknown(f)
print(g)
print(g.nodes['cst'].data)

senarios = generate_senarios_pourcent(19, len(gt), 0.1)
indicators = [get_indicator(gt, s) for s in senarios]
#print(indicators)

updated_g = [switch_senario(g, indicators[i]) for i in range(len(senarios))]

#print(updated_g)

batch = [(g, gt)] + [(u_g, gt) for u_g in updated_g]

training(batch)
