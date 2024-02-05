# use pickle and json to save intermediate files
# pickle is used when the storage is only for speedup time
# json is used when the storage is to fix some element (partition train/test) that should be human readable

import json
import pickle


# load pickle file
def p_load(filename: str):
    with open(filename, 'rb') as file:
        obj = pickle.load(file)
        print('Load pickle ({})'.format(filename))
        return obj


# save to pickle file
def p_save(filename: str, obj, protocol=pickle.HIGHEST_PROTOCOL):
    with open(filename, "wb") as file:
        pickle.dump(obj, file, protocol)
        print('Save pickle ({})'.format(filename))


def p_save_high(filename: str, obj):
    p_save(filename, obj, pickle.HIGHEST_PROTOCOL)


def p_save_default(filename: str, obj):
    p_save(filename, obj, pickle.DEFAULT_PROTOCOL)


# load json file
def j_load(filename: str):
    with open(filename, "r") as f:
        data = f.read()
    obj = json.loads(data)
    print('Load json ({})'.format(filename))
    return obj


# save to json file
def j_save(filename: str, obj):
    data = json.dumps(obj)
    with open(filename, "w") as f:
        f.write(data)
        print('Save json ({})'.format(filename))
