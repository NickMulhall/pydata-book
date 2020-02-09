import pickle
import pandas as pd
import json

def pickle_load(dir):
    #df = pickle.load(open(dir + '/pickle_out.pickle'))
    with open(dir + 'pickle_out.pickle') as f:
       df = pickle_load(f)
    return df
def pickle_save(dir,frame):
    # Two options to write  
    with open(dir + 'pickle_out.pickle', 'wb') as f:
        pickle.dump(frame,f)
    #frame.to_pickle(dir + '/pickle_out.pickle')

path = 'datasets/pickle/example.txt'
records = [json.loads(line) for line in open(path)]
frame = pd.DataFrame(records)
directory = 'datasets/pickle/'
#pickle_save(directory,frame)

df = pickle_load(directory)
print(df[10]) 

