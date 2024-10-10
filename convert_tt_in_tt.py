import numpy as np
import json
import pickle
import os
import matplotlib.pyplot as plt

# fichier pour convertir les tt.npz en tt.pkl(pour MUROLS et FRINAULT)

def convert_old_tt(path):
    """"Fonction qui convertit les tt.npz de MUROLS et FRINAULT  en tt.pkl utilisables par les autres fonctions ensuite"""
    tt = np.load(path+'tt.npz')

    keys = tt.keys()

    triggers, tones = [], []
    mock_triggers, mock_tones = [], []
    condition, block = [], []
    w_triggers, w_tones = [], []

    n = []
    #déterminer le nombre de blocks
    for key in keys:
        key_str = str(key) 
            # S'assurer que le titre a au moins 3 caractères
        if len(key_str) >= 3:
            c = key_str[:2] #extraure la condition
            if c == 'pb':
                n.append(int(key_str[-1])) # extraire le block
    n_block = np.max(n)
    

    for key in keys:
        key_str = str(key) 
            # S'assurer que le titre a au moins 3 caractères
        if len(key_str) >= 3:
            c = key_str[:2] #extraure la condition
            b = key_str[-1] # extraire le block
        if c=='pb':
            pass
            triggers.append(tt[key][1])
            tones.append(tt[key][0])
            condition.append(np.ones_like(tt[key][0]))
            block.append(np.full(tt[key][0].shape, f'Block_00{int(b)}', dtype=object))
        elif c=='tr':
            triggers.append(tt[key][1])
            tones.append(tt[key][0])
            condition.append(np.zeros_like(tt[key][0]))
            block.append(np.full(tt[key][0].shape, f'Block_00{int(b)}', dtype=object))
        elif c =='wp':
            triggers.append(tt[key][1])
            tones.append(tt[key][0])
            condition.append(np.full_like(tt[key][0], -1))
            block.append(np.full(tt[key][0].shape, f'Block_00{int(b)}', dtype=object))
        elif c =='wd':
            print(f"warmdown exists, block is {n_block+1}")
            triggers.append(tt[key][1])
            tones.append(tt[key][0])
            condition.append(np.full_like(tt[key][0], -1))
            block.append(np.full(tt[key][0].shape, f'Block_00{int(n_block)+1}', dtype=object))

        elif c =='mk':
            mock_triggers.append(tt[key][1])
            mock_tones.append(tt[key][0])
            #condition.append(np.full_like(tt[key][0], -1))
            #block.append(np.full(tt[key][0].shape, f'Block_00{int(b)}', dtype=object))
        else:
            pass

    trig_times = np.hstack(triggers)
    tones = np.hstack(tones)
    condition = np.hstack(condition)
    block = np.hstack(block)

    mock_trig_times = np.hstack(mock_triggers)
    mock_tones = np.hstack(mock_tones)

    sorted_indices = np.argsort(trig_times)
    sorted_triggers = trig_times[sorted_indices]
    sorted_tones = tones[sorted_indices]
    sorted_condition = np.array(condition)[sorted_indices]
    sorted_block = np.array(block)[sorted_indices]
    plt.plot(sorted_condition, c = 'red')
    tt = {
        'tones': sorted_tones,
        'triggers': sorted_triggers, 
        'condition' : sorted_condition,
        'block': sorted_block,
        'mock_tones' : mock_tones, 
        'mock_triggers' : mock_trig_times
            }
            
            # save tt.pkl
    with open(path+'headstage_0/tt.pkl', 'wb') as file:
        pickle.dump(tt, file)
    print("tt converted")


path = '/auto/data2/eTheremin/MUROLS/MUROLS_20230301/MUROLS_20230301_SESSION_00/'

convert_old_tt(path)
