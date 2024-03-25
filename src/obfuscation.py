import configparser
import logging
from mechanisms import CMP, Mhl
from tqdm import tqdm
import pandas as pd
from functools import partial
import os
tqdm.pandas()    

if not os.path.exists('./output'):
    os.makedirs('./output')
if not os.path.exists('./output/logs'):
    os.makedirs('./output/logs')

logging.basicConfig(filename='./output/logs/logger.log', level=logging.DEBUG, filemode='w', format='%(name)s - %(levelname)s - %(message)s - %(asctime)s')
logger = logging.getLogger(__name__)

#config = configparser.ConfigParser()
#config.read("../config/config.ini")

mechs = ["CMP", "Mhl"]
params = ['epsilon', 'lambda']
eps = 1

dict_mechs = {"CMP": {"epsilon" : eps}, "Mhl": {"epsilon" : eps, "lambda":1}}

#for mech in mechs:
#    dict_mechs[mech] = {}
#    for param in params:
#        try:
#            dict_mechs[mech][param] = [float(item.strip()) for item in config[mech][param].split(',')]
#        except KeyError as e:
#            logger.info(f"Parameter not needed, catching exception: {e}")
#            continue
#        except ValueError as e:
#            logger.error(f"Error converting {param} values for {mech} to float: {e}")
#            continue

def construct(mech, params):
    if mech == "CMP":
        return CMP(params)
    elif mech == "Mhl":
        return Mhl(params)
    else:
        logger.error(f"Mechanism {mech} not recognized")
        os._exit(1)

def main():
    logger.info("Starting obfuscation")
    logger.info(dict_mechs)
    mech_objs = []
    for mech in dict_mechs.keys():
        logger.info(f"Constructing mechanism {mech}")
        mech_obj  = construct(mech, dict_mechs[mech])
        logger.info(mech_obj)
        mech_objs.append(mech_obj)
    #read original dataset
    df = pd.read_csv(f'./data/boolQ/boolQ.csv', sep = ',', header = 0)
    #obfuscate the dataset
    for mech_obj in mech_objs:
        logger.info(f"Obfuscating dataset with mechanism {mech_obj}")
        for q in df['question']:
            df[q] = mech_obj.obfuscate(q)
        df.to_csv(f'./data/boolQ/obfuscated/{mech_obj.__class__.__name__}/{mech_obj.epsilon}.csv', index=False)
    logger.info("Obfuscation finished")

if __name__ == "__main__":
    main()