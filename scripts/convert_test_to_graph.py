import os
import pandas as pd
import pickle
from rdkit import Chem
from tqdm.auto import tqdm
from hack_lap.utils.prep import clean_graph


def main():
    filename = os.path.join('..', 'data', 'test.csv')
    output = os.path.join('..', 'data', 'test.pkl')
    df = pd.read_csv(filename)
    df = df['Smiles'].tolist()
    dataset = []
    for smile in tqdm(df):
        mol = Chem.MolFromSmiles(smile)
        dataset.append(clean_graph(mol))
    with open(output, 'wb') as fp:
        pickle.dump(dataset, fp, protocol=4)


if __name__ == '__main__':
    main()
