import os
import pandas as pd
import pickle
from rdkit import Chem
from tqdm.auto import tqdm
from hack_lap.utils.prep import clean_graph


def main():
    filename = os.path.join('..', 'data', 'train.csv')
    output = os.path.join('..', 'data', 'train.pkl_')
    df = pd.read_csv(filename)
    df = df.loc[:, ['Smiles', 'Active']].astype({'Active': bool})
    dataset = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        mol = Chem.MolFromSmiles(row['Smiles'])
        dataset.append((clean_graph(mol), row['Active']))
    with open(output, 'wb') as fp:
        pickle.dump(dataset, fp, protocol=4)


if __name__ == '__main__':
    main()
