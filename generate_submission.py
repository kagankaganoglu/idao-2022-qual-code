import yaml
import pandas as pd
import numpy as np

from pathlib import Path
from baseline import prepare_model
from baseline import read_pymatgen_dict
from megnet.models import MEGNetModel
import tensorflow as tf
from megnet.data.crystal import CrystalGraph

def energy_within_threshold(prediction, target):
    # compute absolute error on energy per system.
    # then count the no. of systems where max energy error is < 0.02.
    e_thresh = 0.02
    error_energy = tf.math.abs(target - prediction)

    success = tf.math.count_nonzero(error_energy < e_thresh)
    total = tf.size(target)
    return success / tf.cast(total, tf.int64)

def main(config):
    r_cutoff = 4
    nfeat_bond = 100
    gaussian_centers = np.linspace(0, r_cutoff + 1, nfeat_bond)
    gaussian_width = 0.8
    model = MEGNetModel(
        nblocks=3,
        #,n1=1
        #,n2=1
        #,n3=1
        graph_converter=CrystalGraph(cutoff=r_cutoff),
        centers=gaussian_centers,
        width=gaussian_width,
        loss=["MAE"],
        npass=2,
        lr=2e-4,
        metrics=energy_within_threshold,
        nvocal=95,
        embedding_dim=16,
        random_state=42
    )
    # model = prepare_model(
    #     float(config["model"]["cutoff"]), float(config["model"]["lr"])
    # )
    dataset_path = Path(config['test_datapath'])

    struct = {item.name.strip('.json'): read_pymatgen_dict(item) for item in (dataset_path/'structures').iterdir()}
    private_test = pd.DataFrame(columns=['id', 'structures'], index=struct.keys())
    private_test = private_test.assign(structures=struct.values())
    preds = pd.DataFrame(np.zeros(len(private_test)), index=private_test.index, columns=['predictions'])

    for i in range(1):
        model.load_weights(config['checkpoint_path'+str(i)])

        
        # private_test = private_test.assign(predictions=model.predict_structures(private_test.structures))
        # private_test = model.predict_structures(private_test.structures)
        preds['predictions'] += np.squeeze(model.predict_structures(private_test.structures))
    preds /= 1
    private_test['predictions'] = preds['predictions']
    private_test[['predictions']].to_csv('./submission.csv', index_label='id')

if __name__ == '__main__':
    with open("config.yaml") as file:
        config = yaml.safe_load(file)
    main(config)