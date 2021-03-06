import tensorflow as tf
import unittest
from pymatgen.core import Structure, Lattice

from megnet.utils.descriptor import MEGNetDescriptor, DEFAULT_MODEL
from tensorflow.keras.models import Model


class TestGeneralUtils(unittest.TestCase):
    def test_model_load(self):
        model = MEGNetDescriptor(model_name=DEFAULT_MODEL)
        self.assertTrue(model.model, Model)
        s = Structure(Lattice.cubic(3.6), ["Mo", "Mo"], [[0.5, 0.5, 0.5], [0, 0, 0]])
        atom_features = model.get_atom_features(s)
        bond_features = model.get_bond_features(s)
        glob_features = model.get_global_features(s)
        atom_set2set = model.get_set2set(s, ftype="atom")
        bond_set2set = model.get_set2set(s, ftype="bond")
        s_features = model.get_structure_features(s)
        self.assertListEqual(list(atom_features.shape), [2, 32])
        self.assertListEqual(list(bond_features.shape), [28, 32])
        self.assertListEqual(list(glob_features.shape), [1, 32])
        self.assertListEqual(list(atom_set2set.shape), [1, 32])
        self.assertListEqual(list(bond_set2set.shape), [1, 32])
        self.assertListEqual(list(s_features.shape), [1, 96])


if __name__ == "__main__":
    unittest.main()
