import pytest
from sklearn.datasets import load_wine
from aikall.data.tabular import TabularDataset


def test_data():
    data = load_wine(as_frame=True)

    col_names = data.frame.columns

    types = dict()
    for c in col_names:
        types[c] = float

    td = TabularDataset(table=data.frame, types=types)

    td.set_target(name='target')

    data = load_wine(as_frame=True)

    td = TabularDataset(table=data.frame, types=types, target_name='target')