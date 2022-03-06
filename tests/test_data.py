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

    assert td


def test_split():
    data = load_wine(as_frame=True)

    col_names = data.frame.columns

    types = dict()
    for c in col_names:
        types[c] = float

    td = TabularDataset(table=data.frame, types=types)

    td.train_dev_test_split()


def test_set_target():
    data = load_wine(as_frame=True)

    col_names = data.frame.columns

    types = dict()
    for c in col_names:
        types[c] = float

    td = TabularDataset(table=data.frame, types=types)

    td.train_dev_test_split()

    td.set_target(name='target')


def test_min_max_scaling():
    data = load_wine(as_frame=True)

    col_names = data.frame.columns

    types = dict()
    for c in col_names:
        types[c] = float

    td = TabularDataset(table=data.frame, types=types)

    td.train_dev_test_split()

    td.set_target(name='target')

    td.min_max_scaling()



def test_standard_scaling():
    data = load_wine(as_frame=True)

    col_names = data.frame.columns

    types = dict()
    for c in col_names:
        types[c] = float

    td = TabularDataset(table=data.frame, types=types)

    td.train_dev_test_split()

    td.set_target(name='target')

    td.standard_scaling()