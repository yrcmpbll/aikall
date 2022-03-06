from .dataset import Dataset
import numpy as np


class TabularDataset(Dataset):

    def __init__(self, table, types) -> None:
        """TabularDataset constructor. It takes a Pandas DataFrame and
        a dictionary of {column: type}. The dataset class is then 
        mounted with this specification.

        Args:
            table (pandas.DataFrame): Table with the data.
            types (dictionary): Python dictionary with column names as keys 
                and data types as values.
        """
        super().__init__()

        self.data = table
        self.types = types

        self.train = None
        self.dev = None
        self.test = None
        self.target_name = None
        self.train_target = None
        self.dev_target = None
        self.test_target = None

    def set_target(self, name):
        """Extracts one column from the data table and sets it as target.

        Args:
            name (str): Name of the column to be set as target.
        """
        if (self.train is None) or (self.test is None) or (self.dev is None):
            raise ValueError("Data set splits not yet initialized. \
                                You should call train_dev_test_split first.")
        
        if name not in self.data.columns:
            raise ValueError(f"Target column is not in table.")
        
        self.train_target = self.train.pop(name)
        self.dev_target = self.dev.pop(name)
        self.test_target = self.test.pop(name)
    
    def train_dev_test_split(self, train_frac=.8):
        """Splits the data table in train, dev and test.

        Args:
            train_frac (float, optional): Fraction of the data to be assigned to train. 
                                            Defaults to .8.
        """
        df = self.data

        # shuffle and split
        train, dev, test = \
              np.split(df.sample(frac=1), 
                       [int(.6*len(df)), int(.8*len(df))])
        
        self.train = train
        self.dev = dev
        self.test = test