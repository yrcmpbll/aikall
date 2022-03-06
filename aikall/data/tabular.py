from .dataset import Dataset
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd


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

        self.numeric_features = None
        self.categorical_features = None

        self.train = None
        self.dev = None
        self.test = None
        self.target_name = None
        self.train_target = None
        self.dev_target = None
        self.test_target = None
        self.target_type = None

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

        # Get type from the target.
        # This will be either int or 'cat' for classification
        # or float for regression problems.
        self.target_type = self.types[name]

        # Initialize the type of the features.
        self.numeric_features = list()
        self.categorical_features = list()
        for feat in self.train.columns:
            if (self.types[feat] == int) or (self.types[feat] == float):
                self.numeric_features.append(feat)
            elif self.types[feat] == 'cat':
                self.categorical_features.append(feat)
    
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
    
    @staticmethod
    def __transform_subspace(df, func, features_to_transform):
        trans_df = df[features_to_transform].copy()

        fix_features = [f for f in df.columns if f not in features_to_transform]
        fix_df = df[fix_features].copy()

        trans_df = pd.DataFrame(data=func(trans_df), columns=trans_df.columns)

        return pd.concat([trans_df, fix_df], axis=1)

    def rescale(self, scaler):
        if (self.train_target is None) or (self.test_target is None) or (self.dev_target is None):
            raise ValueError("Target not yet initialized.")

        fitted_scaler = scaler.fit(self.train[self.numeric_features])

        # self.train = fitted_scaler.transform(self.train)
        # self.dev = fitted_scaler.transform(self.dev)
        # self.test = fitted_scaler.transform(self.test)

        def rescale_num_dataset(df):
            return self.__transform_subspace(df=df,
                                             func=scaler.transform,
                                             features_to_transform=self.numeric_features)

        self.train = rescale_num_dataset(self.train)
        self.dev = rescale_num_dataset(self.dev)
        self.test = rescale_num_dataset(self.test)
    
    def min_max_scaling(self):
        scaler = MinMaxScaler()
        self.rescale(scaler=scaler)
    
    def standard_scaling(self):
        scaler = StandardScaler()
        self.rescale(scaler=scaler)