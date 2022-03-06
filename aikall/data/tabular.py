from .dataset import Dataset


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