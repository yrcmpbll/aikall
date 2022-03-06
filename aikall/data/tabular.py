from .dataset import Dataset


class TabularDataset(Dataset):

    def __init__(self, table, types, target_name=None) -> None:
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

        if target_name is None:
            self.target = None
        else:
            if target_name in self.data.columns:
                self.set_target(name=target_name)
            else:
                raise ValueError(f"Column {target_name} not in table.")

    def set_target(self, name):
        self.target = self.data.pop(name)
 