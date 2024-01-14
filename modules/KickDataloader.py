from torch.utils.data import DataLoader

class KickDataloader(DataLoader):
    def __init__(self, dataset):
        super(KickDataloader, self).__init__(dataset)

    def __iter__(self):
        # Custom iterfunction possible
        return super(KickDataloader, self).__iter__()

