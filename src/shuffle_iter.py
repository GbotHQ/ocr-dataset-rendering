from random import shuffle


class DatasetShuffleIterator:
    def __init__(self, dataset, **kwargs):
        self.dataset = dataset
        self.shuffled_dataset = dataset
        self.iterator = iter([])
        self.kwargs = kwargs

        self.shuffle()

    def shuffle(self, **kwargs):
        self.shuffled_dataset = self.shuffled_dataset.shuffle(**kwargs)
        self.iterator = iter(self.shuffled_dataset)

    def __next__(self):
        try:
            return next(self.iterator)
        except StopIteration:
            self.shuffle(**self.kwargs)
            return next(self.iterator)


class ShuffleIterator:
    def __init__(self, lst: list):
        self.index = 0
        self.lst = lst.copy()
        shuffle(self.lst)

    def __next__(self):
        if self.index >= len(self.lst):
            shuffle(self.lst)
            self.index = 0
        item = self.lst[self.index]
        self.index += 1
        return item
