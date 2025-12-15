from dataset import MddDataset

# taken from scPrediXcan tutorial
# https://github.com/hakyimlab/scPrediXcan/blob/master/Scripts/ctPred/Tutorial.ipynb
train_set = ["1", "10", "13", "15", "16", "17", "18", "19", "2", "21", "22", "3", "4", "6", "8", "9", "X", "Y"]
val_set = ["11", "14", "7"]
test_set = ["12", "20", "5"]

def get_train_test_dataset(dataset: MddDataset):
    """Load dataset and split into train and test sets."""
    train_dataset = dataset.split_by_chromosome(train_set)
    test_dataset  = dataset.split_by_chromosome(val_set + test_set)
    return train_dataset, test_dataset