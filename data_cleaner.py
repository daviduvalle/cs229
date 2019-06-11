import pandas as pd
import datetime

def convert(x):

    if not isinstance(x, str):
        return 0

    tokens = x.split('/')
    year = int(tokens[2])
    month = int(tokens[0])
    day = int(tokens[1])
    year = 2000 + year
    date = datetime.datetime(year, month, day)
    return date.timestamp()

def balance(x):
    """Create a balance of positive and negative examples"""
    y = x['Disease(0-100)']
    print('total y_s {}'.format(len(y)))
    count_ones = (y == 1).sum(axis=0)
    count_zeros = (y == 0).sum(axis=0)
    print('total positives {}, total negatives {}'.format(count_ones, count_zeros))
    return count_zeros, count_ones

def randomize_save(data):
    # Randomize and split dataset
    train = data.sample(frac=0.8, random_state=200)
    rest = data.drop(train.index)
    validation = rest.sample(frac=0.5, random_state=200)
    test = rest.drop(validation.index)

    train.to_csv('data/train.csv')
    validation.to_csv('data/validation.csv')
    test.to_csv('data/test.csv')

def main():
    print("cleans data")
    data = pd.read_csv('corn2.csv')
    print('cols {}'.format(len(data.dtypes)))
    data = data.drop(['block', 'range', 'pass', 'dateHarvested', 'productivity', 'location', 'rootLodgedPlants', 'pollenDays', 'pedigree'], axis=1)
    data['Disease(0-100)'] = data['Disease(0-100)'].apply(lambda x: True if x > 12 else False)
    data['datePlanted'] = data['datePlanted'].apply(lambda x: convert(x))
    data['anthesisDate'] = data['anthesisDate'].apply(lambda x: convert(x))
    print('cols {}'.format(len(data.dtypes)))
    # Remove rows with empty cells
    data = data.dropna(axis=0, how='any', thresh=None, subset=None, inplace=False)

    # Remove data to make it balanced
    count_zeros, count_ones = balance(data)
    remove_count = count_zeros - count_ones
    idx = data.index[data['Disease(0-100)'] == 0]
    print('All 0 index {}'.format(len(idx)))
    idx = idx[0:remove_count]
    print('After removal {} '.format(len(idx)))
    data = data.drop(idx)

    randomize_save(data)

main()