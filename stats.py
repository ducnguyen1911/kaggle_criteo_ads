import pandas as pd


chunk_size = 1000000
input_file = 'data/train.csv'


def compute_stats():

    """
    Compute training data statistics
    Categorical Variables: Number of categories, Histogram
    Integer Variables: Min, Max, Mean
    Click: Click, Not Click

    :return:
    """
    stats_integer = pd.DataFrame()
    stats_category = {}
    for i in range(1, 27):
        stats_category['C' + str(i)] = pd.DataFrame()

    clicks = 0

    reader = pd.read_csv(input_file, chunksize=chunk_size)

    count = 0
    for chunk in reader:
        if count % 10 == 0:
            print 'Reading line:' + str(count*chunk_size)

        chunk_integer = chunk.iloc[:, 2:15]
        chunk_category = chunk.iloc[:, 15:]

        if count == 0:
            stats_integer['max'] = chunk_integer.max()
            stats_integer['min'] = chunk_integer.min()
            stats_integer['sum'] = chunk_integer.sum()
            stats_integer['count'] = chunk_integer.count()
        else:
            stats_integer['max_chunk'] = chunk_integer.max()
            stats_integer['min_chunk'] = chunk_integer.min()
            stats_integer['sum_chunk'] = chunk_integer.sum()
            stats_integer['count_chunk'] = chunk_integer.count()

            stats_integer['max'] = stats_integer[['max', 'max_chunk']].max(axis=1)
            stats_integer['min'] = stats_integer[['min', 'min_chunk']].max(axis=1)
            stats_integer['sum'] = stats_integer[['sum', 'sum_chunk']].sum(axis=1)
            stats_integer['count'] = stats_integer[['count', 'count_chunk']].sum(axis=1)

            stats_integer = stats_integer.drop(['max_chunk', 'min_chunk', 'sum_chunk', 'count_chunk'], axis=1)

        for i in range(1, 27):
            category_label = 'C' + str(i)

            frame = pd.DataFrame()
            frame['category'] = chunk_category.groupby(category_label).size().index
            frame['count'] = chunk_category.groupby(category_label).size().values
            stats_category[category_label] = pd.concat([stats_category[category_label], frame])

        clicks += chunk['Label'].sum()

        count += 1

    stats_integer['mean'] = stats_integer['sum'] / stats_integer['count']

    stats_category_agg = pd.DataFrame()
    for i in range(1, 27):
        frame = stats_category['C' + str(i)].groupby('category').sum().describe().transpose()
        frame.reset_index()
        frame.index = ['C' + str(i)]
        stats_category_agg = pd.concat([stats_category_agg, frame])

    print stats_category_agg

    print stats_integer

    return 0

compute_stats()
