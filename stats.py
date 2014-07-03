import pandas as pd
import sys
import gc


def compute_integer_stats(input_file, chunk_size):
    """
    Compute training data statistics
    Integer Variables: Min, Max, Mean
    Click: Click, Not Click

    :return:
    """
    stats_integer = pd.DataFrame()

    clicks = 0
    impressions = 0

    reader = pd.read_csv(input_file, chunksize=chunk_size)

    count = 0
    for chunk in reader:
        print 'Reading line:' + str(count * chunk_size)

        chunk_integer = chunk.iloc[:, 2:15]

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

        clicks += chunk['Label'].sum()
        impressions += chunk.shape[0]

        count += 1

    stats_integer['mean'] = stats_integer['sum'] / stats_integer['count']

    print stats_integer
    print "Total Clicks:" + str(clicks) + " Total Impressions:" + str(impressions)


def compute_category_stats(input_file, category_label, chunk_size):
    """
    Compute training data statistics
    Categorical Variables: Number of categories, Histogram
    Integer Variables: Min, Max, Mean
    Click: Click, Not Click

    :return:
    """
    stats_category = pd.DataFrame()

    reader = pd.read_csv(input_file, chunksize=chunk_size)

    count = 0
    for chunk in reader:
        print 'Reading line:' + str(count * chunk_size)

        chunk_category = chunk.iloc[:, 15:]

        frame = pd.DataFrame()
        frame['category'] = chunk_category.groupby(category_label).size().index
        frame['count'] = chunk_category.groupby(category_label).size().values
        stats_category = pd.concat([stats_category, frame])

        # Aggregate on common category values
        frame = pd.DataFrame()
        frame['category'] = stats_category.groupby('category').sum().index
        frame['count'] = stats_category.groupby("category").sum().values
        stats_category = frame

        # Force garbage collection
        gc.collect()

        count += 1

    return stats_category.describe()


def compute_category_stats_all(input_file, chunk_size):
    """
    Compute training data statistics
    Categorical Variables: Number of categories, Histogram
    Integer Variables: Min, Max, Mean
    Click: Click, Not Click

    :return:
    """
    stats_category = {}
    for i in range(1, 27):
        stats_category['C' + str(i)] = pd.DataFrame()

    reader = pd.read_csv(input_file, chunksize=chunk_size)

    count = 0
    for chunk in reader:
        print 'Reading line:' + str(count * chunk_size)

        chunk_category = chunk.iloc[:, 15:]

        for i in range(1, 27):
            category_label = 'C' + str(i)

            frame = pd.DataFrame()
            frame['category'] = chunk_category.groupby(category_label).size().index
            frame['count'] = chunk_category.groupby(category_label).size().values
            stats_category[category_label] = pd.concat([stats_category[category_label], frame])

            # Aggregate on common category values
            frame = pd.DataFrame()
            frame['category'] = stats_category[category_label].groupby('category').sum().index
            frame['count'] = stats_category[category_label].groupby("category").sum().values
            stats_category[category_label] = frame

            gc.collect()

        count += 1

    stats_category_agg = pd.DataFrame()
    for i in range(1, 27):
        frame = stats_category['C' + str(i)].groupby('category').sum().describe().transpose()
        frame.reset_index()
        frame.index = ['C' + str(i)]
        stats_category_agg = pd.concat([stats_category_agg, frame])

    print stats_category_agg


def main():
    if len(sys.argv) == 4:
        action_type = sys.argv[1]
        if action_type == 'integer':
            chunk_size = int(sys.argv[3])
            input_file = sys.argv[2]
            compute_integer_stats(input_file, chunk_size)
        elif action_type == 'category':
            input_file = sys.argv[2]
            chunk_size = int(sys.argv[3])
            # stats_category = pd.DataFrame()
            # for i in range(1, 27):
            # category_label = 'C' + str(i)
            #     print "Starting analysis for category: " + category_label
            #     frame = compute_category_stats(input_file, category_label, chunk_size).transpose()
            #     frame.reset_index()
            #     frame.index = [category_label]
            #     stats_category = pd.concat([stats_category, frame])
            # print stats_category
            compute_category_stats_all(input_file, chunk_size)
    else:
        print "Usage: python stats.py integer <input file> <chunk size>"
        print "Usage: python stats.py category <input file> <chunk size>"


main()