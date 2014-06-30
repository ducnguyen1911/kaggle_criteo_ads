import pandas as pd
import numpy as np

chunk_size = 100000


def transform(input_file, output_file, file_type):
    """
    Transform train & test files to vowpal wabbit format
    :param input_file: train or test input csv file path
    :param file_type: 'train' or 'test'
    :return:
    """

    output_writer = open(output_file, 'w')
    reader = pd.read_csv(input_file, chunksize=chunk_size)

    count = 0
    for chunk in reader:
        count += 1
        if count % 10 == 0:
            print 'Reading line:' + str(count*chunk_size)

        for row in chunk.iterrows():
            if file_type == 'train':
                label = row[1]['Label']

            row_id = row[1]['Id']
            del row[1]['Id']

            category_data = [(key, row[1][key]) for key in row[1].keys() if key.startswith('C')]
            integer_data = [(key, row[1][key]) for key in row[1].keys() if key.startswith('I')]

            integer_string = ' '.join([key + ':' + str(val) for key, val in integer_data if not np.isnan(val)])
            category_string = ' '.join([str(val) for key, val in category_data if str(val) != 'nan'])

            if file_type == 'train':
                output_line = '{0} {1}|Integer {2} |Category {3}\n'.format(str(label), str(row_id), integer_string,
                                                                           category_string)
            elif file_type == 'test':
                output_line = '{0}|Integer {1} |Category {2}\n'.format(str(row_id), integer_string,
                                                                       category_string)

            output_writer.write(output_line)

    output_writer.close()


# transform('data/train10K.csv', 'data/train10K.vw', 'train')
# transform('data/train100K.csv', 'data/train100K.vw', 'train')
# transform('data/test10K.csv', 'data/test10K.vw', 'test')
# transform('data/test100K.csv', 'data/test100K.vw', 'test')
transform('data/train.csv', 'data/train.vw', 'train')
