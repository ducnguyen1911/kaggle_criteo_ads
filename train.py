import pandas as pd
import numpy as np
import sklearn.feature_extraction
import sklearn.linear_model
import sklearn.metrics
import gc

integer_stats_file = 'data/integer_stats.csv'
category_stats_file = 'data/category_stats.csv'

train_file_prefix = 'train_split'
train_file = range(0, 10)
cv_file = range(0, 1)
test_file_prefix = 'test_split'
test_file = range(10)

integer_features = ['I' + str(i) for i in range(1, 14)]
category_features = ['C' + str(i) for i in range(1, 27)]
features = integer_features + category_features


def transform(input_file, keep_features, stats, file_type='train'):
    data = pd.read_csv(input_file)

    data_id = data['Id']

    if file_type == 'train':
        data_integer = data.iloc[:, 2:15]
        data_category = data.iloc[:, 15:]
    else:
        data_integer = data.iloc[:, 1:14]
        data_category = data.iloc[:, 14:]

    # Set minimum value of features to 0
    data_integer['I2'] = data_integer['I2'].apply(lambda x: max(x, 0))

    # Mean normalization
    data_feature_integer = (data_integer.values.astype('float') - stats['integer']['mean'].values) / stats['integer'][
        'std'].values

    # Replace NaN with mean value
    # X_train_integer = np.isnan(X_train_integer) * stats['integer']['mean'].values + \
    # (~np.isnan(X_train_integer)) * X_train_integer

    # Truncate large integer values to 5x std. dev
    data_feature_integer = np.minimum(data_feature_integer, 5 * stats['integer']['std'].values)

    # f^2 (TBD) & log(1+f) features
    data_feature_integer = np.hstack([np.ones((data_feature_integer.shape[0], 1)),
                                      data_feature_integer, np.log(1 + data_feature_integer)])

    data_feature_integer = [dict(('I' + str(j), u)
                                 for j, u in enumerate(item) if str(u) != 'nan')
                            for item in data_feature_integer]

    # Categorical features
    data_feature_category = [dict((u, 1) for u in item if str(u) != 'nan') for item in data_category.values]

    # Combine integer & categorical features
    data_feature = [dict(x.items() + y.items()) for (x, y) in zip(data_feature_integer, data_feature_category)]

    # Hash features
    fh = sklearn.feature_extraction.FeatureHasher(non_negative=True)
    data_feature = fh.fit_transform(data_feature)

    if file_type == 'train':
        data_label = data['Label']
        return data_feature, data_label, data_id
    elif file_type == 'test':
        return data_feature, data_id


# Main Code
def main():
    # Load statistics
    stats = {'integer': pd.read_csv(integer_stats_file), 'category': pd.read_csv(category_stats_file)}

    clf = sklearn.linear_model.SGDClassifier(loss='log')

    all_classes = np.array([0, 1])

    for j in train_file:
        train_file_name = 'data/{0}{1}.csv'.format(train_file_prefix, str(j).zfill(2))
        print 'Training file' + train_file_name
        X_train, y_train, id_train = transform(train_file_name, features, stats)

        clf.partial_fit(X_train, y_train, classes=all_classes)

        # Force garbage collection
        gc.collect()

    # Load CV data
    for j in cv_file:
        val_file_name = 'data/{0}{1}.csv'.format(train_file_prefix, str(j).zfill(2))
        X_val, y_val, id_val = transform(val_file_name, features, stats)
        y_predict = clf.predict_proba(X_val)
        y_prob = y_predict.max(axis=1)

        print "CV Error: " + str(sklearn.metrics.accuracy_score(y_val.values, y_predict.argmax(axis=1)))
        print "CV Log Loss: " + str(sklearn.metrics.log_loss(y_val.values, y_predict))

    # Predict test data
    with open('test_pred.csv', 'wb+') as f:
        f.write('Id, Predicted\n')
        for j in test_file:
            test_file_name = 'data/{0}{1}.csv'.format(test_file_prefix, str(j).zfill(2))
            print 'Predicting file' + test_file_name

            X_test, id_test = transform(test_file_name, features, stats, file_type='test')
            y_test_predict = clf.predict_proba(X_test)
            y_test_prob = y_test_predict.max(axis=1)
            y_out = np.vstack([id_test, y_test_prob]).transpose()
            np.savetxt(f, y_out, delimiter=",", fmt=['%d', '%.4f'])

            # Garbage collection
            gc.collect()


main()
