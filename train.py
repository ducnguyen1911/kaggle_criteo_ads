import pandas as pd
import numpy as np
import sklearn.feature_extraction
import sklearn.linear_model
import sklearn.metrics
import gc

integer_stats_file = 'data/integer_stats.csv'
category_stats_file = 'data/category_stats.csv'

train_file_prefix = 'train_split'
train_file = range(0, 459)
# train_file = range(1, 40)
cv_file = range(0, 1)
test_file_prefix = 'test_split'
test_file = range(61)
# test_file = range(10)


def transform(input_file, keep_features, stats, file_type='train'):
    data = pd.read_csv(input_file)

    data_id = data['Id']

    keep_integer_features = [x for x in keep_features if x.startswith("I")]
    keep_category_features = [x for x in keep_features if x.startswith("C")]
    if file_type == 'train':
        data_integer = data.iloc[:, 2:15]
        data_category = data.iloc[:, 15:]
    else:
        data_integer = data.iloc[:, 1:14]
        data_category = data.iloc[:, 14:]

    # Set minimum value of features to 0
    data_integer['I2'] = data_integer['I2'].apply(lambda x: max(x, 0))

    # Filter out features
    data_integer = data_integer[keep_integer_features]
    data_category = data_category[keep_category_features]
    stats['integer'] = stats['integer'].loc[keep_integer_features, :]
    stats['category'] = stats['category'].loc[keep_category_features, :]


    # Mean normalization
    data_feature_integer = (data_integer.values.astype('float') - stats['integer']['mean'].values) / stats['integer'][
        'std'].values

    # Replace NaN with mean value
    # data_feature_integer = np.isnan(data_feature_integer) * stats['integer']['mean'].values + \
    # (~np.isnan(data_feature_integer)) * data_feature_integer

    # Truncate large integer values to 5x std. dev
    data_feature_integer = np.minimum(data_feature_integer, 2 * stats['integer']['std'].values)


    # log(1+f) integer features
    data_integer_log_features = np.log(1 + data_feature_integer)

    # combine features
    data_feature_integer = np.hstack([np.ones((data_feature_integer.shape[0], 1)),
                                      data_feature_integer, data_integer_log_features])

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
    stats['integer'].index = stats['integer'].iloc[:, 0]
    stats['category'].index = stats['category'].iloc[:, 0]

    all_classes = np.array([0, 1])

    integer_features = ['I' + str(i) for i in range(1, 14)]
    category_features = ['C' + str(i) for i in range(1, 27)]
    all_features = integer_features + category_features

    # L1 Feature selection
    l1_clf = sklearn.linear_model.SGDClassifier(loss='log', penalty='l1')
    for j in train_file:
        train_file_name = 'data/{0}{1}.csv'.format(train_file_prefix, str(j).zfill(3))
        print 'Training file ' + train_file_name

        X_train, y_train, id_train = transform(train_file_name, all_features, stats)
        l1_clf.partial_fit(X_train, y_train, classes=all_classes)

        # Force garbage collection
        gc.collect()

    print "Total features: " + str(l1_clf.coef_.shape[1])
    print "Features selected:" + str(np.sum(l1_clf.coef_ > 0))


    # Train all features
    clf = sklearn.linear_model.SGDClassifier(loss='log')
    for j in train_file:
        train_file_name = 'data/{0}{1}.csv'.format(train_file_prefix, str(j).zfill(3))
        print 'Training file ' + train_file_name
        X_train, y_train, id_train = transform(train_file_name, all_features, stats)

        # L1-feature selection
        X_train = l1_clf.transform(X_train)

        clf.partial_fit(X_train, y_train, classes=all_classes)

        # Force garbage collection
        gc.collect()

    # Load & cross-validate data
    for j in cv_file:
        val_file_name = 'data/{0}{1}.csv'.format(train_file_prefix, str(j).zfill(3))

        X_val, y_val, id_val = transform(val_file_name, all_features, stats)
        X_val = l1_clf.transform(X_val)

        y_predict = clf.predict_proba(X_val)

        print "CV Error: " + str(sklearn.metrics.accuracy_score(y_val.values, y_predict.argmax(axis=1)))
        print "CV Log Loss: " + str(sklearn.metrics.log_loss(y_val.values, y_predict))

    # Predict test data
    with open('test_pred.csv', 'wb+') as f:
        f.write('Id,Predicted\n')
        for j in test_file:
            test_file_name = 'data/{0}{1}.csv'.format(test_file_prefix, str(j).zfill(2))
            print 'Predicting file' + test_file_name

            X_test, id_test = transform(test_file_name, all_features, stats, file_type='test')
            X_test = l1_clf.transform(X_test)

            y_test_predict = clf.predict_proba(X_test)

            # Probability of a click
            y_test_prob = y_test_predict[:, 1]
            y_out = np.vstack([id_test, y_test_prob]).transpose()

            np.savetxt(f, y_out, delimiter=",", fmt=['%d', '%.4f'])

            # Garbage collection
            gc.collect()


main()
