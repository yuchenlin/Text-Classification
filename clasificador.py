# Dataset: Polarity dataset v2.0
# http://www.cs.cornell.edu/people/pabo/movie-review-data/
#
# Based on the example from  :
# https://marcobonzanini.wordpress.com/2015/01/19/sentiment-analysis-with-python-and-scikit-learn
import sys
import os
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.externals import joblib
from random import shuffle
from sklearn.calibration import CalibratedClassifierCV


if __name__ == '__main__':

    data_dir = 'txt_sentoken'
    classes = ['pos', 'neg']

    # Read the data
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    for curr_class in classes:
        dirname = os.path.join(data_dir, curr_class)
        for fname in os.listdir(dirname):
            with open(os.path.join(dirname, fname), 'r') as f:
                content = f.read()
                if fname.startswith('cv9'):
                    test_data.append(content)
                    test_labels.append(curr_class)
                else:
                    train_data.append(content)
                    train_labels.append(curr_class)

    #spliting dataset for further calibration
    #shuffle elements in test_data and train_labels bc they were all pos and then all neg
    aux=list(zip(test_data,test_labels))
    shuffle(aux)
    test_data,test_labels = zip(*aux)

    valid_data=test_data[: int(len(test_data)/2)]
    valid_labels=test_labels[: int(len(test_labels)/2)]
    test_data2=test_data [int(len(test_data)/2):]
    test_labels2=test_labels [int(len(test_labels)/2):]



    # Create feature vectors
    vectorizer = TfidfVectorizer(min_df=5,
                                 max_df = 0.9,
                                 sublinear_tf=True,
                                 use_idf=True)
    train_vectors = vectorizer.fit_transform(train_data)
    joblib.dump(vectorizer,'vector.pkl')
    #Save the vectorizer object to be used in the web app
    test_vectors = vectorizer.transform(test_data)
    valid_vectors= vectorizer.transform(valid_data)
    test_vectors2= vectorizer.transform(test_data2)



    # Perform classification with SVM, kernel=rbf
    classifier_rbf = svm.SVC()
    classifier_rbf.fit(train_vectors, train_labels)
    prediction_rbf = classifier_rbf.predict(test_vectors)


    # Perform classification with SVM, kernel=linear
    classifier_linear = svm.SVC(kernel='linear')
    classifier_linear.fit(train_vectors, train_labels)
    prediction_linear = classifier_linear.predict(test_vectors)



    # Perform classification with SVM, kernel=linear
    classifier_liblinear = svm.LinearSVC()
    classifier_liblinear.fit(train_vectors, train_labels)
    #Probability calibration
    classifier_liblinear_c=CalibratedClassifierCV(classifier_liblinear,cv='prefit')
    classifier_liblinear_c.fit(valid_vectors,valid_labels)
    joblib.dump(classifier_liblinear_c,'clasificador.pkl')
    #LinearSVC was the classifier with the best prediction score, this one will be used in the web app
    prediction_liblinear = classifier_liblinear_c.predict(test_vectors2)


    # Print results in a nice table
    print("Results for SVC(kernel=rbf)")
    print(classification_report(test_labels, prediction_rbf))
    print("Results for SVC(kernel=linear)")
    print(classification_report(test_labels, prediction_linear))
    print("Results for LinearSVC()")
    print(classification_report(test_labels2, prediction_liblinear))
