import pandas as pd
import re
from os import system, listdir
from os.path import isfile, join
from random import shuffle
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from scipy.sparse import csr_matrix
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from joblib import dump, load # used for saving and loading sklearn objects
from scipy.sparse import save_npz, load_npz # used for saving and loading sparse matrices
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords

#AFTER FIRST RUN THIS SECTION SHOULD BE UNCOMENT FOR FARTHER TESTING IT WILL SPEED THE PROCES
'''
#LOADING PREPARED CLASSIFIERS, PREPROCESSORS, VECTORIED DATA, DATA FRAMES
#DATA FRAMES
imdb_test = pd.read_csv('csv/imdb_test.csv')
imdb_train = pd.read_csv('csv/imdb_train.csv')
#DATA PREPROCESORS
unigram_vectorizer = load('data_preprocessors/unigram_vectorizer.joblib')
bigram_vectorizer = load('data_preprocessors/bigram_vectorizer.joblib')
unigram_tf_idf_transformer = load('data_preprocessors/unigram_tf_idf_transformer.joblib')
bigram_tf_idf_transformer = load('data_preprocessors/bigram_tf_idf_transformer.joblib')
#VECTORIZED DATA FOR TRAINING
X_train_unigram = load_npz('vectorized_data/X_train_unigram.npz')
X_train_unigram_tf_idf = load_npz('vectorized_data/X_train_unigram_tf_idf.npz')
X_train_bigram = load_npz('vectorized_data/X_train_bigram.npz')
X_train_bigram_tf_idf = load_npz('vectorized_data/X_train_bigram_tf_idf.npz')
#VECTORIZED DATA FOR TESTING
X_test_unigram = load_npz('vectorized_data/X_test_unigram.npz')
X_test_bigram = load_npz('vectorized_data/X_test_bigram.npz')
X_test_bigram_tf_idf = load_npz('vectorized_data/X_test_bigram_tf_idf.npz')
X_test_unigram_tf_idf = load_npz('vectorized_data/X_test_unigram_tf_idf.npz')
#CLASSIFIERS
clf_SGDC_without_tuning = load('classifiers/clf_SGDC_without_tuning.joblib')
sgd_classifier = load('classifiers/sgd_classifier.joblib')
clf_random_forest_without_tuning = load('classifiers/clf_random_forest_without_tuning.joblib')
clf_random_forest = load('classifiers/random_forest_classifier.joblib')
clf_multinomialNB_without_tuning = load('classifiers/clf_multinomialNB_without_tuning.joblib')
clf_multinomialNB = load('classifiers/multinomialNB_classifier.joblib')
#LOADING THE LABELS
y_test = imdb_test['label'].values
y_train = imdb_train['label'].values
'''


# CREATING DATAFRAMES
def create_data_frame(folder: str) -> pd.DataFrame:
    
    #folder - the root folder of train or test dataset
    #Returns: a DataFrame with the combined data from the input folder
    
    pos_folder = f'{folder}/pos'  # positive reviews
    neg_folder = f'{folder}/neg'  # negative reviews

    def get_files(fld: str) -> list:
        
        #fld - positive or negative reviews folder
        #Returns: a list with all files in input folder
        
        return [join(fld, f) for f in listdir(fld) if isfile(join(fld, f))]

    def append_files_data(data_list: list, files: list, label: int) -> None:
        
        #Appends to 'data_list' tuples of form (file content, label)
        #for each file in 'files' input list

        for file_path in files:
            with open(file_path, 'r', encoding="utf8") as f:
                text = f.read()
                data_list.append((text, label))

    pos_files = get_files(pos_folder)
    neg_files = get_files(neg_folder)

    data_list = []
    append_files_data(data_list, pos_files, 1)
    append_files_data(data_list, neg_files, 0)
    shuffle(data_list)

    text, label = tuple(zip(*data_list))
    # replacing line breaks with spaces
    text = list(map(lambda txt: re.sub('(<br\s*/?>)+', ' ', txt), text))
    text = list(map(lambda txt: re.sub("[^a-zA-Z]"," ", txt), text))
    stops = set(stopwords.words("english"))
    for i in range(0,len(text)):
        words = text[i].split()
        words = [w for w in words if not w in stops]
        text[i] = " ".join(words)
    return pd.DataFrame({'text': text, 'label': label})

imdb_train = create_data_frame('aclImdb/train')
imdb_test = create_data_frame('aclImdb/test')

#LOADING THE LABELS
y_test = imdb_test['label'].values
y_train = imdb_train['label'].values

#CREATING SUB FODLERS
system("mkdir csv")
system("mkdir classifiers")
system("mkdir data_preprocessors")
system("mkdir vectorized_data")

imdb_train.to_csv('csv/imdb_train.csv', index=False)
imdb_test.to_csv('csv/imdb_test.csv', index=False)

# TEXT VECTORIZATION
# Unigram Counts
unigram_vectorizer = CountVectorizer(ngram_range=(1, 1))
unigram_vectorizer.fit(imdb_train['text'].values)
dump(unigram_vectorizer, 'data_preprocessors/unigram_vectorizer.joblib')

X_train_unigram = unigram_vectorizer.transform(imdb_train['text'].values)
save_npz('vectorized_data/X_train_unigram.npz', X_train_unigram)

# Unigram Tf-Idf
unigram_tf_idf_transformer = TfidfTransformer()
unigram_tf_idf_transformer.fit(X_train_unigram)
dump(unigram_tf_idf_transformer, 'data_preprocessors/unigram_tf_idf_transformer.joblib')

X_train_unigram_tf_idf = unigram_tf_idf_transformer.transform(X_train_unigram)
save_npz('vectorized_data/X_train_unigram_tf_idf.npz', X_train_unigram_tf_idf)

# Bigram Counts

bigram_vectorizer = CountVectorizer(ngram_range=(1, 2))
bigram_vectorizer.fit(imdb_train['text'].values)
dump(bigram_vectorizer, 'data_preprocessors/bigram_vectorizer.joblib')

X_train_bigram = bigram_vectorizer.transform(imdb_train['text'].values)
save_npz('vectorized_data/X_train_bigram.npz', X_train_bigram)

#Bigram Tf-Idf

bigram_tf_idf_transformer = TfidfTransformer()
bigram_tf_idf_transformer.fit(X_train_bigram)
dump(bigram_tf_idf_transformer, 'data_preprocessors/bigram_tf_idf_transformer.joblib')

X_train_bigram_tf_idf = bigram_tf_idf_transformer.transform(X_train_bigram)
save_npz('vectorized_data/X_train_bigram_tf_idf.npz', X_train_bigram_tf_idf)

#Test unigram counts
X_test_unigram = unigram_vectorizer.transform(imdb_test['text'].values)
save_npz('vectorized_data/X_test_unigram.npz', X_test_unigram)

#Test bigram counts
X_test_bigram = bigram_vectorizer.transform(imdb_test['text'].values)
save_npz('vectorized_data/X_test_bigram.npz', X_test_bigram)

#Test bigram tf_idf
X_test_bigram_tf_idf = bigram_tf_idf_transformer.transform(X_test_bigram)
save_npz('vectorized_data/X_test_bigram_tf_idf.npz', X_test_bigram_tf_idf)

#Test unigram tf_idf
X_test_unigram_tf_idf = unigram_tf_idf_transformer.transform(X_test_unigram)
save_npz('vectorized_data/X_test_unigram_tf_idf.npz', X_test_unigram_tf_idf)

#TRAINING THE SGDC MODELS

def train_and_show_scores_SGDC(X: csr_matrix, y: np.array, title: str) -> None:
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, train_size=0.75, stratify=y
    )

    clf = SGDClassifier()
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    valid_score = clf.score(X_valid, y_valid)
    print(f'{title}\nTrain score: {round(train_score, 2)} ; Validation score: {round(valid_score, 2)}\n')
print(f'Score for SGDC model on training data with cross validation')
train_and_show_scores_SGDC(X_train_unigram, y_train, 'Unigram Counts')
train_and_show_scores_SGDC(X_train_unigram_tf_idf, y_train, 'Unigram Tf-Idf')
train_and_show_scores_SGDC(X_train_bigram, y_train, 'Bigram Counts')
train_and_show_scores_SGDC(X_train_bigram_tf_idf, y_train, 'Bigram Tf-Idf')

#Saving classifier without hyper parameter
clf_SGDC_without_tuning = SGDClassifier()
clf_SGDC_without_tuning.fit(X_train_bigram_tf_idf,y_train)
dump(clf_SGDC_without_tuning, 'classifiers/clf_SGDC_without_tuning.joblib')

#HYPERTUNING THE MODEL
print(f'Hyper tuning the SGDC classifier')
#Our model works the best on bigram tf-idf thats why we will use it for hypertuning
X_train_SGDC = X_train_bigram_tf_idf

# Phase 1: loss, learning rate and initial learning rate

clf = SGDClassifier()

distributions = dict(
    loss=['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
    learning_rate=['optimal', 'invscaling', 'adaptive'],
    eta0=uniform(loc=1e-7, scale=1e-2)
)

random_search_cv = RandomizedSearchCV(
    estimator=clf,
    param_distributions=distributions,
    cv=5,
    n_iter=50,
    verbose=2
)
random_search_cv.fit(X_train_SGDC, y_train)
print(f'Best params: {random_search_cv.best_params_}')
print(f'Best score: {random_search_cv.best_score_}\n')

# Phase 2: penalty and alpha

clf = SGDClassifier()

distributions = dict(
    penalty=['l1', 'l2', 'elasticnet'],
    alpha=uniform(loc=1e-6, scale=1e-4)
)

random_search_cv = RandomizedSearchCV(
    estimator=clf,
    param_distributions=distributions,
    cv=5,
    n_iter=50,
    verbose=2
)
random_search_cv.fit(X_train_SGDC, y_train)
print(f'Best params: {random_search_cv.best_params_}')
print(f'Best score: {random_search_cv.best_score_}\n')

#SAVING THE BEST CLASIFIER
#Becouse in our first phase of searching the best estimatior we got learning rate optimal we can ignore eta0
sgd_classifier = random_search_cv.best_estimator_
dump(sgd_classifier, 'classifiers/sgd_classifier.joblib')


#RANDOM FOREST
def train_and_show_scores_random_forest(X: csr_matrix, y: np.array, title: str) -> None:
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, train_size=0.75, stratify=y
    )

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    valid_score = clf.score(X_valid, y_valid)
    print(f'{title}\nTrain score: {round(train_score, 2)} ; Validation score: {round(valid_score, 2)}\n')
    
print(f'Score for Random Forest model on training data with cross validation')
train_and_show_scores_random_forest(X_train_unigram, y_train, 'Unigram Counts')
train_and_show_scores_random_forest(X_train_unigram_tf_idf, y_train, 'Unigram Tf-Idf')
train_and_show_scores_random_forest(X_train_bigram, y_train, 'Bigram Counts')
train_and_show_scores_random_forest(X_train_bigram_tf_idf, y_train, 'Bigram Tf-Idf')

#SAVING CLASSIFIER WITHOUT TUNING
clf_random_forest_without_tuning = RandomForestClassifier()
clf_random_forest_without_tuning.fit(X_train_unigram,y_train)

dump(clf_random_forest_without_tuning, 'classifiers/clf_random_forest_without_tuning.joblib')


#HYPER TUNING THE MOEDEL
print(f'Hyper tuning the Random Forest classifier')
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
#Our model works the best on unigram thats why we will use it for hypertuning
X_train_random_forest = X_train_unigram

clf = RandomForestClassifier()
clf_random = RandomizedSearchCV(estimator=clf, param_distributions = random_grid, n_iter = 10, cv = 3, verbose=2, random_state=42, n_jobs = -1)
clf_random.fit(X_train_random_forest, y_train)
print(f'Best Params:{clf_random.best_params_}')
print(f'Best score: {clf_random.best_score_}\n')

#SAVING THE BEST RANDOM FOREST CLASSIFIER
clf_random_forest = clf_random.best_estimator_
dump(clf_random.best_estimator_, 'classifiers/random_forest_classifier.joblib')


#MULTINOMIAL NAIVE BAYES
def train_and_show_scores_naive_bayes(X: csr_matrix, y: np.array, title: str) -> None:
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, train_size=0.75, stratify=y
    )

    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    valid_score = clf.score(X_valid, y_valid)
    print(f'{title}\nTrain score: {round(train_score, 2)} ; Validation score: {round(valid_score, 2)}\n')

print(f'Score for MultinomialNB model on training data with cross validation')
train_and_show_scores_naive_bayes(X_train_unigram, y_train, 'Unigram Counts')
train_and_show_scores_naive_bayes(X_train_unigram_tf_idf, y_train, 'Unigram Tf-Idf')
train_and_show_scores_naive_bayes(X_train_bigram, y_train, 'Bigram Counts')
train_and_show_scores_naive_bayes(X_train_bigram_tf_idf, y_train, 'Bigram Tf-Idf')

#Saving classifier without hyper parameter
clf_multinomialNB_without_tuning = MultinomialNB()
clf_multinomialNB_without_tuning.fit(X_train_bigram_tf_idf,y_train)
dump(clf_multinomialNB_without_tuning, 'classifiers/clf_multinomialNB_without_tuning.joblib')


#Hyper Tuning MultinomialNB
print(f'Hyper tuning the Multinomial Naive Bayes classifier')
#Our model works the best on bigram tf-idf
X_train_multinomialNB = X_train_bigram_tf_idf

alpha = [float(x) for x in np.linspace(0.0000000001, 1.0, num = 5)]
fit_prior = [True, False]

tuned_parameters = {
    'alpha': alpha,
    'fit_prior': fit_prior
}

clf_multinomialNB = MultinomialNB()
clf_multinomialNB = GridSearchCV(clf_multinomialNB, tuned_parameters, cv=10, verbose=1, n_jobs=-1)
clf_multinomialNB.fit(X_train_multinomialNB,y_train)
print(f'Best Params:{clf_multinomialNB.best_params_}')
print(f'Best score: {clf_multinomialNB.best_score_}\n')

#SAVING THE BEST MULTINOMIAL NAIVE BAYES CLASSIFIER
dump(clf_multinomialNB.best_estimator_, 'classifiers/multinomialNB_classifier.joblib')

#Testing the models

print(f'Score of SGDC linear classifier without tuning: {clf_SGDC_without_tuning.score(X_test_bigram_tf_idf,y_test)}')
print(f'Score of SGDC linear classifier with hyperparameter: {sgd_classifier.score(X_test_bigram_tf_idf, y_test)}')
print(f'Score of Random Forest classifier without tuning: {clf_random_forest_without_tuning.score(X_test_unigram,y_test)}')
print(f'Score of Random Forest classifier with hyperparameter: {clf_random_forest.score(X_test_unigram,y_test)}')
print(f'Score of MultinomialNB classifier without tuning: {clf_multinomialNB_without_tuning.score(X_test_bigram_tf_idf,y_test)}')
print(f'Score of MultinamialNB classifier with hyper parameter: {clf_multinomialNB.score(X_test_bigram_tf_idf, y_test)}')
