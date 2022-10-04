import pandas as pd
import os
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import re
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

dir = "D:/NUS/Sem5/CS3244/Project/"
train_data = "train-balanced-sarcasm.csv"

def clean(input):
  #convert to lower case
  cleaned = input.lower()
  return cleaned

if __name__ == "__main__":
    df = pd.read_csv(os.path.join(dir + train_data))
    #IMPORTANT, MUST DROP ROWS WITH NANS
    df = df.dropna()

    X = df["comment"]
    Y = df["label"]

    # label: 0 is not sarcastic, 1 is sarcastic
    #TODO: TRAIN, TEST, VALIDATION SPLIT
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=42)

    X_train = X_train.values
    X_test = X_test.values
    Y_train = Y_train.values
    Y_test = Y_test.values

    #### Feature Extraction #####
    vec = CountVectorizer(binary=True, ngram_range=(2,2))
    vec = vec.fit(X_train)
    X_train = vec.transform(X_train)
    X_test = vec.transform(X_test)
    ######################################

    ####### Logistic Regression ##########
    # https://towardsdatascience.com/logistic-regression-using-python-sklearn-numpy-mnist-handwriting-recognition-matplotlib-a6b31e2b166a
    sk_log_reg = LogisticRegression(verbose=1, solver='sag', penalty='l2', max_iter=40)
    model = sk_log_reg.fit(X_train, Y_train)

    predictions = model.predict(X_test)
    accuracy = model.score(X_test, Y_test)
    print("Accuracy", accuracy)
    conf_mat = metrics.confusion_matrix(Y_test, predictions)
    print("Confusion Matrix: \n", conf_mat)



# import pandas as pd
# import os
# from sklearn.model_selection import train_test_split
# import numpy as np
# from sklearn.feature_extraction.text import CountVectorizer
# # from nltk.tokenize import word_tokenize
# # from nltk.corpus import stopwords
# # from string import punctuation
# # from nltk.stem import PorterStemmer
#
# dir = "D:/NUS/Sem5/CS3244/Project/"
# train_data = "train-balanced-sarcasm.csv"
#
# # Function taken from: https://www.kaggle.com/code/tomgglynnejones/logistic-regression-models-with-1-and-2-grams
# # def text_to_stemmed(input_text):
# #     '''Dependencies: ntlk.tokenize.word_tokenize, ntlk.corpus.stopwords,
# #     ntlk.stem.PorterStemmer, string.punctuation must be imported'''
# #     tokenized = word_tokenize(input_text)
# #     stopwords_list = stopwords.words('english') + list(punctuation)
# #     stripped = [word.lower() for word in tokenized if word.lower() not in stopwords_list]
# #     stemmed = [PorterStemmer().stem(word) for word in stripped]
# #     space = ' '
# #     stemmed_str = space.join(stemmed)
# #     return stemmed_str
#
# # Function taken from: https://practicaldatascience.co.uk/machine-learning/how-to-use-count-vectorization-for-n-gram-analysis
# def get_ngrams(text, ngram_from=2, ngram_to=2, n=None, max_features=9999999):
#     vec = CountVectorizer(ngram_range=(ngram_from, ngram_to),
#                           max_features=max_features,
#                           stop_words='english').fit(text)
#     bag_of_words = vec.transform(text)
#     sum_words = bag_of_words.sum(axis=0)
#     words_freq = [(word, sum_words[0, i]) for word, i in vec.vocabulary_.items()]
#     words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
#
#     return words_freq[:n]
#
# if __name__ == "__main__":
#     df_data = pd.read_csv(os.path.join(dir + train_data))
#     # print("Distribution of labels in training data:")
#     # print(df_train["label"].value_counts())
#
#     #No need exact timestamp, data is sufficient to correlate era with sarcasm
#     df_data = df_data.drop("created_utc", axis=1)
#     #Author on reddit shoudlnt correlate with sarcasm
#     df_data = df_data.drop("author", axis=1)
#     # print(df_data.head())
#
#     #label: 0 is not sarcastic, 1 is sarcastic
#     #Train & Test split
#     #Need to convert data into Unicode or String: https://stackoverflow.com/questions/39303912/tfidfvectorizer-in-scikit-learn-valueerror-np-nan-is-an-invalid-document
#     X_train, X_test, Y_train, Y_test = train_test_split(df_data["comment"].values.astype('str'), df_data["label"].values.astype('str'), test_size = 0.5, random_state=42)
#     X_train = list(X_train)
#     X_test = list(X_test)
#     Y_train = list(Y_train)
#     Y_test = list(Y_test)
#
#
#
#     #Feature Extraction - Bag of Bigrams gave best accuracy in the paper (https://arxiv.org/pdf/1704.05579.pdf)
#     #Use the above as a baseline model
#
#     #Cleaning the comments using NLTK
#     # train_df['comment'] = train_df['comment'].apply(text_to_stemmed)
#
#     #Implement Bigram: https://practicaldatascience.co.uk/machine-learning/how-to-use-count-vectorization-for-n-gram-analysis
#     freq = get_ngrams(X_train)
#     print(freq)
#
#
