'''Create data class with review'''
import random

class Sentiment:
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    POSITIVE = "POSITIVE"

class Review:
    def __init__(self, text, score):
        self.text = text
        self.score = score
        self.sentiment = self.get_sentiment()

    def get_sentiment(self):
        if self.score <= 2:
            return Sentiment.NEGATIVE
        elif self.score == 3:
            return Sentiment.NEUTRAL
        else: #Score of 4 or 5
            return Sentiment.POSITIVE

class ReviewContainer:
    def __init__(self, reviews):
        self.reviews = reviews

    def get_text(self):
        return [x.text for x in self.reviews]

    def get_sentiment(self):
        return [x.sentiment for x in self.reviews]

    def evenly_distribute(self):
        negative = list(filter(lambda x: x.sentiment == Sentiment.NEGATIVE, self.reviews))
        positive = list(filter(lambda x: x.sentiment == Sentiment.POSITIVE, self.reviews))
        positive_shrunk = positive[:len(negative)]
        self.reviews = negative + positive_shrunk
        random.shuffle(self.reviews)

##        print(negative[0].text)
##        print(len(negative))
##        print(len(positive))

'''Import data'''
import json

file_name = 'Books_small_10000.json'

reviews = []
with open(file_name) as f:
    for line in f:
        review = json.loads(line)
        reviews.append(Review(review['reviewText'],review['overall']))

##print(reviews[5].score)
##print(reviews[5].sentiment)

'''Prep Data'''
from sklearn.model_selection import train_test_split

training, test = train_test_split(reviews, test_size=0.33, random_state=42)

train_container = ReviewContainer(training)
test_container = ReviewContainer(test)

##print(len(training))
##print(len(test))
##print(training[0].sentiment)
train_container.evenly_distribute()
train_x = train_container.get_text()
train_y = train_container.get_sentiment()

test_container.evenly_distribute()
test_x = test_container.get_text()
test_y = test_container.get_sentiment()

##print(train_y.count(Sentiment.POSITIVE))
##print(train_y.count(Sentiment.NEGATIVE))

##print(train_x[0])
##print(train_y[0])
##print(test_x[0])
##print(test_y[0])

'''Bag of words vectorization'''
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
##TfidVectorizer gives more weight to less frequently used words and viceversa

##vectorizer = CountVectorizer()
vectorizer = TfidfVectorizer()
train_x_vectors = vectorizer.fit_transform(train_x)

test_x_vectors = vectorizer.transform(test_x)

##print(train_x[0])
##print(train_x_vectors[0].toarray())


'''Classification'''
'''Linear SVM'''
from sklearn import svm
clf_svm = svm.SVC(kernel = 'linear')

clf_svm.fit(train_x_vectors, train_y)
##print(test_x[0])
##print(test_x_vectors[0])
svm_classified = clf_svm.predict(test_x_vectors[0])
print(svm_classified)

'''Decision Tree'''
from sklearn.tree import DecisionTreeClassifier

clf_dec = DecisionTreeClassifier()
clf_dec.fit(train_x_vectors, train_y)

tree_classified = clf_dec.predict(test_x_vectors[0])
print(tree_classified)

'''Naive Bayes'''
from sklearn.naive_bayes import GaussianNB

clf_gnb = DecisionTreeClassifier()
clf_gnb.fit(train_x_vectors, train_y)

gnb_classified = clf_gnb.predict(test_x_vectors[0])
print(gnb_classified)

'''Logistic Regression'''
from sklearn.linear_model import LogisticRegression

clf_log = LogisticRegression()
clf_log.fit(train_x_vectors, train_y)

log_classified = clf_log.predict(test_x_vectors[0])
print(log_classified)

'''Evaluation'''
##Mean Accuracy
print(clf_svm.score(test_x_vectors, test_y))
print(clf_dec.score(test_x_vectors, test_y))
print(clf_gnb.score(test_x_vectors, test_y))
print(clf_log.score(test_x_vectors, test_y))

## F1 Scores
from sklearn.metrics import f1_score

f1_score_svm = f1_score(test_y, clf_svm.predict(test_x_vectors), average = None, labels=[Sentiment.POSITIVE, Sentiment.NEGATIVE])
f1_score_dec = f1_score(test_y, clf_dec.predict(test_x_vectors), average = None, labels=[Sentiment.POSITIVE, Sentiment.NEGATIVE])
f1_score_gnb = f1_score(test_y, clf_gnb.predict(test_x_vectors), average = None, labels=[Sentiment.POSITIVE, Sentiment.NEGATIVE])
f1_score_log = f1_score(test_y, clf_log.predict(test_x_vectors), average = None, labels=[Sentiment.POSITIVE, Sentiment.NEGATIVE])

print(f1_score_svm)
print(f1_score_dec)
print(f1_score_gnb)
print(f1_score_log)
##Predicting positives well, but negatives poorly

##print(train_y.count(Sentiment.POSITIVE))
##print(train_y.count(Sentiment.NEGATIVE))

'''Out of Sample Test'''
test_set = ['I thoroughly enjoyed this, 5 stars', 'bad book do not buy', 'horrible waste of time']
new_test = vectorizer.transform(test_set)

new_test_pred = clf_svm.predict(new_test)
##print(new_test_pred)

'''Fine tuning'''
##from sklearn.model_selection import GridSearchCV
##
##parameters = {'kernel':('linear', 'rbf'), 'C': (1,4,8,16,32)}
##svc = svm.SVC()
##clf = GridSearchCV(svc, parameters, cv = 5)
##clf_fit = clf.fit(train_x_vectors, train_y)
##clf_best_est = clf_fit.best_estimator_
##clf_params = clf_fit.best_params_
##print(clf_best_est)
##print(clf_params)

#Look at mean accuracy with optimized value
##print(clf.score(test_x_vectors, test_y))


'''Saving Model'''
##import pickle
##
##with open('sentiment_classifier.pkl', 'wb') as f:
##    pickle.dump(clf, f)

'''Load model'''
##with open('sentiment_classifier.pkl', 'rb') as f:
##    loaded_clf = pickle.load(f)
##
##print(test_x[0])
##print(loaded_clf.predict(test_x_vectors[0]))
##

'''Confusion Matrix'''
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib as plt

y_pred = clf_svm.predict(test_x_vectors)

cm = confusion_matrix(test_y, y_pred)
##df_cm = pd.DataFrame(c, index = reverse(labels), columns = labels)
##
print(sn.heatmap(cm, annot=True, fmt = 'd'))
##plt.show()
