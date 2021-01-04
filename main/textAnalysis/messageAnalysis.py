from main.textAnalysis import utils as utils
import numpy as np

from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.svm import SVC
import joblib
from random import shuffle


'''Uncomment the comment below when training and/or testing sentiment model'''
# contacts = readTexts.readTextsfromXml('sms.xml')

unigramLexicon = utils.loadUnigramLexicon()


def tokenizeTexts(contacts):
    for contactName in contacts:
        contactHistory = contacts[contactName]

        outgoingTexts = contactHistory.outgoing.tokenized
        incomingTexts = contactHistory.incoming.tokenized

        return incomingTexts, outgoingTexts


def getWordScore(word):
    wordScore = 0
    if word in unigramLexicon:
        return True, unigramLexicon[word]
    return False, wordScore


def getWordSentiment(word):
    isInLexicon, wordScore = getWordScore(word)
    if isInLexicon:
        return wordScore
    else:
        # attempts to get word score via word synonyms if word not in lexicon
        synonyms = utils.getSynonyms(word)
        # if word has no synonyms return 0 for word score
        if len(synonyms) == 0:
            return 0
        wordScores = []
        for synonym in synonyms:
            _, wordScore = getWordScore(synonym)
            wordScores.append(wordScore)
        averageWordScore = sum(wordScores)/len(wordScores)
        return averageWordScore


def getFeatureVector(sample):
    wordScores = []
    for word in sample:
        wordScore = getWordSentiment(word)
        wordScores.append(wordScore)

    numWords = len(sample)

    fnList1 = [
        np.std,
        np.mean,
    ]

    if numWords == 0:
        featList = [0, 0]
    else:
        featList = [func(wordScores) for func in fnList1]
    featList = featList + utils.tfidfVectorizer(sample)
    return featList


def featurizeInput(samples):
    out = []
    for sample in samples:
        fv = getFeatureVector(sample)
        out.append(fv)
    out = np.array(out)
    return out


# returns list containing features for each positive sample
def loadPositive():
    positiveSamples = utils.loadPositivesamples()
    fv = featurizeInput(positiveSamples)
    return fv


# returns list containing features for each negative sample
def loadNegative():
    negativeSamples = utils.loadNegativeSamples()
    fv = featurizeInput(negativeSamples)
    return fv


# returns two equally sized lists
def makeSameSize(smallerList, biggerList):
    shuffle(smallerList)
    return smallerList[:len(biggerList)]


def preprocces(data):
    scaler = preprocessing.StandardScaler().fit(data)
    joblib.dump(scaler, f'Standard scaler.pkl')
    return scaler.transform(data)


def setLabelsAndData(useStored=False, store=True):
    print("Gathering Labels and data.")
    # use stored features if specified
    if useStored:
        data = np.load('main/textAnalysis/data.npy', allow_pickle=True)
        labels = np.load('main/textAnalysis/labels.npy', allow_pickle=True)
        return data, labels
    # load negative/ positive classes
    negative = loadNegative()
    positive = loadPositive()
    # make size of positive/ negative sets the same
    positive = makeSameSize(smallerList=positive, biggerList=negative)

    labels = []
    data = []
    textTypes = [positive, negative]
    # store features for quicker testing if specified
    for i, textType in enumerate(textTypes):
        labels.extend([i]*len(textType))
        # add each sample from each text type to collective data set
        for sample in textType:
            data.append(sample)
    preprocces(data)
    if store:
        np.save('main/textAnalysis/data.npy', np.array(data))
        np.save('main/textAnalysis/labels.npy', np.array(labels))
    return np.array(data), np.array(labels)


# calculates the mean accuracy for a given classifier over a number of trials
def getAccuracy(classifier, data, labels):
    testScores = []
    # set up container for class level results of each classification trial
    cv = KFold(n_splits=10, random_state=65, shuffle=True)
    # make predictions
    for train_index, test_index in cv.split(data):
        dataTrain, dataTest, labelsTrain, labelsTest = data[train_index], data[test_index], labels[train_index], labels[test_index]
        classifier.fit(dataTrain, labelsTrain)
        joblib.dump(classifier, f'main/textAnalysis/{str(classifier)}.pkl')
        testScores.append(classifier.score(dataTest, labelsTest))
    return np.mean(testScores)


# returns the accuracy for a series of classifiers
def classify(useStored=True, store=True):
    data, labels = setLabelsAndData(useStored=useStored, store=store)
    print('Classifying....')
    clfs = [GaussianNB(), LogisticRegression(max_iter=200)]
    # initializes dictionary that will contain classifier as a key and accuracy as a value
    accuracies = dict()
    # retrieves accuracy of each classifier
    for clf in clfs:
        print(f'Using {clf}')
        accuracy = getAccuracy(clf, data, labels)
        accuracies[str(clf)] = accuracy

    return accuracies


# accuracies = classify(useStored=False, store=True)
#
# for clf in accuracies:
#     print(f"{clf} Mean Accuracy: {accuracies[clf]}")

#
# data = np.load('data.npy', allow_pickle=True)
# print(data)
#
# print(np.where(np.isnan(data)))


def isPositive(sample, clf, scaler):
    print(f'Predicting {sample}')
    fv = featurizeInput([sample])
    fv = scaler.transform(np.array(fv))
    pred = clf.predict_proba(fv)
    confidence = round(max(pred[0][0], pred[0][1]), 2)*100
    if pred[0][1]<pred[0][0]:
        return True, confidence
    else:
        return False, confidence


def experimentalIsPositive(sample):
    sample = utils.getTokens(sample)
    sample = utils.processText(sample)
    clf = joblib.load('main/textAnalysis/LogisticRegression().pkl')
    scaler = joblib.load('main/textAnalysis/Standard scaler.pkl')
    print(f'Predicting {sample}')
    fv = featurizeInput([sample])
    fv = scaler.transform(np.array(fv))
    pred = clf.predict_proba(fv)
    confidence = round(max(pred[0][0], pred[0][1]), 2) * 100
    if pred[0][1] < pred[0][0]:
        return 'Positive', confidence, sample
    else:
        return 'Negative', confidence, sample

