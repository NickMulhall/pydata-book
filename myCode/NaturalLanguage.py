import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import PunktSentenceTokenizer
from nltk.corpus import names
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import names
# from https://pythonspot.com/tokenizing-words-and-sentences-with-nltk/
# covering tokenisation, stopwords, steming, speech tagging, prediction and sentiment analysis
# note, must open Python interactive in VS and (1) import nltk and (2) nltk.download() execute both statements
# functions for this Python program !
def gender_features(word): 
    return {'last_letter': word[-1]} 
def word_feats(words):
    return dict([(word, True) for word in words])
# tokenise words
data = "All work and no play makes jack dull boy, all work and no play makes jack a dull boy."
print(word_tokenize(data))
# tokenise sentences
data = "All work and no play makes jack dull boy. All work and no play makes jack a dull boy."
print(sent_tokenize(data))
# you can store words and sentences in arrays
data = "All work and no play makes jack dull boy. All work and no play makes jack a dull boy."
phrases = sent_tokenize(data)
words = word_tokenize(data)
print(phrases)
print(words)
# text may contain stop words such as 'the', 'is' and 'are'. You can filter stop words by language
data = "All work and no play makes jack dull boy. All work and no play makes jack a dull boy."
stopWords = set(stopwords.words('english'))
words = word_tokenize(data)
wordsFiltered = []
for w in words:
    if w not in stopWords:
        wordsFiltered.append(w)
print(wordsFiltered)
# steming e.g. wait = waited, waits, waiting
words = ["gaming","gamed","games"]
ps = PorterStemmer()
for word in words:
    print(ps.stem(word))
# steming sentenses
ps = PorterStemmer()
sentence = "gaming, the gamers play games"
words = word_tokenize(sentence)
for word in words:
    print(word + ":" + ps.stem(word))
# tagging speech, outputs a tuple for each word with a speech code
document = 'Whether you\'re new to programming or an experienced developer, it\'s easy to learn and use Python.'
sentences = nltk.sent_tokenize(document)   
for sent in sentences:
    print(nltk.pos_tag(nltk.word_tokenize(sent)))
# filter the speech based on the type of word - the classes include past and present tenses
document = 'Today the Netherlands celebrates King\'s Day. To honor this tradition, the Dutch embassy in San Francisco invited me to'
sentences = nltk.sent_tokenize(document)   
data = []
for sent in sentences:
    data = data + nltk.pos_tag(nltk.word_tokenize(sent))
for word in data: 
    if 'NNP' in word[1]: 
        print(word)
# nlp predictions - 4 steps
# step 1 - data preparation load data and training. a collection of tuples included in nltk, you can append also
names = ([(name, 'male') for name in names.words('male.txt')] + 
	 [(name, 'female') for name in names.words('female.txt')])
# step 2 - feature extraction - based on the DS prepare our feature using the last letter of the name
featuresets = [(gender_features(n), g) for (n,g) in names] 
# step 3 - train
train_set = featuresets
classifier = nltk.NaiveBayesClassifier.train(train_set) 
# step 4 - predict gender
#name = input("Enter a name: ")
name = "Hayley"
print(classifier.classify(gender_features(name)))
# sentiment analysis - data is feed into a classifier from which a prediction is made
# the two steps are training and prediction
# define 3 classes for positive, negative and neutral
positive_vocab = [ 'awesome', 'outstanding', 'fantastic', 'terrific', 'good', 'nice', 'great', ':)' ]
negative_vocab = [ 'bad', 'terrible','useless', 'hate', ':(' ]
neutral_vocab = [ 'movie','the','sound','was','is','actors','did','know','words','not' ]
# convert each word into a feature using a simplified bag of words model
positive_features = [(word_feats(pos), 'pos') for pos in positive_vocab]
negative_features = [(word_feats(neg), 'neg') for neg in negative_vocab]
neutral_features = [(word_feats(neu), 'neu') for neu in neutral_vocab]
# training set is the sum of these three feature sets
train_set = negative_features + positive_features + neutral_features
# train the classifier
classifier = NaiveBayesClassifier.train(train_set) 
# Finally, prediction based on a reviewers sentence, by example
# The larger the dataset then the more accurate the prediction is !
neg = 0
pos = 0
sentence = "Awesome movie, I liked it"
sentence = sentence.lower()
words = sentence.split(' ')
for word in words:
    classResult = classifier.classify( word_feats(word))
    if classResult == 'neg':
        neg = neg + 1
    if classResult == 'pos':
        pos = pos + 1
print('Positive: ' + str(float(pos)/len(words)))
print('Negative: ' + str(float(neg)/len(words)))


