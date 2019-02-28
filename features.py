import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
from nltk.stem import PorterStemmer
import string

def getStructuralFeatures(utterance, startingUser, numUtterances):
    "Returns a vector with the structural features of an utterance."
    sentence = utterance["utterance"]
    # Stopword removal
    stop = set(stopwords.words('english'))
    wordTokens = word_tokenize(sentence) # remove punctuation
    filteredSentence = [w for w in wordTokens if not w in stop] 
    
    # Calculate features
    pos = getAbsolutePosition(utterance);
    posNorm = getNormalizedPosition(utterance, numUtterances);
    numWords = getNumWords(filteredSentence);
    numWordsUnique = getNumWordsUnique(filteredSentence)
    starter = isStarter(utterance, startingUser)
    return [pos] + [posNorm] + [numWords] + numWordsUnique + [starter];

def isStarter(utterance, startingUser):
    "Returns true if the sender of the utterance is the starting user of the dialog."
    if startingUser == utterance["user_id"]:
        return True
    return False

def getNumWords(filteredSentence):
    "Returns the number of words of a sentence after stopword removal."
    return len(filteredSentence)

def getNumWordsUnique(filteredSentence):
    "Returns the number of unique words in a sentence after stopword removal\
    and the number of unique words in a sentence after stemming."
    porterS = PorterStemmer()
    filteredSet = set(filteredSentence)
    stemmedSet = set(list([porterS.stem(filteredWord) for filteredWord in list(filteredSet)]))
    return [len(filteredSet), len(stemmedSet)]

def getAbsolutePosition(utterance):
    "Returns the absolute position of an utterance within a dialog."
    return utterance['utterance_pos'];

def getNormalizedPosition(utterance, numUtterances):
    "Returns the normalized position of an utterance wtihin a dialog."
    return utterance['utterance_pos'] / numUtterances;

def getSentimentFeatures(utterance):
    "Returns sentiment feature vector for an utterance dictionary."
    sentence = utterance["utterance"]
    
    # Compute features
    thank = containsThank(sentence);
    exPoint = containsExclamationPoint(sentence);
    feedback = containsFeedback(sentence);
    vader = getVaderSentimentScores(sentence);
    posNeg = getNumPosNegWords(sentence);
    return [thank] + [exPoint] + [feedback] + vader + posNeg;

def getVaderSentimentScores(sentence):
    "Returns vector with VADER sentiment scores: negative, neutral, positive."
    analyzer = SentimentIntensityAnalyzer()
    vs = analyzer.polarity_scores(sentence)
    return [vs["neg"], vs["neu"], vs["pos"]]

def containsExclamationPoint(sentence):
    "Returns true if an utterance contains an exclamation point."
    if "!" in sentence:
        return True
    return False
    
def containsThank(sentence):
    "Returns true if an utterance contains 'thank'."
    if "thank" in sentence:
        return True
    return False
    
def containsFeedback(sentence):
    "Returns true if an utterance contains 'does not' or 'did not'."
    if "does not" in sentence or "did not" in sentence:
        return True
    return False
    
def getNumPosNegWords(sentence):
    "Returns number of positive and negative words in utterance."
    posFile = open("C:/Users/nele2/Documents/IR/IR Core Project/positive-words.txt").read();
    negFile = open("C:/Users/nele2/Documents/IR/IR Core Project/negative-words.txt").read();
    numPos = 0;
    numNeg = 0;
    
    table = str.maketrans(dict.fromkeys(string.punctuation))
    sentence = sentence.translate(table)  
    for word in sentence.split():
        if word in posFile:
            numPos += 1;
        elif word in negFile:
            numNeg += 1;
    return [numPos, numNeg];

with open('C:/Users/nele2/Documents/IR/IR Core Project/MSDialog-Intent.json') as json_file:  
    data = json.load(json_file)

for dialogID in data.keys():
    startingUtterance = data[dialogID]["utterances"][0]
    startingUser = startingUtterance["user_id"] # User who starts the dialog
    numUtterances = len(data[dialogID]["utterances"]) # number of utterances in dialog
    for dialogPartID in data[dialogID]["utterances"]:
        
        sentiment = getSentimentFeatures(dialogPartID)
        structural = getStructuralFeatures(dialogPartID, startingUser, numUtterances)
        print("Sentiment:", sentiment)
        print("Structural:", structural)
    