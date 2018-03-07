import json
import datetime
from collections import defaultdict
from matplotlib import pyplot as py
import matplotlib
import csv
import string
from sklearn.feature_extraction.text import TfidfVectorizer as tfidfV
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

NGRAM_RNG = (1,2)
MAX_FEAT  = 2000
#read data
cmtsDict = {}
cmts = []
prices = {}
posts = {}
postDate = {} #post: date
# prices [date] -> pricedata
cmtsDate = defaultdict(str) #date: cmts
upvotes = defaultdict(int)

#train on Jamacian data
with open("../GRLC_All_time_data.csv") as inFile:
	rows = csv.reader(inFile)
	first = False
	for row in rows:
		if not first:
			first = True
			continue
		prices[ datetime.datetime.strptime(row[0],"%b %d, %Y").date().strftime("%m-%d-%y") ] = {'Open':row[1],'High':row[2],'Low':row[3],'Close':row[4],'Volume':row[5],'Market Cap':row[6] }

with open('garlicoin_top_new.json','r') as infile:
	posts = json.loads(infile.read())

with open('garlicoin_cmts_new.json','r') as infile:
	cmtsDict = json.loads(infile.read())

for p in prices:
	upvotes[p] = 0

for p in posts:
	da = datetime.date.fromtimestamp( int(posts[p]['created']) ).strftime('%m-%d-%y') 
	postDate[p] = da
	upvotes[da] += posts[p]['score']

for cmt in cmtsDict:
	aStr = ""
	for cd in cmtsDict[cmt]:
		aList = ''.join([c for c in cd['body'].strip() if not c in string.punctuation]).split()
		if aList == ['deleted']:
			continue
		aList = [c for c in aList if 'http' not in c]
		aStr += ' '.join(aList)
		aStr += " "
	pd = postDate[cmt]
	cmtsDate[pd] += aStr

#make tfidf vectorizer
cmts = zip(cmtsDate.keys(),cmtsDate.values())
cmts.sort()
cmtsOnly = zip(*cmts)[1]

tfV = tfidfV(ngram_range=NGRAM_RNG, max_features=MAX_FEAT, stop_words='english',strip_accents='unicode')
tfidfData = tfV.fit_transform(cmtsOnly).toarray()

#1, # of upvotes, todays price, tfidf[:2000]
def feature(date):

	feat = [1]
	feat.append(upvotes[date])
	feat.append(float(prices[date]['Open']))
	#feat.append(float(prices[date]['Close']))
	feat += tfidfData[ Xdates.index(date) ].tolist()
	return feat

Xdates = list(zip(*cmts)[0])
Xdates = [d for d in Xdates if d in prices]
X = [feature(d) for d in Xdates]
Y = []
for d in Xdates:
	newDate = datetime.datetime.strptime(d,"%m-%d-%y").date() + datetime.timedelta(days=1)
	Y.append( float(prices[ newDate.strftime("%m-%d-%y") ]['Close']) )

#clf = linear_model.Ridge(1.0, fit_intercept=False)
clf = linear_model.LinearRegression(fit_intercept=False)
clf.fit(X, Y)
theta = clf.coef_
predictions = clf.predict(X)
print("MSE on train tfidf:" + str(mean_squared_error(Y,predictions)))

weights = theta.tolist()[3:]
words = tfV.get_feature_names()

weightIList = zip( weights, range(len(weights)))
weightIList.sort()
weightIList.reverse()

print("top 10 weights")
for w in weightIList[:10]:
	print(str(words[ w[1] ])+ " "+ str(w[0])  )

print("bottom 10 weights")
for w in weightIList[-10:]:
	print(str(words[ w[1] ])+ " "+ str(w[0])  )

'''
#test with nascar 
with open("../DOGE_Nascar_data.csv") as inFile:
	rows = csv.reader(inFile)
	first = False
	for row in rows:
		if not first:
			first = True
			continue
		prices[ datetime.datetime.strptime(row[0],"%b %d, %Y").date().strftime("%m-%d-%y") ] = {'Open':row[1],'High':row[2],'Low':row[3],'Close':row[4],'Volume':row[5],'Market Cap':row[6] }

with open('nascar_top_new.json','r') as infile:
	posts = json.loads(infile.read())

with open('nascar_cmts_new.json','r') as infile:
	cmtsDict = json.loads(infile.read())

for p in prices:
	upvotes[p] = 0

for p in posts:
	da = datetime.date.fromtimestamp( int(posts[p]['created']) ).strftime('%m-%d-%y') 
	postDate[p] = da
	upvotes[da] += posts[p]['score']

for cmt in cmtsDict:
	aStr = ""
	for cd in cmtsDict[cmt]:
		
		aList = ''.join([c for c in cd['body'].strip() if not c in string.punctuation]).split()
		
		#for a in aList:
		#		if len(a) > 15: #most likely a link
		#		aList.remove(a)
		
		aStr += ' '.join(aList)
		aStr += " "
	#if cmt not in postDate:
	#c	continue
	pd = postDate[cmt]
	cmtsDate[pd] += aStr


#make tfidf vectorizer
cmts = zip(cmtsDate.keys(),cmtsDate.values())
cmts.sort()
cmtsOnly = zip(*cmts)[1]

nascarTfidfData = tfV.transform(cmtsOnly).toarray()
def feature(date):
	feat = [1]
	feat.append(upvotes[date])
	feat.append(float(prices[date]['Open']))
	#feat.append(float(prices[date]['Close']))
	feat += nascarTfidfData[ Xdates.index(date) ].tolist()
	return feat

Xdates = list(zip(*cmts)[0])
X = [feature(d) for d in Xdates]
Y = []
for d in Xdates:
	newDate = datetime.datetime.strptime(d,"%m-%d-%y").date() + datetime.timedelta(days=1)
	Y.append( float(prices[ newDate.strftime("%m-%d-%y") ]['Close']) )

predictions = clf.predict(X)
print("MSE on test tfidf:" + str(mean_squared_error(Y,predictions)))
#1, # of upvotes, todays price, tfidf[:2000]
'''
