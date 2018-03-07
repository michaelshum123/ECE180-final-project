import json
import pandas
import datetime
from collections import defaultdict
from matplotlib import pyplot as py
import matplotlib
import csv

### JAMACIAN BOBSLEDING TEAM!!!
'''
prices = {}
posts = []
cmts  = []


with open("../DOGE_Jamaica_Olympics_data.csv") as inFile:
	rows = csv.reader(inFile)
	first = False
	for row in rows:
		if not first:
			first = True
			continue
		prices[ datetime.datetime.strptime(row[0],"%b %d, %Y").date().strftime("%m-%d-%y") ] = {'Open':row[1],'High':row[2],'Low':row[3],'Close':row[4],'Volume':row[5],'Market Cap':row[6] }

with open("dogecoin_bobsled.json","w") as outFile:
	json.dumps(prices, outFile)

with open("bobsled_top_new.json") as inFile:
	posts = json.loads(inFile.read())

#with open("bobsled_cmts_new.json") as inFile:
#	cmts = json.loads(inFile.read())

upvotes = defaultdict(int)
for p in prices:
	upvotes[p] = 0

for p in posts:
	da = datetime.date.fromtimestamp( int(posts[p]['created']) ).strftime('%m-%d-%y') 
	upvotes[da] += posts[p]['score']

wow = zip(upvotes.keys(), upvotes.values())
wow.sort()

Xtext = list(zip(*wow)[0])
upvoteY = list(zip(*wow)[1])
volumeY = [int(prices[r]['Volume']) for r in Xtext]
scale = max(volumeY)/max(upvoteY)
upvoteY = [scale*y for y in upvoteY]

X = [datetime.datetime.strptime(d, '%m-%d-%y').date() for d in Xtext]

py.plot_date(X,upvoteY,'r-',label='upvotes')
py.plot_date(X,volumeY,'b-',label='volume')
py.title("Dogecoin Jamacian Bobsleding Team Fundraiser")
py.show()

#### NASCAR!!!

prices = {}
posts = []
cmts  = []


with open("../DOGE_Nascar_data.csv") as inFile:
	rows = csv.reader(inFile)
	first = False
	for row in rows:
		if not first:
			first = True
			continue
		prices[ datetime.datetime.strptime(row[0],"%b %d, %Y").date().strftime("%m-%d-%y") ] = {'Open':row[1],'High':row[2],'Low':row[3],'Close':row[4],'Volume':row[5],'Market Cap':row[6] }

with open("dogecoin_nascar.json","w") as outFile:
	json.dumps(prices, outFile)

with open("nascar_top_new.json") as inFile:
	posts = json.loads(inFile.read())

#with open("bobsled_cmts_new.json") as inFile:
#	cmts = json.loads(inFile.read())

upvotes = defaultdict(int)
for p in prices:
	upvotes[p] = 0

for p in posts:
	da = datetime.date.fromtimestamp( int(posts[p]['created']) ).strftime('%m-%d-%y') 
	upvotes[da] += posts[p]['score']

wow = zip(upvotes.keys(), upvotes.values())
wow.sort()

Xtext = list(zip(*wow)[0])
upvoteY = list(zip(*wow)[1])
volumeY = [int(prices[r]['Volume']) for r in Xtext]
scale = max(volumeY)/max(upvoteY)
upvoteY = [scale*y for y in upvoteY]

X = [datetime.datetime.strptime(d, '%m-%d-%y').date() for d in Xtext]

py.plot_date(X,upvoteY,'r-',label='upvotes')
py.plot_date(X,volumeY,'b-',label='volume')
py.title("Dogecoin Nascar Fundraiser")
py.show()

'''

##GARLICOIN
prices = {}
posts = []
cmts  = []


with open("../GRLC_All_time_data.csv") as inFile:
	rows = csv.reader(inFile)
	first = False
	for row in rows:
		if not first:
			first = True
			continue
		prices[ datetime.datetime.strptime(row[0],"%b %d, %Y").date().strftime("%m-%d-%y") ] = {'Open':row[1],'High':row[2],'Low':row[3],'Close':row[4],'Volume':row[5],'Market Cap':row[6] }

with open("grlc_alltime.json","w") as outFile:
	json.dumps(prices, outFile)

with open("garlicoin_top_new.json") as inFile:
	posts = json.loads(inFile.read())

#with open("bobsled_cmts_new.json") as inFile:
#	cmts = json.loads(inFile.read())

upvotes = defaultdict(int)
for p in prices:
	upvotes[p] = 0

for p in posts:
	da = datetime.date.fromtimestamp( int(posts[p]['created']) ).strftime('%m-%d-%y') 
	if da not in prices:
		continue
	upvotes[da] += posts[p]['score']


##fix this
wow = zip(upvotes.keys(), upvotes.values())
wow.sort()

Xtext = list(zip(*wow)[0])
upvoteY = list(zip(*wow)[1])
volumeY = [int(prices[r]['Volume']) for r in Xtext]
diffocY = [float(prices[r]['Close'])-float(prices[r]['Open']) for r in Xtext ]
#scale = max(volumeY)/max(upvoteY)
#upvoteY = [scale*y for y in upvoteY]

X = [datetime.datetime.strptime(d, '%m-%d-%y').date() for d in Xtext]

'''
fig = py.figure()
host = fig.add_subplot(111)
volume = host.twinx()

#diffoc.set_ylim(-0.01,0.01)
host.set_xlabel("Date")
host.set_ylabel("Upvotes")

volume.set_ylabel("Volume")

p1, = host.plot_date(X,upvoteY,'r-',label="Upvotes")
p2, = volume.plot_date(X,volumeY,'b-',label="Volume")
#host.legend([p1, p2, p3], loc='best')

py.title("Garlicoin Upvotes & Volume")
py.show()
'''
fig = py.figure()
host = fig.add_subplot(111)
diffoc = host.twinx()
host.set_xlabel("Date")
host.set_ylabel("Upvotes")
diffoc.set_ylabel("Diff in Open and Close Price")
host.plot_date(X,upvoteY,'r-',label="Upvotes")
diffoc.plot_date(X,diffocY,'g-',label="Open - Close Price")
py.title("Garlicoin Upvotes & Diff in Close-Open")
py.show()
