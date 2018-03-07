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

upvotes = defaultdict(int)
for p in prices:
	upvotes[p] = 0

for p in posts:
	da = datetime.date.fromtimestamp( int(posts[p]['created']) ).strftime('%m-%d-%y') 
	if da not in prices:
		continue
	upvotes[da] += posts[p]['score']

wow = zip(upvotes.keys(), upvotes.values())
wow.sort()

Xtext = list(zip(*wow)[0])
upvoteY = list(zip(*wow)[1])
volumeY = [int(prices[r]['Volume']) for r in Xtext]
highY   = [float(prices[r]['High']) for r in Xtext]
marketY = [int(prices[r]['Market Cap']) for r in Xtext]
diffocY = [float(prices[r]['Close'])-float(prices[r]['Open']) for r in Xtext ]
#scale = max(volumeY)/max(upvoteY)
#upvoteY = [scale*y for y in upvoteY]

X = [datetime.datetime.strptime(d, '%m-%d-%y').date() for d in Xtext]


fig, ((ax1, ax2), (ax3, ax4)) = py.subplots(2, 2)
fig.set
volume = ax1.twinx()
high   = ax2.twinx()
market = ax3.twinx()
diffoc = ax4.twinx()

ax1.set_xlabel("Date")
ax2.set_xlabel("Date")
ax3.set_xlabel("Date")
ax4.set_xlabel("Date")

ax1.set_ylabel("Upvotes")
ax2.set_ylabel("Upvotes")
ax3.set_ylabel("Upvotes")
ax4.set_ylabel("Upvotes")

volume.set_ylabel("Volume")
high.set_ylabel("High Price")
market.set_ylabel("Market Cap")
diffoc.set_ylabel("Close - Open Price")

ln1 = ax1.plot_date(X,upvoteY,'r-',label="Upvotes")
ln2 = volume.plot_date(X,volumeY,'b-',label="Volumez")
lns = ln1+ln2
labs = [l.get_label() for l in lns]
ax1.legend(lns,labs, loc='best')

ln1 = ax2.plot_date(X,upvoteY,'r-',label="Upvotes")
ln2 = high.plot_date(X,highY,'b-',label="High Pricez")
lns = ln1+ln2
labs = [l.get_label() for l in lns]
ax2.legend(lns,labs, loc='best')

ln1 =ax3.plot_date(X,upvoteY,'r-',label="Upvotes")
ln2 =market.plot_date(X,marketY,'b-',label="Market Capz")
lns = ln1+ln2
labs = [l.get_label() for l in lns]
ax3.legend(lns,labs, loc='best')

ln1 =ax4.plot_date(X,upvoteY,'r-',label="Upvotes")
ln2 =diffoc.plot_date(X,diffocY,'b-',label="Close - Open Price")
lns = ln1+ln2
labs = [l.get_label() for l in lns]
ax4.legend(lns,labs, loc='best')

ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax4.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
fig.tight_layout()
py.suptitle("Garlicoin")
py.show()