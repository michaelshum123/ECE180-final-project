'''
ECE 180 Final Project
Reddit comment fetcher
By Michael Shum
'''
import urllib2
import json
import time

def getComment( dat ) :

	output = []
	data = []
	if type(dat) is not dict:
		return []

	if dat['kind'] == 'Listing':
		data = dat['data']['children']
		for d in data:
			newO = {}
			if d['kind'] != 't1':
				if d['kind'] == 'more':
					continue
				print("not t1! " + d['kind'])
				continue
			inData = d['data']
			if 'replies' in inData and type(inData['replies']) is not str:
				output +=  getComment(inData['replies'])
			if inData['score'] <= 1: #more than 1 upvote only
				continue
			newO['score'] = inData['score']
			newO['body']  = inData['body']
			#print(newO)
			output.append(newO)
	return output

#baselink = "https://www.reddit.com/r/garlicoin/comments/"
def readPostAndWrite(baselink,infile,outfile, last=""):
	inPosts = []
	with open(infile,"r") as inFile:
		posts = json.loads(inFile.read())
		for p in posts:
			inPosts.append(p)

	out = {}
	if last != "":
		with open(outfile,"r") as inFile:
			out = json.loads(inFile.read())

	for p in inPosts:
		if last != "" and p != last:
			continue

		postOut = {}
		qLink = baselink + p + '/.json'
		print(qLink)
		req = urllib2.Request(qLink)
		req.add_header('User-agent','ece180-python-bot-1337')
		res = urllib2.urlopen(req)
		web = json.loads(res.read())

		onePost = getComment(web[1])
		out[p] = onePost


		with open(outfile,"w") as outf:

			json.dump(out,outf)
			print("wrote "+p+" to file!")
		time.sleep(1)


#fetch comments from posts
#bobsledding

#

#bobsled
'''
readPostAndWrite( baselink="https://www.reddit.com/r/dogecoin/comments/",
	infile="bobsled_top_new.json",outfile="bobsled_cmts_new.json")
'''

#nascar
readPostAndWrite( baselink="https://www.reddit.com/r/dogecoin/comments/",
	infile="nascar_top_new.json",outfile="nascar_cmts_new.json",last="25dg4m")

#garlicoin

readPostAndWrite( baselink="https://www.reddit.com/r/garlicoin/comments/",
	infile="garlicoin_top_new.json",outfile="garlicoin_cmts_new.json")

# https://www.reddit.com/r/garlicoin/comments/812m0k/.json

