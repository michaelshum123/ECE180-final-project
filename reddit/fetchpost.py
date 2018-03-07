'''
Reddit data post fetcher

'''
import urllib
import json
import time

def readAndWrite(link,lower_time,upper_time,outfile,last=""):
	after = "&after=t3_"
	last_post = last
	out = {}
	for i in range(15):
		qLink = link
		count = 0
		if last_post != "":
			qLink += after + last_post
		
		print(qLink)
		res = urllib.urlopen(qLink)
		web = json.loads(res.read())
		if not('data' in web and 'children' in web['data']):
			continue

		data = web['data']['children']
		for d in data: #d = dict
			count += 1
			newId = d['data']['id']
			if count == len(data):
				print("last: " + newId)
				last_post = newId
			if not( lower_time < d['data']['created'] and d['data']['created'] < upper_time):
				continue
			if newId in out:
				print("FOUND DUPLICATE!!")
				continue
			newElem = { 'score':d['data']['score'],'num_comments': d['data']['num_comments'], 'created':d['data']['created'],'title':d['data']['title'] }
			out[newId] = newElem

		with open(outfile,"w") as outf:
			json.dump(out,outf)
			print("wrote "+str(i)+" to file!")
		time.sleep(2)


#fetch posts and comments within time periods

#bobsledding

#top
#readAndWrite( link="https://www.reddit.com/r/dogecoin/top/.json?sort=top&t=all&limit=100",
#	lower_time=1388534400, upper_time=1393632000,outfile="bobsled_top_new.json")


#nascar
#readAndWrite( link="https://www.reddit.com/r/dogecoin/top/.json?sort=top&t=all&limit=100",
#	lower_time=1396310400, upper_time=1401580800,outfile="nascar_top_new.json")

#garlicoin
readAndWrite( link="https://www.reddit.com/r/garlicoin/top/.json?sort=top&t=all&limit=100",
	lower_time=0, upper_time=1519862400,outfile="garlicoin_top_new.json")

