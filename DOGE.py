
# coding: utf-8

# In[2]:


import pandas as pd
import csv
import string
import datetime
try:
    import lxml
except:
    assert 0==1,'be sure to install lxml'

def get_doge_history(start,end,fname):
    '''
    This function requires lxml to run
    This function scrapes the historical data of garlicoin from the coinmarketcap website and creates a file named fname
    that is a csv that will have information such as date, open price, high price, low price, close price, volume, and 
    market cap. The user can set the start and end dates by entering in integer representations of the date for the
    start and end variables in the form yyyymmdd.
    param: start: start date in the form yyyymmdd
    type: int
    param: end: end date in the form yyyymmdd
    type: int
    param: fname: desired name of csv file
    Returns the dataframe that csv file was made off of
    '''
    
    ####This section makes sure the user inputs the proper arguments####
    assert isinstance(start,int),'please enter the date as an integer'
    assert isinstance(end,int),'please enter the date as an integer'
    assert len(str(start))==8,'please enter a valid date with the format yyyymmdd'
    assert len(str(end))==8,'please enter a valid date with the format yyyymmdd'
    assert isinstance(fname,basestring),'please enter a valid string to name the file'
    assert fname[-4:]=='.csv',"please make a csv file (ex. 'example.csv')"
    aYear = int(str(start)[:4])
    aMonth = int(str(start)[4:6])
    aDate = int(str(start)[6:])
    bYear = int(str(end)[:4])
    bMonth = int(str(end)[4:6])
    bDate = int(str(end)[6:])
    today = datetime.datetime.now()
    try:
        a = datetime.date(aYear,aMonth,aDate)
        b = datetime.date(bYear,bMonth,bDate)
    except:
        assert 0==1,'choose valid dates'
    a = datetime.date(aYear,aMonth,aDate)
    b = datetime.date(bYear,bMonth,bDate)
    assert a<b,'choose a date range where start is an earlier date than end'
    assert a>=datetime.date(2018,1,27),'chose too early of a start date'
    assert b<datetime.date(today.year,today.month,today.day),'chose too late of an end date'
    ####This section makes sure the user inputs the proper arguments####
    
    
    #initialize lists to contain data
    Date = []
    Open = []
    High = []
    Low = []
    Close = []
    Volume = []
    MarketCap = []
    #scrape table from coinmarketcap
    url = 'https://coinmarketcap.com/currencies/dogecoin/historical-data/?start={0}&end={1}'.format(str(start),str(end))
    tables = pd.read_html(url)
    df = tables[0]
    df.columns = [c.replace(' ','_') for c in df.columns]
    for i in range(len(df)):
        Date.append(df.Date[i])
        Open.append(df.Open[i])
        High.append(df.High[i])
        Low.append(df.Low[i])
        Close.append(df.Close[i])
        Volume.append(df.Volume[i])
        MarketCap.append(df.Market_Cap[i])
    rows = zip(Date,Open,High,Low,Close,Volume,MarketCap)
    #write to fname
    with open(fname,'wb') as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow(row)
    return df
get_doge_history(20180127,20180306,'test.csv')


# In[3]:





# In[ ]:




