import pandas as pd
import yfinance as yf
import yahoofinancials
import matplotlib.pyplot as plt
import bs4 as bs
import pickle
import requests
from sklearn.metrics import precision_recall_fscore_support as score
from keras.callbacks import Earlystopping,ModelCheckpoint
from sklearn.decomposition import PCA
from sklearn.utils import class_weight


def pickle_save(to_save,save_path):
    with open(save_path, "wb") as fp:
        pickle.dump(to_save, fp)


def pickle_load(load_path):

    with open(load_path, "rb") as fp:
        b = pickle.load(fp)
    return b


"""
simpel struktur: 0/1 binary prediction structure (win or lose).
one model for each stock

data:

stocks in SP 500:

- current price
- max price (within time period?)
- momentum (hours days moving average)
- volume
- sector
- medieomtale (scraping popular youtube videos, stock forums?)

"""

"""



#tsla_df['Close'].plot(title='TESLA stock price')

ticker = yf.Ticker('TSLA')

"""

#tsla_ticker_df = ticker.history(period='max')
#print(tsla_ticker_df)



def save_sp500_tickers():
    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text
        tickers.append(ticker)

    with open("sp500tickers.pickle", "wb") as f:
        pickle.dump(tickers, f)

    tickers = [t[:-1] for t in tickers]

    return tickers

#tickers = save_sp500_tickers()

def remove_delisted_stocks(tickers):
    delisted = []
    for t in tickers:
        stock_df = yf.download(t,
                              start='2020-01-01',
                              end='2020-02-08',
                              progress=False,
                              interval='1d')
        if stock_df.shape != (27,6):
            delisted.append(t)
            print(stock_df.shape)
    print(delisted)
    tickers = [t for t in tickers if t not in delisted]
    return tickers
#tickers = remove_delisted_stocks(tickers)





#pickle_save(tickers,"C:/Users/Mikkel/Desktop/machine learning/stock_prediction/sp_500_companies.txt")
stock_names = pickle_load("C:/Users/Mikkel/Desktop/machine learning/stock_prediction/sp_500_companies.txt")
print(stock_names)
#print(len(b))




def download_stock_data(stock_names):
    complete_table = pd.DataFrame(columns=["result", "date", "open", "high_t1", "low_t1", "AdjClose_t1", "volume_t1", "stock"])
    for name in stock_names:
        print(name)
        stock_df = yf.download(name,start='2019-10-01',end='2020-02-18',progress=False,interval='1d')

        t1_info = stock_df.iloc[:-1, [1,2,4,5]].reset_index(drop=True)
        open = stock_df.iloc[1:, [0]].reset_index(drop=True)
        date = stock_df.index[1:].to_frame().reset_index(drop=True)
        stock = pd.DataFrame([name] * open.shape[0])
        result = pd.DataFrame([1 if x > 0 else 0 for x in (stock_df['Adj Close'] - stock_df['Open'])][1:])
        table = pd.concat([result,date,open,t1_info,stock],axis=1,ignore_index=True)
        table.columns = ["result","date","open","high_t1","low_t1","AdjClose_t1","volume_t1","stock"]
        complete_table = pd.concat([complete_table, table], axis=0, ignore_index=True)
    return complete_table
#complete_table = download_stock_data(stock_names)
#print(complete_table.shape)

#complete_table.to_pickle("C:/Users/Mikkel/Desktop/machine learning/stock_prediction/sp_500_data_2019okt2020feb18.txt")

stock_table = pickle_load("C:/Users/Mikkel/Desktop/machine learning/stock_prediction/sp_500_data_2019okt2020feb18.txt")

stock_table = pd.DataFrame(stock_table)







