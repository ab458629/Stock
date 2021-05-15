from pandas_datareader import data as web
from mpl_finance import candlestick_ohlc
from matplotlib import pyplot as plt
from matplotlib import style
import csv
import matplotlib.dates as mdates
import fix_yahoo_finance as yf
import datetime as dt
import pandas as pd
import numpy as np
import datetime

style.use('dark_background')
yf.pdr_override()

fig = plt.figure(figsize=(18,12))

stock = "2330"

analysis_result = './Data/' + stock + '_analysis_result.csv'
stock_data = './Data/' + stock + '.csv'

start = dt.datetime(2020, 10, 1)
end = dt.datetime.now()
price = None

FAST = 5
MID = 10
SLOW = 30

def DeepMA(data):
    price = data["Close"]
    data["MA_FAST"] = np.round(price.rolling(window=FAST, center=False).mean(), 2)
    data["MA_MID"] = np.round(price.rolling(window=MID, center=False).mean(), 2)
    data["MA_SLOW"] = np.round(price.rolling(window=SLOW, center=False).mean(), 2)
    data["MA_FAST_MINUS_MA_SLOW"] = data["MA_FAST"] - data["MA_SLOW"]
    data["MA_FAST_MINUS_MA_SLOW_INTERVAL"] = data["MA_FAST_MINUS_MA_SLOW"] - data["MA_FAST_MINUS_MA_SLOW"].shift(1)

    data["MA_SIGNAL"] = np.sign(data["MA_FAST_MINUS_MA_SLOW_INTERVAL"])
    data["SIGNAL"] = np.sign(data["MA_SIGNAL"] - data["MA_SIGNAL"].shift(1))

    ax0 = plt.subplot2grid((30, 12), (0, 0), rowspan=10, colspan=12)
    plt.title("Stock Analysis: " + stock, size=16)
    ax1 = plt.subplot2grid((30, 12), (12, 0), rowspan=3, colspan=12)
    ax2 = plt.subplot2grid((30, 12), (17, 0), rowspan=3, colspan=12)
    ax3 = plt.subplot2grid((30, 12), (22, 0), rowspan=3, colspan=12)
    ax4 = plt.subplot2grid((30, 12), (26, 0), rowspan=3, colspan=12)


    ax0.plot(data["MA_FAST"], color='cornflowerblue', linewidth=1.2, alpha=0.7, label='MA' + str(FAST))
    ax0.plot(data["MA_MID"], color='lime', linewidth=1.2, alpha=0.7, label='MA' + str(MID))
    ax0.plot(data["MA_SLOW"], color='red', linewidth=1.2, alpha=0.7, label='MA' + str(SLOW))
    ax0.plot(data.loc[data["SIGNAL"] == 1, "Close"], '^',  markersize=7, color='yellow', label="Buy")
    ax0.plot(data.loc[data["SIGNAL"] == -1, "Close"], 'v', markersize=7, color='paleturquoise', label="Sell")

    data["SIGNAL"].fillna(0, inplace=True)
    ax1.plot(data["SIGNAL"], linewidth=1.2, color='lightcoral', alpha=0.7, label="DEEP_MA_SIGNAL")
    ax1.yaxis.set_label_position("right")
    ax1.set_ylabel('Deep MA Analysis', size=12)

    orignal_index = data.index

    data = data.reset_index()
    data['Date'] = data['Date'].apply(lambda d: mdates.date2num(d.to_pydatetime()))
    candlestick = [tuple(x) for x in data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].values]
    candlestick_ohlc(ax0, candlestick, width=0.5, colorup='r', colordown='green', alpha=0.7)
    ax0.grid(True)
    ax0.legend()
    data['Date'] = orignal_index

    def bias(ma):
        data['MA'] = data['Close'].rolling(ma).mean()
        data['BIAS'] = (data['Close'] - data['MA']) / data['MA'] * 100
        return data['BIAS']

    bias = bias(20)
    ax2.plot(bias, alpha=0.7, linewidth=1.2, color='yellow', label="BIAS")
    ax2.axhline(5, color='red', alpha=0.3)
    ax2.axhline(-5, color='paleturquoise', alpha=0.3)
    ax2.set_ylim(-10, 10)
    ax2.yaxis.set_label_position("right")
    ax2.set_ylabel('BIAS', size=12)

    def rsi(data, period=7):
        import pandas as pd
        price = data["Close"]
        closedif = (price - price.shift(1)).dropna()

        upprc = pd.Series(0, index=closedif.index)
        upprc[closedif > 0] = closedif[closedif > 0]

        downprc = pd.Series(0, index=closedif.index)
        downprc[closedif < 0] = - closedif[closedif < 0]

        rsi = []

        for i in range(period, len(upprc) + 1):
            up_mean = np.mean(upprc.values[(i - period):i], dtype=np.float32)
            up_down = np.mean(downprc.values[(i - period):i], dtype=np.float32)
            rsi.append(100 * up_mean / (up_mean + up_down))

        rsi = pd.Series(rsi, index=closedif.index[(period - 1):])

        return (rsi)

    rsi21 = rsi(data, 21)
    rsi14 = rsi(data, 14)
    rsi7 = rsi(data, 7)

    ax3.plot(rsi21, color='lime', linewidth=1.2, alpha=0.7, label='RSI21')
    ax3.plot(rsi14, color='yellow', linewidth=1.2, alpha=0.7, label='RSI14')
    ax3.plot(rsi7, color='cornflowerblue', linewidth=1.2, alpha=0.7, label='RSI7')

    ax3.axhline(70, color='red', alpha=0.3)
    ax3.axhline(50, color='white', alpha=0.3)
    ax3.axhline(30, color='paleturquoise', alpha=0.3)
    ax3.set_ylim(-10, 110)
    ax3.yaxis.set_label_position("right")
    ax3.set_ylabel('RSI', size=12)
    ax3.legend()

    volume = [x[5] for x in candlestick]
    volume = np.asarray(volume)

    pos = data['Open'] - data['Close'] < 0
    mid = data['Open'] - data['Close'] == 0
    neg = data['Open'] - data['Close'] > 0
    ax4.bar(orignal_index[pos], volume[pos], color='red', width=0.5, align='center')
    ax4.bar(orignal_index[mid], volume[mid], color='yellow', width=0.5, align='center')
    ax4.bar(orignal_index[neg], volume[neg], color='green', width=0.5, align='center')
    ax4.yaxis.set_label_position("right")
    ax4.set_ylabel('Volume', size=12)
    plt.show()

    Profit(data)

def Profit(data):
    trade = pd.concat([
        pd.DataFrame({"price": data.loc[data["SIGNAL"] == 1, "Close"],

                      "operation": "Buy"}),
        pd.DataFrame({"price": data.loc[data["SIGNAL"] == -1, "Close"],

                      "operation": "Sell"})
    ])

    trade.sort_index(inplace=True)
    trade.index = range(1, len(trade) + 1)

    trade['ops_diff'] = trade['price'].diff()

    trade['profit'] = trade[trade['operation'] == 'Sell']['ops_diff']

    print(trade)

    wintimes = sum(1 for x in trade['profit'] if x > 0)
    tradetimes = int(len(trade['profit'].dropna()))

    losttimes = sum(1 for x in trade['profit'] if x < 0)

    sum_win_price = sum((n) for n in trade['profit'] if n > 0)
    sum_win_price_times = (sum(1 for x in trade['profit'] if x > 0))

    sum_lose_price = sum((abs(n)) for n in trade['profit'] if n < 0)
    sum_lose_price_times = sum(1 for x in trade['profit'] if x < 0)

    Total_profit_and_loss = trade['profit'].sum()

    print("勝率", "%.2f%%" % (wintimes / tradetimes * 100))
    print("平均獲利金額", "%.2f" % (sum_win_price / sum_win_price_times))
    print("平均虧損金額", "%.2f" % (sum_lose_price / sum_lose_price_times))
    print("盈虧比", "%.2f" % ((sum_win_price / sum_win_price_times) / (sum_lose_price / sum_lose_price_times)))
    print("總損益", "%.2f" % (Total_profit_and_loss))
    print("交易次數", tradetimes)
    print("獲利金額", "%.2f" % (sum_win_price))
    print("獲利次數", wintimes)
    print("虧損金額", "%.2f" % (sum_lose_price))
    print("虧損次數", losttimes)
    print("單次最大虧損", min((trade['profit'].dropna())))
    print("單次最大獲利", max((trade['profit'].dropna())))

    data.to_csv(analysis_result, encoding='utf-8')

def main():

    with open(stock_data, 'w') as csvfile:
        writer = csv.writer(csvfile)
        df = web.get_data_yahoo([str(stock) + '.TW'], start, end)
        df.to_csv(csvfile)

    data = pd.read_csv(stock_data, parse_dates=True, index_col='Date')

    DeepMA(data)

if __name__ == "__main__":
    main()
