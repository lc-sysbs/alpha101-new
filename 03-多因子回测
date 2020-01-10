from WindPy import *
from datetime import *
from WindAlgo import *
from scipy.stats import rankdata
import numpy as np
import pandas as pd
from datetime import datetime
import scipy.stats as stats
import numpy.linalg as la
from sympy.matrices import Matrix, GramSchmidt
w.start()

# 回测筛选股票池
def get_stocks(trDate,A_stocks):
    status = w.wss(A_stocks, "trade_status,maxupordown,riskwarning,ipo_date", tradeDate=trDate, usedf=True)[1]
    date_least=w.tdaysoffset(-6,trDate,'Period=M').Data[0][0]   
    trade_codes=list(status[(status['TRADE_STATUS']=='交易')&(status['IPO_DATE']<=date_least)&(status['MAXUPORDOWN']==0)&(status['RISKWARNING']=='否')].index)    
    return trade_codes  

def initialize(context):            
    context.capital = 10000000        # 回测的初始资金
    context.securities = w.wset("sectorconstituent", "date=20140101;windcode=000906.SH").Data[1]
    context.start_date = "20140101"  # 回测开始时间
    context.end_date = "20200101"    # 回测结束时间
    context.commission = 0.0003      # 手续费
    context.benchmark = '000906.SH'  # 设置回测基准

def handle_data(bar_datetime, context, bar_data):
    pass
    
def my_schedule1(bar_datetime, context, bar_data): 
    bar_datetime_str = bar_datetime.strftime('%Y-%m-%d')
    stock800 = w.wset("sectorconstituent", "date="+bar_datetime_str+";windcode=000906.SH").Data[1]
    stocks = get_stocks(bar_datetime_str,stock800)  # 获取筛选后的股票池
    start_time = w.tdaysoffset(-20, bar_datetime_str,usedf=True).Data[0][0].strftime('%Y-%m-%d')
    # 计算因子值
    close = w.wsd(stocks,'close',start_time,bar_datetime_str,usedf = True)[1].reset_index().drop(columns = ['index'])
    vwap = w.wsd(stocks,'vwap',start_time,bar_datetime_str,usedf = True)[1].reset_index().drop(columns = ['index'])
    Open = w.wsd(stocks,'open',start_time,bar_datetime_str,usedf = True)[1].reset_index().drop(columns = ['index'])
    volume = w.wsd(stocks,'volume',start_time,bar_datetime_str,usedf = True)[1].reset_index().drop(columns = ['index'])
    returns = w.wsd(stocks,'pct_chg',start_time,bar_datetime_str,usedf = True)[1].reset_index().drop(columns = ['index'])
    high = w.wsd(stocks,'high',start_time,bar_datetime_str,usedf = True)[1].reset_index().drop(columns = ['index'])
    low = w.wsd(stocks,'low',start_time,bar_datetime_str,usedf = True)[1].reset_index().drop(columns = ['index'])
    alpha = {'alpha6':alpha6(Open, volume).iloc[-1],'alpha14':alpha14(Open,volume,returns).iloc[-1],
             'alpha26':alpha26(volume,high).iloc[-1],'alpha54':alpha54(Open,close,high,low).iloc[-1]}
    data = pd.DataFrame(alpha,index=stocks)
    profit = w.wss(stocks, "fa_oigr_ttm,pe_ttm", tradeDate=bar_datetime_str, usedf=True)[1] 
    profit.index = data.index
    data_i = profit['FA_OIGR_TTM']/profit['PE_TTM']
    data.insert(0,'1/PEG',data_i)
    data = factor_sum(data.fillna(value = 0)) # 因子打分
    data = data.sort_values([data.columns.values[-1]],ascending=False) # 按打分大小排序
    code_list = list(data[:round(len(stocks)/10)].index)  # 选出top组股票
    wa.change_securities(code_list) # 改变证券池 
    context.securities = code_list    
    list_sell = list(wa.query_position().get_field('code')) # 获取当前仓位股票池
    for code in list_sell:
        if code not in code_list:
            volumn = wa.query_position()[code]['volume'] # 找到每个股票的持仓量 
            res = wa.order(code,volumn,'sell',price='close', volume_check=False) 

def my_schedule2(bar_datetime, context,bar_data):
    buy_code_list=list(set(context.securities)-(set(context.securities)-set(list(bar_data.get_field('code')))))
    list_now = list(wa.query_position().get_field('code')) # 获取当前仓位股票池
    for code in buy_code_list:
        if code not in list_now:
            res = wa.order_percent(code,1/len(buy_code_list),'buy',price='close', volume_check=False)  # 等权买入
            
 # 中位数去极值
def extreme_process_MAD(sample):  # 输入的sample为时间截面的股票因子df数据
    factor_name = list(sample.columns)
    for name in factor_name:
        x = sample[name]
        median = x.median()
        MAD = abs(x - median).median()
        x[x>(median+3*1.4826*MAD)] = median+3*1.4826*MAD
        x[x<(median-3*1.4826*MAD)] = median-3*1.4826*MAD
        sample[name] = x
    return sample       
    
# 标准化
def standardize(sample):
    factor_name = list(sample.columns)
    for name in factor_name:
        x = sample[name]
        sample[name] = (x - np.mean(x))/(np.std(x))
    return sample  
    
# 因子等权打分
def factor_sum(sample):
    factor_name = list(sample.columns)
    # 正交化
    matrix0 = []
    for i in range(len(factor_name)):
        matrix0.append(Matrix(sample[factor_name[i]]))
    matrix1 = np.array(GramSchmidt(matrix0))
    sample = pd.DataFrame(matrix1.T,index=sample.index,columns=factor_name)
    sample = standardize(sample)
    sample['alpha_sum'] = sample[factor_name[0]] * 0
    for i in range(len(factor_name)):
        sample['alpha_sum'] = sample['alpha_sum'] + sample[factor_name[i]]    
    return sample   
    
# 因子函数
def correlation(x, y, window):
    return x.rolling(window).corr(y)
def delta(df, period):
    return df.diff(period)
def rank(df):
    return df.rank(pct=True,axis=1)
def ts_max(df, window):
    return df.rolling(window).max()
def rolling_rank(na):
    return rankdata(na)[-1]
def ts_rank(df,window):
    return df.rolling(window).apply(rolling_rank)
def alpha6(Open, volume):
    alpha = -1 * correlation(Open, volume, 10)
    return alpha.replace([-np.inf, np.inf], 0).fillna(value = 0)
def alpha14(Open,volume,returns):
    x1 = correlation(Open, volume, 10).replace([-np.inf, np.inf], 0).fillna(value=0)
    x2 = -1 * rank(delta(returns, 3))
    alpha = x1 * x2
    return alpha.fillna(value = 0)
def alpha26(volume,high):
    x = correlation(ts_rank(volume, 5), ts_rank(high, 5), 5).replace([-np.inf, np.inf], 0).fillna(value=0)
    alpha = ts_max(x, 3)
    return alpha.fillna(value = 0)
def alpha54(Open,close,high,low):
    x = (low - high).replace(0, -0.0001)
    alpha = (low - close) * (Open ** 5) / (x * (close ** 5))
    return alpha

wa = BackTest(init_func = initialize, handle_data_func=handle_data)  # 实例化回测对象
wa.schedule(my_schedule1, "m", 0)   # w表示在每周执行一次策略，0表示偏移，表示月初第一个交易日往后0天
wa.schedule(my_schedule2, "m", 0) 
res = wa.run(show_progress=True)    # 调用run()函数开始回测,show_progress可用于指定是否显示回测净值曲线图
nav_df = wa.summary('nav')       # 获取回测结果，回测周期内每一天的组合净值
