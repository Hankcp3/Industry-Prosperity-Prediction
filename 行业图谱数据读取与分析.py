#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


# In[2]:


import os
import logging
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)


# # 基础功能

# In[3]:


# 全局变量
# 指定数据文件位置
file_dir = '行业数据库'


# ## 读取宏观指标

# ### 类定义

# In[4]:


class DB:
    """
        行业数据类
        
        每个行业数据库需要有两个文件：
            指标说明文件：指标_XXX.xlsx
            全部数据文件：数据_XXX.xlsx
        
        其中“XXX”为行业名称
            
        “指标说明文件”至少包括“指标名称”列，且要求“非空唯一”，其他列用作每个指标的元信息
        “全部数据文件”包括三列：“指标”、“日期”、“数据”，确保所有列都非空
    """    
    def __init__(self, industry):
        """
            构造函数
            
            参数：行业名称
        """
        self.industry=industry
        
        # 根据行业名称，得到两个文件的名称，其中文件路径由全局变量file_dir指定
        self.__field_file = os.path.join(file_dir, '指标_{0}.xlsx'.format(industry))
        self.__data_file = os.path.join(file_dir, '数据_{0}.xlsx'.format(industry))
        
        self.__ready = False # 数据加载状态，False：尚未加载，True：已加载
        
        self.__field = None   # 指标名称列表
        self.__meta = None    # 指标元信息
        self.__data = None    # 全部数据
        
    
    def load(self, reload=False):
        """
            加载数据，如果已经加载过数据，默认不会重新加载
            
            如果需要重新加载，传递参数reload=True
        """
        
        if reload or not self.__ready:
            self.__load_field()
            self.__load_data()
            self.__ready = True
            print('数据加载完毕！')
        else:
            print('数据已经加载！\n如需重新加载，请提供参数reload=True')
            
        
    def __load_field(self):
        """
            读取指标信息
        """
        field = pd.read_excel(self.__field_file).dropna(subset=['指标名称']) # 删除指标名称为空的行，暂不报错
        
        # 检查数据
        self.__check_field(field)
        
        field = field.set_index(['指标名称'])
        self.__field = field.index.tolist()          # 指标名称列表
        self.__meta = field.to_dict(orient='index')  # 指标元信息

        
    def __check_field(self, field):
        """
            对指标信息进行检查
            
            目前仅要求指标名称不能重复，后面有其他要求都可以写在这里
            
            注意这里不要对field进行任何修改
        """
        assert not any(field['指标名称'].duplicated()), "指标信息文件中存在重复的指标名称！"
    
    
    def __load_data(self):
        """
            读取全部数据
        """
        data = pd.read_excel(self.__data_file)
        self.__check_data(data)
        
        try:
            data['日期'] = pd.to_datetime(data['日期'])
        except:
            print('数据文件中的“日期”无法转换为DatetimeIndex对象！')
        
        self.__data = data.drop_duplicates()
        
        
    def __check_data(self, data):
        """
            对数据进行检查
            
            暂时仅要求self.__field中的指标在data中都存在
            
            注意这里不要对data进行任何修改
        """
        # 如果temp不为空集，意味着指标信息文件中有一些指标在数据文件中没有对应数据
        # 这里暂不报错，仅提供警告
        temp = set(self.__field) - set(data['指标'])
        if len(temp)>0:
            logging.warning('以下字段没有对应数据：\n\n·{0}\n\n使用get方法提取以上指标会返回空序列。'.format('\n·'.join(temp)))
    
    
    def get(self, name):
        """
            用于提取数据的方法
            
            目前仅支持一次提取一个指标，后面可根据需要增加一次提取多个指标的功能
        """
        assert self.__ready, "尚未加载数据！请先调用load()方法！"
        assert name in self.__field, "'{0}'不在数据指标列表中！".format(name)
        temp = self.__data[self.__data['指标']==name].set_index(['日期'])['数据'].sort_index()
        temp.name = name
        return temp
    
    @property
    def field(self):
        """
            查看数据中的全部指标
        """
        assert self.__ready, "尚未加载数据！请先调用load()方法！"
        return self.__field
    
    @property
    def meta(self):
        """
            查看指标元信息
            
            指标的元信息有以下用途：
                做数据处理时，根据指标元信息，采用特定的处理方法，例如日频和月频数据可能需要分别处理等
                做图形展示时，可能需要根据指标的“节点”信息来绘制图形
                ......
            总之，元信息尽量完整、详细，这样后面处理和分析时用处才大
                
        """
        assert self.__ready, "尚未加载数据！请先调用load()方法！"
        return self.__meta


# ### 使用示例

# In[9]:


# 创建对象
x = DB('涤纶长丝')


# In[10]:


# 加载数据之前无法提取数据
x.get('涤纶长丝产量 (万吨)')


# In[11]:


# 加载数据
x.load()


# In[8]:


# 不需要重复加载，除非强制重新加载
x.load()


# In[12]:


# 提取数据
x.get('涤纶长丝产量 (万吨)')


# In[13]:


# 查看指标元信息
x.meta['涤纶长丝产量 (万吨)']


# In[14]:


# 提取数据文件中没有的指标，不会报错，但返回的是空序列
x.get('中国盛泽化纤价格指数:化纤纤维:POY涤纶长丝')


# In[12]:


# 提取不在列表中的指标会报错
x.get('xxxx')


# In[15]:


# 查看全部指标列表
x.field


# ## 读取自定义指数

# ### 函数定义

# In[16]:


def get_close(industry):
    """
        从指数数据文件中提取收盘价数据
        
        数据命名规范：
            指数_XXX.xlsx
        文件中第一个Sheet为收盘数据所在的表格，其中至少包括“日期”和“收盘价(元)”两列
    """
    close = pd.read_excel(os.path.join(file_dir, '指数_{0}.xlsx'.format(industry)))[['日期', '收盘价(元)']]
    close = close.set_index(['日期'])
    close.index = pd.to_datetime(close.index)
    start = close.index.min()
    end = close.index.max()
    # 非交易日收盘价用过去距离最近的收盘价填充
    close = close.reindex(pd.date_range(start, end)).fillna(method='ffill')
    return close['收盘价(元)']


# ### 使用示例

# In[17]:


close = get_close('涤纶长丝')


# In[18]:


close


# ## 数据处理函数

# In[19]:


# 根据收盘价数据和日期索引，获取收益率
def get_ret(close, index, method):
    # 传入的index，应该是close的index的一个子集，否则有一些日期提取到的是空收盘价
    # assert set(index)<=set(close.index), '存在没有行情的日期索引。'  # 目前给的自定义指数数据有问题，暂时先注释掉这一行
    temp = close.reindex(index)
    if method==1:
        ret = temp.shift(-1)/temp - 1 # 下一期收益率
    elif method==-1:
        ret = temp/temp.shift(1) - 1 # 当期收益率
    else:
        raise Exception('收益率计算的method参数错误')
    return ret


# In[18]:


# 数据重新采样
# 仅实现从高频到低频的转换，例如：日—>周、月，周—>月
def resample(zhibiao, method, **kwargs):
    """
        指标数据重新采样
        一个指标应该采用哪种采样方法，应当在“指标信息文件”中用添加“采样方法”列，专门指定具体方法
        例如一些日度、周度指标变成月度
        或者日度指标变成周度
        
        目前仅提供以下几种选线，其他情况可以根据需要自己添加
        
        注意：该函数返回的数据可能存在缺失值
    """
    if method==0:
        # 不处理
        return zhibiao
    elif method==1:
        # 原始数据：日频或周频
        # 目标：得到月频数据
        # 方法：先变成日频，并用向前填充的方法补充缺失值，最后再转为月频，相当于取月末最新可得数据
        # 潜在问题：如果某个月份不存在样本点，由于采用日度向前填充，则最终会采用过去某个月份的值
        zhibiao = zhibiao.asfreq('D').fillna(method='ffill').asfreq('M')
    elif method==2:
        # 原始数据：日频或周频
        # 目标：得到月频数据
        # 方法：取当月均值
        # 潜在问题：如果某个月份不存在样本点，则该月份数据缺失
        zhibiao = zhibiao.groupby(by=pd.to_datetime(zhibiao.index)+pd.offsets.MonthEnd(0)).mean().asfreq('M')
    elif method==3:
        # 原始数据：日频
        # 目标：得到周频
        # 方法：取当周平均，记到星期日所在日期
        # 潜在问题：如果某个周度不存在样本点，则该月份数据缺失
        zhibiao = zhibiao.groupby(by=pd.to_datetime(zhibiao.index).map(lambda d:d + pd.offsets.DateOffset(days=6-d.dayofweek))).mean().asfreq('W')
    else:
        raise Exception('采样method参数错误')
    return zhibiao


# In[19]:


def fill_na(zhibiao, method, **kwargs):
    """
        处理缺失值
        目前仅提供一种用最新数据替代的方法（向前填充）
    """
    if method==0:
        return zhibiao
    elif method==1:
        zhibiao = zhibiao.fillna(method='ffill')
    else:
        raise Exception('处理缺失值method参数错误')
    return zhibiao


# In[20]:


def smooth(zhibiao, method, **kwargs):
    """
        滤波
    """
    if method==0:
        return zhibiao
    elif method==1:
        # 季节分解
        zhibiao = sm.tsa.seasonal_decompose(zhibiao, two_sided=False).trend
    elif method==2:
        # 移动平均
        ma = kwargs.get('ma', None)
        if ma is None: raise Exception('需要提供参数：ma')
        zhibiao = zhibiao.rolling(window=ma).mean()
    elif method==3:
        # HP滤波
        hp = kwargs.get('hp', None)
        if hp is None: raise Exception('需要提供参数：hp')
        zhibiao = sm.tsa.filters.hpfilter(zhibiao, lamb=hp)[1]
    else:
        raise Exception('滤波method参数错误')
    return zhibiao


# ## 测算函数

# In[21]:


# 实现R语言中的ccf函数
def ccf(x, y, lag_max = 20, ci=0.95):
    """
        计算两个时间序列的Cross Correlation Function
        返回：Cross Correlation以及置信区间
    """
    from scipy.stats import norm
    assert isinstance(lag_max, int) and lag_max>=0, '\'lag_max\' must be a non-negative integer!'
    assert len(x)==len(y), 'The two time series must be the same length!'
    assert ci>0 and ci<=1, '\'ci\' must be in the interval (0,1]'
    # import scipy.signal as ss
    # result = ss.correlate(x - np.mean(x), y - np.mean(y), method='direct') / (np.std(x) * np.std(y) * len(y))
    result = np.correlate(x - np.mean(x), y - np.mean(y), mode='full') / (np.std(x) * np.std(y) * len(y))
    length = (len(result) - 1) // 2
    lag_max = np.min([length, lag_max])
    lo = length - lag_max
    hi = length + (lag_max + 1)
    upperCI = norm.ppf((1+ci)/2)/np.sqrt(len(y))
    lowerCI = -upperCI
    
    lags = list(range(-lag_max, lag_max+1))
    result = pd.Series(result[lo:hi], index=lags, name='CCF')
    result.index.name='xLags'
    return result, lowerCI, upperCI


# In[22]:


def t_test(zhibiao, ret, state):
    """
        不同指标状态下，收益率的T检验
        
        zhibiao：经过数据清洗后的指标序列
        ret：与zhibiao对应的收益率序列
    """
    temp = zhibiao - zhibiao.shift(1) # 指标环比
    temp = np.sign(temp.dropna())
    
    # 根据指标环比状态，将样本分成两类
    index1 = temp.index[temp==state]
    index2 = temp.index[temp!=state]
    sample1 = ret.reindex(index1).dropna()
    sample2 = ret.reindex(index2).dropna()
    
    # 计算两个样本的信息：样本均值，样本容量，样本均值差，样本均值的T检验p值（单边）
    mu1 = sample1.mean() 
    N1 = len(sample1)
    mu2 = sample2.mean()
    N2 = len(sample2)
    diff = mu1 - mu2
    p = sm.stats.ttest_ind(sample1, sample2, alternative='larger', usevar='unequal')[1]
    return pd.Series({
        'mu1' : mu1,
        'N1' : N1,
        'mu2' : mu2,
        'N2' : N2,
        'diff' : mu1-mu2,
        'p' : p
    })
    


# In[23]:


def ccf_test(zhibiao, ret, lag_max, ci):
    """
        计算zhibiao和ret之间的样本Cross Correlation，并进行检验
    """
    index1 = zhibiao.dropna().index
    index2 = ret.dropna().index
    index = index1.intersection(index2)
    x = ret.reindex(index)
    y = zhibiao.reindex(index)
    r, l, u = ccf(x, y, lag_max=lag_max, ci=ci)

    sigOrder = ','.join(r.index[(r.abs()>u) & (r.index>0)].map(str)) # 显著的领先阶数
    bestOrder = r[r.index>=0].abs().idxmax() # 最优的领先阶数
    bestCorr = r[bestOrder] # 最优领先阶数的相关系数
    isBestCorrSig = 1 if np.abs(bestCorr) > u else 0

    r['显著阶数'] = sigOrder
    r['最优领先阶数'] = bestOrder
    r['最优相关系数'] = bestCorr
    r['最优系数显著'] = isBestCorrSig
    r['最优相关系数绝对值'] = np.abs(bestCorr)
    r['lowerCI'] = l
    r['upperCI'] = u
    return r


# # 应用

# In[24]:


industry = '涤纶长丝'


# ### 创建用于提取指标数据的对象

# In[25]:


# 创建对象
db = DB(industry)


# In[26]:


db.load()


# In[27]:


db.field


# ### 提取收盘价数据

# In[28]:


close = get_close(industry)


# In[29]:


close


# ### T检验与CCF检验

# In[30]:


# 这两个指标数据有问题：
# 市场价(低端价):涤纶长丝(POY 150D/48F):国内市场------------------数据有重复日期，但是数据值不一样
# 市场价:涤纶长丝POY:国内主要轻纺原料市场------------------------数据截止到2008年，收盘价从2009年才有，没办法检验
drop_list = ['市场价(低端价):涤纶长丝(POY 150D/48F):国内市场','市场价:涤纶长丝POY:国内主要轻纺原料市场']


# In[31]:


ttest_result = {}
ccf_result = {}

for i in db.field:
    if i in drop_list: continue
    print(i)
    zhibiao = db.get(i)
    zhibiao = resample(zhibiao, method=1)
    zhibiao = fill_na(zhibiao, method=1)
    zhibiao = smooth(zhibiao, method=2, ma=6)
    if len(zhibiao) ==0:
        continue
    # T检验：收益率采用下一期收益率
    ret = get_ret(close, zhibiao.index, method=1)
    ttest_result[i] = t_test(zhibiao, ret, state=1)
    
    # CCF检验：收益率采用同期收益率
    ret = get_ret(close, zhibiao.index, method=-1)
    ccf_result[i] = ccf_test(zhibiao, ret, lag_max=12, ci=0.95)
    
    
ttest_result = pd.DataFrame(ttest_result).T
ccf_result = pd.DataFrame(ccf_result).T


# In[33]:


ttest_result.to_excel('T检验_{0}.xlsx'.format(industry))
ccf_result.to_excel('CCF_{0}.xlsx'.format(industry))


# In[ ]:





# In[ ]:




