from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd
from tqdm import tqdm
from pyecharts.charts import Line
from pyecharts import options as opts
import numpy as np

SEARCH_TOPIC_NUMS = 300

df = pd.read_excel('../outputs/分词文本.xlsx')
df = df.dropna(subset=['cut'])
n_features = 10000 #提取10000个特征词语
# 定义CountVectorizer模型
tf_vectorizer = CountVectorizer(strip_accents='unicode',
                                max_features=n_features,
                                max_df=0.5,
                                min_df=15)

# 将文本加入训练，并且构成词向量空间，然后再将这些词转换成向量
tf = tf_vectorizer.fit_transform(df['cut'].values.astype('U'))
frequences = tf_vectorizer.vocabulary_

plexs_list = []
for i in tqdm(range(1,SEARCH_TOPIC_NUMS+1)):
    lda = LatentDirichletAllocation(n_components=i, max_iter=50,
                                    learning_method='batch',
                                    learning_offset=50, random_state=0)
    lda.fit(tf)
    perplexity = lda.perplexity(tf)
    plexs_list.append(perplexity)


# 创建Line对象，设置基本属性
line_chart = Line()
line_chart.set_global_opts(title_opts=opts.TitleOpts(title='困惑度变化图'),
                           xaxis_opts=opts.AxisOpts(name="主题数"),
                           yaxis_opts=opts.AxisOpts(name="数值"))

# 添加数据
topic_nums = list(np.arange(1, SEARCH_TOPIC_NUMS+1))
line_chart.add_xaxis([str(num) for num in topic_nums])
line_chart.add_yaxis('相似度', plexs_list, label_opts=opts.LabelOpts(is_show=False, position=None))

# 渲染图表
line_chart.render('../outputs/perplexity_charts.html')

