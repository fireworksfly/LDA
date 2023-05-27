import re
import itertools
import pandas as pd
import numpy as np
from tqdm import tqdm
from pyecharts.charts import Line
from pyecharts import options as opts

from gensim import corpora, models

SEARCH_TOPIC_NUMS = 300


def cos(vector1, vector2):
    dot_product = 0.0
    normA = 0.0
    normB = 0.0
    for a, b in zip(vector1, vector2):
        dot_product += a * b
        normA += a ** 2
        normB += b ** 2

    if normA == 0.0 or normB == 0.0:
        return None
    else:
        return dot_product / ((normA * normB) ** 0.5)


def lda_search(x_corpus, x_dict):
    # 初始化平均余弦相似度
    mean_similarity = []
    mean_similarity.append(1)

    # 循环生成主题并计算主题间相似度
    for i in tqdm(np.arange(2, SEARCH_TOPIC_NUMS + 1)):
        # LDA模型训练
        lda = models.LdaModel(x_corpus, num_topics=i, id2word=x_dict)
        term = lda.show_topics(num_words=100, num_topics=i)

        # 提取各主题词
        top_word = []
        for k in np.arange(i):
            top_word.append([''.join(re.findall('"(.*)"', i))
                             for i in term[k][1].split('+')])  # 列出所有词
        # print(top_word)

        # 构造词频向量
        word = sum(top_word, [])  # 列出所有的词
        unique_word = set(word)  # 去除重复的词

        # 构造主题词列表，行表示主题号，列表示各主题词
        mat = []
        for j in np.arange(i):
            top_w = top_word[j]
            mat.append(tuple([top_w.count(k) for k in unique_word]))

        p = list(itertools.permutations(list(np.arange(i)), 2))
        l = len(p)
        top_similarity = [0]
        for w in np.arange(l):
            vector1 = mat[p[w][0]]
            vector2 = mat[p[w][1]]
            top_similarity.append(cos(vector1, vector2))

        # 计算平均余弦相似度
        mean_similarity.append(sum(top_similarity) / l)
    return mean_similarity


if __name__ == '__main__':
    df = pd.read_excel('../outputs/分词文本.xlsx')
    df = df.dropna(subset=['分词'])
    input_list = []
    for index, row in df.iterrows():
        word = str(row['分词'])
        temp_list = word.split(' ')
        input_list.append(temp_list)

    dictionary = corpora.Dictionary(input_list)
    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=10000)
    corpus = [dictionary.doc2bow(text) for text in input_list]

    similarity_list = lda_search(corpus, dictionary)
    print('相似度列表：', similarity_list)

    # 绘制主题相似度图

    # 创建Line对象，设置基本属性
    line_chart = Line()
    line_chart.set_global_opts(title_opts=opts.TitleOpts(title='主题相似度变化图'),
                               xaxis_opts=opts.AxisOpts(name="主题数"),
                               yaxis_opts=opts.AxisOpts(name="数值"))

    # 添加数据
    topic_nums = list(np.arange(1, SEARCH_TOPIC_NUMS + 1))
    line_chart.add_xaxis([str(num) for num in topic_nums])
    line_chart.add_yaxis('相似度', similarity_list, label_opts=opts.LabelOpts(is_show=False, position=None))

    # 渲染图表
    line_chart.render('../outputs/CosSimilarity_charts.html')
