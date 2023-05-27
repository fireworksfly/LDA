import pandas as pd
import jieba
import re


def clean_zh_text(text):
    # 这里可以根据自己需要进行添加
    comp = re.compile('【.*】')
    return comp.sub('', text)


def add_word(words_txt):
    with open(words_txt, 'r', encoding='utf-8') as f:
        words = f.readlines()
        for word in words:
            one = word.strip()
            jieba.add_word(one, freq=None, tag=None)
            jieba.suggest_freq(one, tune=True)


def my_cut(text, stop_words):
    """
    首先对文章进行分词， 然后对词进行停用
    :param text: 博文
    :param stop_words: 停用词表
    :return:
    """
    return [w for w in jieba.cut(text) if w not in stop_words and len(w) > 1]


if __name__ == '__main__':
    stop_words = []
    with open("./stop_words.txt", encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            stop_words.append(line.strip())
    add_word("add_words.txt")
    # *********
    df = pd.read_excel('./你的文件.xlsx')  # ！！！改成自己的文件名

    for index, row in df.iterrows():
        # 可以根据自己需要进行修改
        text = clean_zh_text(str(row['文本']))
        print(text)
        cut_words = my_cut(text, stop_words)
        result = ' '.join(cut_words)
        df.loc[index, '分词'] = result
    df.to_excel('./outputs/分词文本.xlsx', index=False)
