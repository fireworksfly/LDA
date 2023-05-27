# lda: 中文文本主题分布
##  简单说明
本项目是基于jieba分词对中文文本进行主题建模，运行该程序之前请准备好相xlsx格式的文本文件（文本所在的列请把列名改为“文本”‘）并放入程序主目录中。

## 项目结构
```
|- LDA
	|- best_topic_nums
		|- cos_similarity.py
		|- perplexity.py
	|- outputs
	|- add_words.txt
	|- cut_words.py
	|- lda.ipynb
	|- stop_words.txt
	|- requirements.txt

```
其中outputs文件夹存放程序输出的文件
## 环境
`gensim==4.3.1
jieba==0.42.1
numpy==1.24.2
pandas==1.5.3
pyecharts==2.0.3
scikit_learn==1.2.2
tqdm==4.65.0`

## 运行
- First, 进行分词（请把文件放到程序主目录）并且将 `cut_words.py`文件中的读取文件名改成自己的，如果需要额外添加停用词，请在add_words按照文件格式进行添加自己的停用词（里面原来的词可以选择删除）, 运行`python cut_words.py` 得到"分词文本.xlsx"
- Then, 确定最佳主题数（如果自己已确定，此步可略过），本项目有两种确定的方式。分别是困惑度和主题间余弦相似度，可根据需要选择其一。 运行`python ./best_topic_nums/cos_similarity.py`或者`python ./best_topic_nums/perplexity.py`得到一个html文件，可以通过浏览器打开，通过图表确定最佳主题数。
- Finally, 运行` lda.ipynb`文件得到“主题.txt”和“lda可视化.html”文件

