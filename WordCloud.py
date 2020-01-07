import os
import re
import operator
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter

root_path = '..\\..\\'  #根目录为向上两级
py_code_list = []
for filepath,dirnames,filenames in os.walk(root_path):  # 用os.walk递归找到所有文件
	for i in filenames:
		if len(i)>3 and i[-3:]=='.py':   #判断是python文件
			fp = open(os.path.join(filepath,i),encoding='utf-8')  #以utf-8格式打开此文件
			py_code_list.append(fp.read())  #加入存代码的列表中
py_code_str = ' '.join(py_code_list)  # 拼接成一个字符串
words = re.findall('[a-zA-Z]+',py_code_str)  # 用正则表达式过滤出英文单词
word_count = Counter(words)  #统计词频
wc = WordCloud(
        background_color='black',
        max_font_size=40, 
        scale=3,
        stopwords=[]
).fit_words(word_count) #生成词云，不过滤停用词
wc.to_file('wc.jpg')   #输出词云图片
word_count_sorted = sorted(word_count.items(),key=operator.itemgetter(1),reverse=True) #根据词频排序，从大到小
top_10_words = [i[0] for i in word_count_sorted[:10]] #柱状图横坐标
top_10_words_counts = [i[1] for i in word_count_sorted[:10]] #柱状图纵坐标
plt.figure()
plt.bar(top_10_words,top_10_words_counts)   # 画柱状图
plt.xlabel('word')  
plt.ylabel('count')
plt.title('top_10_words')
plt.savefig("bar.jpg")  #保存图片
