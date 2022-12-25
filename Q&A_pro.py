import jsonpath
import json
import pandas as pd
import re
import numpy as np
import jieba
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from queue import PriorityQueue
from collections import defaultdict

def read_corpus():
    file = open('train-v2.0.json','r')
    content = file.read()
    str_all = json.loads(content)
    qlist = jsonpath.jsonpath(str_all, '$..question')
    alist = jsonpath.jsonpath(str_all, '$..text')
    assert len(qlist) == len(alist)
    return alist,qlist

def understand_corpus(alist,qlist):
    alist_str = []
    qlist_str = []
    for line in alist[:10]:
        astr = line.split(" ")
        for line in astr:
            #用lower得到所有sentence的小写，lower是作用于str的
            alist_str.append(line.lower())
    #对其中的单词进行数量统计
    astr_number = pd.value_counts(alist_str)
    print("answers :", astr_number)

    for line in  qlist[:10]:
        qstr = line.split(" ")
        for line in qstr:
            qlist_str.append(line.lower())
    qstr_number = pd.value_counts(qlist_str)
    print("question :", qstr_number)
    print(type(astr_number.values))
    # 获取计数频次大于20的单词，小于的全部去掉
    astr_bigger = astr_number[astr_number>20]
    print(astr_bigger)
    for i in astr_bigger.index:
        print("zheli",i)
    return alist_str,qlist_str

def re_q_a(alist,qlist):
    # 去掉特殊符号
    rule = r'[#!~+?,.""]'
    res = re.compile(rule)
    out_a = []
    out_q = []
    for i in alist :
        out_str_a = re.sub(res, " ", i)
        out_a.append(out_str_a)
    for j in qlist:
        out_str_q = re.sub(res," ",j)
        out_q.append(out_str_q)
    return out_a,out_q

def filter_stopwords(qlist):

    stop_words = set(stopwords.words('english'))
    print(stop_words)
    out_str_a = ' '
    out_str_q = ' '
    filter_a = []
    filter_q_new = []
    #对question做处理
    for line in qlist[:1000]:
        word_tokens = word_tokenize(line)
        filter_sentence_q = [w for w in word_tokens if not w in stop_words]
        #filter_stop.append(str(filter_sentence_a))
        for line in filter_sentence_q:
            out_str_q += line
            out_str_q += ' '
            out_str_q.strip()
            out_str_q.rstrip()
        out_str_q += ","
        #将str类型通过split变成列表
        filter_q = out_str_q.split(',')
        filter_q_new = [i for i in filter_q if i != '']

    print(len(filter_q_new))
        #filter_stop.append(out_str_new)

    return filter_q_new

#得到qlist的tf-idf的fit_transform
def tfidf(qlist):
    tfidf2 = TfidfVectorizer()
    # 这里的fit_transform和transform的区别
    # sparsity_matrix_a = tfidf2.fit_transform(alist)
    sparsity_matrix_b = tfidf2.fit_transform(qlist)
    print(sparsity_matrix_b)
    # 用s.A将它转换成numpy.array的形式
    print(np.count_nonzero(sparsity_matrix_b.A))
    # 计算矩阵的稀疏度
    sparsity = 1 - np.count_nonzero(sparsity_matrix_b.A) / sparsity_matrix_b.A.size
    print(sparsity)

    return tfidf2

def top5results(input_q,qlist,alist,tfidf_q,sparsity_q):
    """
    给定用户输入的问题 input_q, 返回最有可能的TOP 5问题。这里面需要做到以下几点：
    1. 对于用户的输入 input_q 首先做一系列的预处理，然后再转换成tf-idf向量（利用上面的vectorizer)
    2. 计算跟每个库里的问题之间的相似度
    3. 找出相似度最高的top5问题的答案
    """

    rule = r'[#!~+?,]'
    res = re.compile(rule)


    stop_words = set(stopwords.words('english'))
    #对input做分词
    pre_input = word_tokenize(input_q)
    print(pre_input)
    #对分词做停用词过滤
    filter_sentence_input = [w for w in pre_input if not w in stop_words]
    print(filter_sentence_input)
    #将处理后的句子还原成一句话，才能用td-idf进行对单个句子重要性的判断
    input_str = ' '
    for line in filter_sentence_input:
        input_str += line
        input_str += " "

    input_str += ","
    input_str = re.sub(res, "", input_str)
    # 将str类型通过split变成列表
    filter_input = input_str.split(',')
    print(filter_input)

    #用上面的vectorize做transform
    matrix_input = tfidf_q.transform(filter_input)
    #print(matrix_input)
    sim = cosine_similarity(matrix_input,sparsity_q)
    print("这里是相似度的list", sim)
    print(type(sim[0]))


    res = []
    queue = PriorityQueue()
    sim_index = []
    sim_list = sim[0].tolist()
    print(type(sim_list))
    #这这里对queue排序一下,加负号是为了找到正数最大值
    for i in range(0, len(sim[0])):
        queue.put(-sim[0][i])
    for i in range(10):
        res.append(queue.get())
    for i in res :
        index = sim_list.index(-i)
        sim_index.append(index)
    print(res)
    print(sim_index)
    question_all = []
    answer_all = []
    for index in sim_index:
        question = qlist[index]
        question_all.append(question)
        answer = alist[index]
        answer_all.append(answer)
    print("相似的question是：", question_all)
    print("对应的answer是：", answer_all)

    return ""

def top5results_inverse(input_q,qlist,alist,tfidf_q):
    """
    先处理input_q
    """
    filter_sentence_input_new = []
    rule = r'[#!~+?,]'
    res = re.compile(rule)
    stop_words = set(stopwords.words('english'))
    # 对input做分词
    pre_input = word_tokenize(input_q)
    print(pre_input)
    # 对分词做停用词过滤
    filter_sentence_input = [w for w in pre_input if not w in stop_words]
    for line in filter_sentence_input:
        input_str = re.sub(res,'', line)
        print(input_str)
        #判断一下输出是空的元素,空则不输出
        if input_str:
            filter_sentence_input_new.append(input_str)
    print("这里是处理后的输入句子",filter_sentence_input_new)
    out_str = ' '
    out_list = []
    for str in filter_sentence_input_new:
        out_str += str
        out_str += " "
    out_str += ","
    out_list =out_str.split(",")
    print(out_list)

    char_2_idx = {}
    idx_2_char = {}
    inverse_idx = {}
    alist_new1 = []
    qlist_new1 = []
    """
    用倒排表的方法，得到在question原始列表中候选的项
    """
    #在这里要用input的单个词去做，因为用的是词向量
    for i, w in enumerate(qlist):
        char_2_idx[w] = i
        idx_2_char[i]  = w
        for char in filter_sentence_input_new:
            if char in w:
                inverse_idx[i] = w
    #生成新的canidate的question


    #用寻找出来的answer生成新的列表
    for i in inverse_idx.keys():
        qlist_new = qlist[i]
        alist_new = alist[i]
        qlist_new1.append(qlist_new)
        alist_new1.append(alist_new)
    print("这里是question对应的answer",alist_new1)
    print("这里是question", qlist_new1)


    tfidf2 = TfidfVectorizer()
    # print(char_2_idx)
    # print(idx_2_char)
    sparsity_candidate = tfidf_q.transform(qlist_new1)
    input_spasity = tfidf_q.transform(out_list)

    similary = cosine_similarity(input_spasity,sparsity_candidate)
    print(similary[0])
    res = []
    queue = PriorityQueue()
    sim_index = []
    sim_list = similary[0].tolist()
    print(type(sim_list))
    # 这这里对queue排序一下,加负号是为了找到正数最大值
    for i in range(0, len(similary[0])):
        queue.put(-similary[0][i])
    #找到其中最大的5个
    for i in range(10):
        res.append(queue.get())
    for i in res:
        index = sim_list.index(-i)
        sim_index.append(index)
    print(res)
    print(sim_index)
    question_all = []
    answer_all = []
    for index in sim_index:
        question = qlist_new1[index]
        question_all.append(question)
        answer = alist_new1[index]
        answer_all.append(answer)
    print("相似的question是：", question_all)
    print("对应的answer是：", answer_all)

    return ""

def main():
    list = read_corpus()
    #print("这个是原始问题的前100个输出：",list[0][:100])
    #print("这个是原始答案的前100个输出",list[1][:100])
    re_out = re_q_a(list[0],list[1])
    #对数据做一个可视化的处理，明白其中的特性等等
    #understand_corpus(list[0],list[1])
    #过滤停用词
    list_new = filter_stopwords(re_out[1])
    input_q = "What was the name of  Beyoncé's first solo album"
    print("输入的query：",input_q)
    tfidf_q = tfidf(list_new)
    top5results_inverse(input_q,list_new,re_out[0],tfidf_q)

main()