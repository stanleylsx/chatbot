# -*- coding: utf-8 -*-
# @Time : 2020/12/12 20:20 
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : recall.py 
# @Software: PyCharm
from utils.faiss_utils import FaissUtils
from utils.sentence2vec import Sentence2VecUtils
import pandas as pd


class Recall:
    def __init__(self, logger):
        self.faiss_utils = FaissUtils(logger)
        self.s2v = Sentence2VecUtils(logger)
        self.s2v.load_pca_u()
        self.faiss_utils.load_index()
        self.df = pd.read_csv('data/qa_corpus/faq/liantongzhidao_answer.csv', encoding='utf-8', index_col=0)

    def recall(self, sentence):
        vec = self.s2v.get_sif_vector(sentence)
        ind, dis = self.faiss_utils.get_query_result([vec])
        recall_df = self.df.iloc[ind[0], :].reset_index(drop=True)
        return recall_df
