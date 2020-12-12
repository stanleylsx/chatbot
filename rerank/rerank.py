# -*- coding: utf-8 -*-
# @Time : 2020/12/12 23:15 
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : rerank.py 
# @Software: PyCharm
from utils.sim_hash import SimHash


class ReRank:
    def __init__(self, logger):
        self.simhash = SimHash()

    def rerank(self, recall_df, sentence):
        recall_df['score'] = recall_df.sentence.apply(lambda sen1: self.simhash.cal_similarity_score(sen1, sentence))
        recall_df = recall_df.sort_values(by='score', ascending=False)
        reply = recall_df.at[0, 'reply']
        return reply
