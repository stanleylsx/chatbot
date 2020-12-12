# -*- coding: utf-8 -*-
# @Time : 2020/12/12 16:53 
# @Author : lishouxian
# @Email : gzlishouxian@gmail.com
# @File : sim_hash.py 
# @Software: PyCharm
import jieba
import jieba.analyse
import numpy as np


class SimHash:
    def __init__(self):
        self.bit_num = 64
        self.stop_words = 'data/stopwords/stopwords.txt'

    def get_binary_str(self, keyword):
        """
        获取关键词对应的二进制哈希值
        """
        if keyword == '':
            return 0
        else:
            x = ord(keyword[0]) << 7
            m = 1000003
            mask = 2 ** 128 - 1
            for c in keyword:
                x = ((x * m) ^ ord(c)) & mask
            x ^= len(keyword)
            if x == -1:
                x = -2
            x = bin(x).replace('0b', '').zfill(self.bit_num)[-self.bit_num:]
            return str(x)

    def finger_value(self, sentence):
        """
        使用sim hash方法，将文本转化为fingerprint
        """
        def map_func(n):
            return 0 if n <= 0 else 1

        jieba.analyse.set_stop_words(self.stop_words)
        key_words = jieba.analyse.extract_tags(sentence, topK=20, withWeight=True, allowPOS=())
        if not key_words:
            raise Exception('没有提取到关键词')
        ret = []
        for keyword, weight in key_words:
            binary_str = self.get_binary_str(keyword)
            temp = []
            for c in binary_str:
                if c == '1':
                    temp.append(weight)
                else:
                    temp.append(-weight)
            ret.append(temp)
        add_ret = np.sum(np.array(ret), axis=0)
        finger_ret = np.frompyfunc(map_func, 1, 1)(add_ret)
        return ''.join([str(n) for n in finger_ret])

    def cal_similarity_score(self, sen1, sen2):
        """
        计算相似度
        """
        def hamming_dis(v1, v2):
            """
            O(N)的办法计算二进制中为1的个数
            """
            t1 = '0b' + ''.join(v1)
            t2 = '0b' + ''.join(v2)
            n = int(t1, 2) ^ int(t2, 2)
            count = 0
            while n:
                n &= (n - 1)
                count += 1
            return count

        vec_1 = self.finger_value(sen1)
        vec_2 = self.finger_value(sen2)
        hamming_distance = hamming_dis(vec_1, vec_2)
        score = 1 - hamming_distance / self.bit_num
        return score
