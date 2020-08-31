import jieba
import re


class Seg:
    """
    结巴分词封装api
    """
    def __init__(self):
        self.stopwords = set()
        self.stopword_filepath = '../data/stopwords/stopwords.txt'
        self.read_in_stopword()

    @staticmethod
    def load_userdict(file_name):
        """
        用户字典
        """
        jieba.load_userdict(file_name)

    def read_in_stopword(self):
        """
        读取停用词
        """
        with open(self.stopword_filepath, 'r', encoding='utf-8') as file:
            word_lines = file.readlines()
            self.stopwords = [re.sub(r'[\r\n]', '', word) for word in word_lines]

    def cut(self, sentence, stopword=True, cut_all=False):
        seg_list = jieba.cut(sentence, cut_all)
        results = []
        for seg in seg_list:
            if stopword and seg in self.stopwords:
                continue
            results.append(seg)

        return results

    def cut_for_search(self, sentence, stopword=True):
        """
        适合用于搜索引擎构建倒排索引的分词，粒度比较细
        """
        seg_list = jieba.cut_for_search(sentence)

        results = []
        for seg in seg_list:
            if stopword and seg in self.stopwords:
                continue
            results.append(seg)

        return results
