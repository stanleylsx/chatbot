import pandas as pd
from recall.sentence_similarity import SentenceSimilarity


class RecallModel:
    """
    召回池
    """
    def __init__(self, seg, faq_corpus_path, chat_corpus_path):
        self.seg = seg
        faq_question_list_kw, self.faq_question_list, self.faq_answer_list = self.read_corpus(faq_corpus_path)
        chat_question_list_kw, self.chat_question_list, self.chat_answer_list = self.read_corpus(chat_corpus_path)
        self.faq_invert_table = self.create_invert_table(faq_question_list_kw)
        self.chat_invert_table = self.create_invert_table(chat_question_list_kw)

    def read_corpus(self, path):
        """
        FAQ模式数据读取
        """
        question_list = []
        # 问题的关键词列表
        question_list_segment = []
        answer_list = []
        data = pd.read_csv(path, header=None, low_memory=False)
        data_ls = data.values.tolist()
        for t in data_ls:
            question_list.append(str(t[0]))
            question_list_segment.append(self.seg.cut(str(t[0])))
            answer_list.append(str(t[1]))
        return question_list_segment, question_list, answer_list

    @staticmethod
    def create_invert_table(question_list_kw):
        """
        简单的倒排表
        """
        invert_table = {}
        for index, tmpLst in enumerate(question_list_kw):
            for kw in tmpLst:
                if kw in invert_table.keys():
                    invert_table[kw].append(index)
                else:
                    invert_table[kw] = [index]
        return invert_table

    def filter_question_by_invert_table(self, question, task):
        """
        根据问题从倒排表中筛选关键词相交的问题
        """
        idx_list = []
        questions = []
        answers = []
        input_question_segment = self.seg.cut(question)
        if task == 'faq':
            invert_table = self.faq_invert_table
            question_list = self.faq_question_list
            answer_list = self.faq_answer_list
        else:
            invert_table = self.chat_invert_table
            question_list = self.chat_question_list
            answer_list = self.chat_answer_list

        for kw in input_question_segment:
            if kw in invert_table.keys():
                idx_list.extend(invert_table[kw])
        idx_set = set(idx_list)
        for index in idx_set:
            questions.append(question_list[index])
            answers.append(answer_list[index])
        return questions, answers

    def recall(self, question, top_k, task):
        # 利用关键词匹配得到与原来相似的问题集合
        filter_question_list, filter_answer_list = self.filter_question_by_invert_table(question, task)
        # 初始化模型
        try:
            sim = SentenceSimilarity(self.seg, filter_question_list, model_type='tf_idf')
        except AssertionError:
            question_results_k, filter_question_list, filter_answer_list = [], [], []
        else:
            question_results_k = sim.similarity_k(question, top_k)
        return question_results_k, filter_question_list, filter_answer_list

