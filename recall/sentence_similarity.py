from gensim import corpora, models, similarities
from collections import defaultdict


class SentenceSimilarity:
    """
    构建QQ之间的相似度矩阵
    """

    def __init__(self, seg, sentences, model_type='tf_idf'):
        self.seg = seg
        self.cut_sentences = [self.seg.cut_for_search(sentence) for sentence in sentences]
        self.filter_low_frequency_words()
        assert self.cut_sentences != [], '候选句子不足,无法构建相似度矩阵'
        self.dictionary = corpora.Dictionary(self.cut_sentences)
        self.corpus_simple = [self.dictionary.doc2bow(cut_sentence) for cut_sentence in self.cut_sentences]
        self.model, self.index = self.create_model(model_type)

    def filter_low_frequency_words(self, min_frequency=1):
        """
        过滤低频词
        """
        frequency = defaultdict(int)
        for text in self.cut_sentences:
            for token in text:
                frequency[token] += 1
        self.cut_sentences = [[token for token in cut_sentence if frequency[token] > min_frequency]
                              for cut_sentence in self.cut_sentences]

    # 对新输入的句子（比较的句子）进行预处理
    def sentence2vec(self, sentence):
        sentence = self.seg.cut_for_search(sentence)
        vec_bow = self.dictionary.doc2bow(sentence)
        return self.model[vec_bow]

    def create_model(self, model_type):
        if model_type == 'lsi':
            model = models.LsiModel(self.corpus_simple)
        elif model_type == 'lda':
            model = models.LdaModel(self.corpus_simple)
        else:
            model = models.TfidfModel(self.corpus_simple)
        corpus = model[self.corpus_simple]
        index = similarities.MatrixSimilarity(corpus, num_features=len(self.dictionary))
        return model, index

    def similarity_k(self, sentence, k):
        sentence_vec = self.sentence2vec(sentence)

        sims = self.index[sentence_vec]
        sim_k = sorted(enumerate(sims), key=lambda item: item[1], reverse=True)[:k]

        indices = [i[0] for i in sim_k]
        scores = [i[1] for i in sim_k]
        return indices, scores
