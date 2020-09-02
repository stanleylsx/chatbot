import time
from recall.jieba_segment import Segment
from recall.recall_model import RecallModel

if __name__ == '__main__':
    # 设置外部词
    stop_words_path = '../data/stopwords/stopwords.txt'
    segment = Segment(stop_words_path)
    segment.load_userdict('../data/userdict/userdict.txt')
    faq_corpus_path = '../data/qa_corpus/faq/liantongzhidao_faq.csv'
    chat_corpus_path = '../data/qa_corpus/chat/xiaohuangji_chat.csv'
    recall = RecallModel(segment, faq_corpus_path=faq_corpus_path, chat_corpus_path=chat_corpus_path)
    while True:
        question = input("请输入问题(q退出): ")
        if question == 'q':
            break
        time1 = time.time()
        question_k, question_list_s, answer_list_s = recall.recall(question, 5, task='chat')
        print(question_k)
        if not question_k:
            print('亲，我不明白您说的什么')
            continue
        print('亲，我们给您找到的答案是： {}'.format(answer_list_s[question_k[0][0]]))
        for idx, score in zip(*question_k):
            print("same questions： {},                score： {}".format(question_list_s[idx], score))
        time2 = time.time()
        cost = time2 - time1
        print('Time cost: {} s'.format(cost))
