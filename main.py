from utils.logger import get_logger
import time


if __name__ == "__main__":
    mode = 'predict'
    logger = get_logger('./logs')
    if mode == 'train_index':
        import pandas as pd
        from utils.sentence2vec import Sentence2VecUtils
        from utils.faiss_utils import FaissUtils
        faiss_utils = FaissUtils(logger)
        s2v = Sentence2VecUtils(logger)
        s2v.load_pca_u()
        df = pd.read_csv('data/qa_corpus/faq/liantongzhidao_answer.csv', encoding='utf-8', index_col=0)
        df['vec'] = df.sentence.apply(s2v.get_sif_vector)
        vectors = df.vec.to_list()
        faiss_utils.train_index(vectors)
    elif mode == 'predict':
        from recall.recall import Recall
        from rerank.rerank import ReRank
        sentence = '怎么办理宽带业务'
        recall = Recall(logger)
        rerank = ReRank(logger)
        while True:
            logger.info('please input a question (enter [exit] to exit.)')
            sentence = input()
            if sentence == 'exit':
                break
            start = time.time()
            recall_df = recall.recall(sentence)
            reply = rerank.rerank(recall_df, sentence)
            print(reply)
            print(time.time() - start)


