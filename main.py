import pandas as pd
from utils.logger import get_logger
from utils.sentence2vec import Sentence2VecUtils
from utils.faiss_utils import FaissUtils

if __name__ == "__main__":
    logger = get_logger('./logs')
    faiss_utils = FaissUtils(logger)
    s2v = Sentence2VecUtils(logger)
    s2v.load_pca_u()

    df = pd.read_csv('data/qa_corpus/faq/liantongzhidao_vec.csv', encoding='utf-8', index_col=0)
    # vectors = df.vec.apply(lambda line: eval(line)).to_list()
    # faiss_utils.train_index(vectors)
    vec = s2v.get_sif_vector('怎么升级套餐')
    faiss_utils.load_index()
    ind, dis = faiss_utils.get_query_result([vec])
    print(ind, dis)
    print(df.iloc[ind[0], :])
