import numpy as np
import itertools

from konlpy.tag import Okt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch

class keybert():
    def __init__(self, doc, model_path) -> None:
        self.doc = doc
        pass






















































doc = """
본 발명은 디지털 트윈을 이용하여 양식장을 운영 및 관리하는 것을 목적으로 한다. 보다 구체적으로는 실제 양식장 환경 상태와 동일한 가상의 디지털 트윈 양식장을 생성하여 사용자가 시뮬레이션을 수행하는 것을 목적으로 한다. 즉, 실제 양식장과 동일한 가상의 디지털 트윈 양식장을 형성하여, 이를 통해 실제 양식장을 관리할 수 있는 시스템 및 방법을 제공한다. 또한, 인공지능 시스템을 탑재하여 사용자의 개입 없이도 가상의 양식장 시뮬레이션을 통해 최적의 개선조건을 도출하고, 이를 바탕으로 실제 양식장을 관리할 수 있으며, 시뮬레이션을 통해 도출된 최적의 개선조건을 사용자가 확인하고 제어할 수도 있다.
"""

# POS tagging
okt = Okt()

tokenized_doc = okt.pos(doc)
tokenized_nouns = ' '.join([word[0] for word in tokenized_doc if word[1] == 'Noun'])

print('품사 태깅 10개만 출력 :',tokenized_doc[:10])
print('명사 추출 :',tokenized_nouns)


# unigram, bigram, trigram extraction
n_gram_range = (1, 2)

count = CountVectorizer(ngram_range=n_gram_range).fit([tokenized_nouns])
candidates = count.get_feature_names_out()

print('trigram 개수 :',len(candidates))
print('trigram 다섯개만 출력 :',candidates[:5])


# SBERT load
model = SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens')

# get embedding
doc_embedding = model.encode([doc])
candidate_embeddings = model.encode(candidates)

# model save (all model)
torch.save(model, 'SBERT_model.pt')

# model load (all model)
model = torch.load('SBERT_model.pt') 

# extraction top 5 keyword
top_n = 5
distances = cosine_similarity(doc_embedding, candidate_embeddings)
keywords = [candidates[index] for index in distances.argsort()[0][-top_n:]]
print(keywords)


# Maximal Marginal Relevance
def mmr(doc_embedding, candidate_embeddings, words, top_n, diversity):

    # 문서와 각 키워드들 간의 유사도가 적혀있는 리스트
    word_doc_similarity = cosine_similarity(candidate_embeddings, doc_embedding)

    # 각 키워드들 간의 유사도
    word_similarity = cosine_similarity(candidate_embeddings)

    # 문서와 가장 높은 유사도를 가진 키워드의 인덱스를 추출.
    # 만약, 2번 문서가 가장 유사도가 높았다면
    # keywords_idx = [2]
    keywords_idx = [np.argmax(word_doc_similarity)]

    # 가장 높은 유사도를 가진 키워드의 인덱스를 제외한 문서의 인덱스들
    # 만약, 2번 문서가 가장 유사도가 높았다면
    # ==> candidates_idx = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10 ... 중략 ...]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    # 최고의 키워드는 이미 추출했으므로 top_n-1번만큼 아래를 반복.
    # ex) top_n = 5라면, 아래의 loop는 4번 반복됨.
    for _ in range(top_n - 1):
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # MMR을 계산
        mmr = (1-diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # keywords & candidates를 업데이트
        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [words[idx] for idx in keywords_idx]

# get embedding vector
def get_embedding(model, words):
    return model.encode(words)

# get similarity with doc embedding
def similarity_with_doc(doc_embedding, words_embeddings, words):

    word_doc_similarity = cosine_similarity(words_embeddings, doc_embedding)
    embedding_dict = {}
    for i in range(len(words)):
        embedding_dict[words[i]] = word_doc_similarity[i]

    return embedding_dict

# top n words extraciton with MMR
words = mmr(doc_embedding, candidate_embeddings, candidates, top_n=10, diversity=0.9)
print(words)
# get words embedding & similarity
words_embedding = get_embedding(model, words)
similarity_with_doc(doc_embedding, words_embedding, words)