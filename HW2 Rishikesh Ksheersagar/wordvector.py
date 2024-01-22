# For Programming Problem 1 and 2

import numpy as np
np.random.seed(595)


def get_vocabulary(corpus):
    vocabulary = []
    v_size = -1
    for d in corpus:
        for w in d:
            if w not in vocabulary:
                vocabulary.append(w)
    v_size = len(vocabulary)

    return sorted(vocabulary), v_size


def get_co_occurrence_matrix(corpus, window_size=5):
    vocabulary, v_size = get_vocabulary(corpus)
    M = np.zeros((v_size, v_size), np.int32)
    word2ind = {}
    for _ in vocabulary:
        word2ind[_] = vocabulary.index(_)

    def get_n_ind(start, window, l):
        min_ind = max(0, start - window)
        max_ind = min(len(l)-1, start + window)
        return(range(min_ind, max_ind+1))

    for sent in corpus:
        for w1_ind in range(len(sent)):
            r = get_n_ind(w1_ind, window_size, sent)
            for w2_ind in r:
                if w1_ind!=w2_ind:
                    M[word2ind[sent[w1_ind]]][word2ind[sent[w2_ind]]] += 1
                    

    return M, word2ind


def reduce_dimension(M, k=2):
    from sklearn.decomposition import TruncatedSVD
    from numpy.linalg import norm
    n_iter = 10
    random_state = 595
    svd = TruncatedSVD(n_components=k, n_iter=n_iter, random_state=random_state)
    M_R = svd.fit_transform(M)
    M_reduced = M_R / norm(M_R, ord=2, axis=-1, keepdims=True)
    return np.array(M_reduced)


def cosine_similarity(vec1, vec2):
    from numpy.linalg import norm
    sim = np.dot(vec1,vec2) / (norm(vec1) * norm(vec2))
    return sim


def n_most_similar_word_pairs(m_reduced, n=5):
    import random
    sim_dict = {}
    for v1_ind in range(len(m_reduced)):
        for v2_ind in range(len(m_reduced)):
            if (v1_ind != v2_ind):
                sim_dict[(v1_ind, v2_ind)] = cosine_similarity(m_reduced[v1_ind], m_reduced[v2_ind])

    d = [k for k,v in sim_dict.items() if float(v) >= 1]
    if len(d)>n:
        word_ind_p = random.choices(d,k=n)
    else:
        sim_dict_sorted = sorted(sim_dict.items(), key=lambda x:x[1], reverse=True)
        word_ind_p = []
        for _ in range(n):
            word_ind_p.append(sim_dict_sorted[_][0])

    return word_ind_p


def estimate_analogy(m, g, w):
    x = g-m+w
    return x
