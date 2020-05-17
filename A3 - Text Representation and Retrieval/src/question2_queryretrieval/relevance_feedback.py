from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import Normalizer
import numpy as np

def relevance_feedback(vec_docs, vec_queries, sim, n=10):
    """
    relevance feedback
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)
    """
    vec_docs_np = vec_docs.toarray()
    vec_queries_np = vec_queries .toarray()
    alpha = 0.75
    beta = 1
    for i in range(3):
        for q in range(sim.shape[1]):
            rel_quer_docs = np.argsort(-sim[:, q])[:n]
            rel_doc_sum = np.zeros((1,vec_docs.shape[1]))
            non_rel_doc_sum = np.zeros((1,vec_docs.shape[1]))
            for r in range(sim.shape[0]):
                if(r in rel_quer_docs):
                    rel_doc_sum += vec_docs_np[r]
                else:
                    non_rel_doc_sum += vec_docs_np[r]
            rel_doc_vec_mean = rel_doc_sum/ n
            non_rel_doc_vec_mean = non_rel_doc_sum / (sim.shape[0]-n)
            vec_queries_np[q] = vec_queries_np[q] + (alpha*rel_doc_vec_mean )- (beta*non_rel_doc_vec_mean)
    rf_sim = cosine_similarity(vec_docs_np, vec_queries_np)
    return rf_sim


def relevance_feedback_exp(vec_docs, vec_queries, sim, tfidf_model, n=10):
    """
    relevance feedback with expanded queries
    Parameters
        ----------
        vec_docs: sparse array,
            tfidf vectors for documents. Each row corresponds to a document.
        vec_queries: sparse array,
            tfidf vectors for queries. Each row corresponds to a document.
        sim: numpy array,
            matrix of similarities scores between documents (rows) and queries (columns)
        tfidf_model: TfidfVectorizer,
            tf_idf pretrained model
        n: integer
            number of documents to assume relevant/non relevant

    Returns
    -------
    rf_sim : numpy array
        matrix of similarities scores between documents (rows) and updated queries (columns)"""

    vec_docs_np = vec_docs.toarray()
    vec_queries_np = vec_queries.toarray()
    alpha = 0.75
    beta = 1
    for i in range(3):
        for q in range(sim.shape[1]):
            rel_quer_docs = np.argsort(-sim[:, q])[:n]
            rel_doc_sum = np.zeros((1,vec_docs.shape[1]))
            non_rel_doc_sum = np.zeros((1,vec_docs.shape[1]))
            for r in range(sim.shape[0]):
                if(r in rel_quer_docs):
                    rel_doc_sum += vec_docs_np[r]
                else:
                    non_rel_doc_sum += vec_docs_np[r]
            rel_doc_vec_mean = rel_doc_sum/ n
            non_rel_doc_vec_mean = non_rel_doc_sum / (sim.shape[0]-n)
            vec_queries_np[q] = vec_queries_np[q] + (alpha*rel_doc_vec_mean ) - (beta*non_rel_doc_vec_mean)
    normalise = Normalizer().fit(vec_docs_np)
    normalise.transform(vec_docs_np)
    tt_sim = np.dot(np.transpose(vec_docs_np),vec_docs_np)
    # print(tt_sim.shape)
    for query in range(sim.shape[1]):

        most_imp_term = np.argmax(vec_queries_np[query])
        similar_terms = np.argsort(-tt_sim[most_imp_term])[:11]
        for term in range(len(similar_terms)):
            vec_queries_np[query, similar_terms[term]] = vec_queries_np[query, most_imp_term]

    # transformer = Normalizer().fit(vec_docs_np)
    # transformer.transform(vec_docs_np)

    rf_sim = cosine_similarity(vec_docs_np, vec_queries_np)

    return rf_sim
