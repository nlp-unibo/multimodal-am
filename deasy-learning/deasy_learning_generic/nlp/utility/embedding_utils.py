

import gensim
import gensim.downloader as gloader
import numpy as np
from tqdm import tqdm

from deasy_learning_generic.utility.log_utils import Logger


def build_embeddings_matrix(vocab_size, word_to_idx, embedding_model, embedding_dimension=300,
                            merge_vocabularies=False):
    """
    Builds embedding matrix given the pre-loaded embedding model.
    """

    if merge_vocabularies:
        vocab_size = len(set(list(word_to_idx.keys()) + list(embedding_model.vocab.keys()))) + 1
        vocabulary = word_to_idx
        for key in tqdm(embedding_model.vocab.keys()):
            if key not in vocabulary:
                vocabulary[key] = max(list(vocabulary.values())) + 1
    else:
        vocabulary = word_to_idx

    embedding_matrix = np.zeros((vocab_size, embedding_dimension))

    for word, i in tqdm(vocabulary.items()):
        try:
            embedding_vector = embedding_model[word]

            # Check for any possible invalid term
            if embedding_vector.shape[0] != embedding_dimension:
                embedding_vector = np.zeros(embedding_dimension)
        except (KeyError, TypeError):
            embedding_vector = np.random.uniform(low=-0.05, high=0.05, size=embedding_dimension)

        embedding_matrix[i] = embedding_vector

    return embedding_matrix, vocabulary


def load_embedding_model(model_type: str,
                         embedding_dimension: int = 50) -> gensim.models.keyedvectors.KeyedVectors:
    """
    Loads a pre-trained word embedding model via gensim library.

    :param model_type: name of the word embedding model to load.
    :param embedding_dimension: size of the embedding space to consider

    :return
        - pre-trained word embedding model (gensim KeyedVectors object)
    """

    # Find the correct embedding model name
    if model_type.strip().lower() == 'word2vec':
        download_path = "word2vec-google-news-300"

    elif model_type.strip().lower() == 'glove':
        download_path = "glove-wiki-gigaword-{}".format(embedding_dimension)
    elif model_type.strip().lower() == 'fasttext':
        download_path = "fasttext-wiki-news-subwords-300"
    else:
        raise AttributeError(f"Unsupported embedding model type (Got {model_type})!"
                             f" Available ones: word2vec, glove, fasttext")

    # Check download
    try:
        emb_model = gloader.load(download_path)
    except ValueError as e:
        Logger.get_logger(__name__).exception("Invalid embedding model name! Check the embedding dimension: \n"
                                              "Word2Vec: 300\n"
                                              "GloVe: 50, 100, 200, 300\n"
                                              "Fasttext: 300")
        print("Word2Vec: 300")
        print("Glove: 50, 100, 200, 300")
        raise e

    return emb_model
