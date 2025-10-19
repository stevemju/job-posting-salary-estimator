from sentence_transformers import SentenceTransformer


def get_embedding_dimension(model_name) -> int:
    model = SentenceTransformer(model_name)
    return model.get_sentence_embedding_dimension()
