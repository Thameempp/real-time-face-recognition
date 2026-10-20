import numpy as np

from config import MATCHING_STRATEGY, SIMILARITY_THRESHOLD


def normalize_embedding(embedding):
    embedding = np.asarray(embedding, dtype=np.float32)
    norm = np.linalg.norm(embedding)

    if norm == 0:
        return embedding

    return embedding / norm


class EmbeddingMatcher:
    """Match normalized face embeddings against a cached enrollment gallery."""

    def __init__(self, known_embeddings, known_names, strategy=MATCHING_STRATEGY):
        self.embeddings = np.asarray(known_embeddings, dtype=np.float32)
        self.names = np.asarray(known_names, dtype=str)
        self.strategy = strategy

        if strategy not in {"top3", "max"}:
            raise ValueError(f"Unsupported matching strategy: {strategy}")
        if len(self.embeddings) != len(self.names):
            raise ValueError("Embedding and identity counts do not match")

        self.indices_by_person = {
            person: np.flatnonzero(self.names == person)
            for person in dict.fromkeys(self.names.tolist())
        }

    def match(self, embedding, threshold=SIMILARITY_THRESHOLD):
        return self.match_many([embedding], threshold=threshold)[0]

    def match_many(self, embeddings, threshold=SIMILARITY_THRESHOLD):
        if not len(self.embeddings) or not len(embeddings):
            return [("Unknown", 0.0) for _ in embeddings]

        queries = np.asarray(embeddings, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries[np.newaxis, :]
        if queries.shape[1] != self.embeddings.shape[1]:
            return [("Unknown", 0.0) for _ in queries]

        norms = np.linalg.norm(queries, axis=1, keepdims=True)
        valid = np.isfinite(queries).all(axis=1) & (norms[:, 0] > 0)
        normalized = np.zeros_like(queries)
        normalized[valid] = queries[valid] / norms[valid]
        similarities = normalized @ self.embeddings.T
        results = []

        for row_index, row in enumerate(similarities):
            if not valid[row_index]:
                results.append(("Unknown", 0.0))
                continue

            best_name = "Unknown"
            best_score = -1.0
            for person_name, indices in self.indices_by_person.items():
                person_scores = row[indices]
                if self.strategy == "max":
                    score = float(np.max(person_scores))
                else:
                    top_count = min(3, len(person_scores))
                    top_scores = np.partition(
                        person_scores,
                        -top_count,
                    )[-top_count:]
                    score = float(np.mean(top_scores))

                if score > best_score:
                    best_name = person_name
                    best_score = score

            if best_score < threshold:
                results.append(("Unknown", best_score))
            else:
                results.append((best_name, best_score))

        return results


def recognize_embedding(
    embedding,
    known_embeddings,
    known_names,
    strategy=MATCHING_STRATEGY,
    threshold=SIMILARITY_THRESHOLD,
):
    return EmbeddingMatcher(
        known_embeddings,
        known_names,
        strategy=strategy,
    ).match(embedding, threshold=threshold)
