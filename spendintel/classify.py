"""Taxonomy-aware classification. Zero-shot (no labels) is the default; label-assisted mode is separate and used only if labels are configured."""
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from .text import clean_text

def _softmax(z, t=1.0):
    z = z * t; z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


class Embedder:
    """Use locally cached E5 when available; otherwise use offline TF-IDF."""

    def __init__(self, backend="auto", model="intfloat/multilingual-e5-small"):
        self.backend = "tfidf"
        self.st = None
        self.fallback_reason = None

        if backend in ("auto", "e5"):
            try:
                from sentence_transformers import SentenceTransformer

                # Never download model weights during pipeline execution.
                self.st = SentenceTransformer(
                    model,
                    local_files_only=True
                )
                self.backend = "e5"

            except Exception as exc:
                self.st = None
                self.fallback_reason = str(exc)

                if backend == "e5":
                    raise RuntimeError(
                        "E5 model is not available locally. "
                        "Download/cache it before execution or use --embedder auto."
                    ) from exc

    def scores(self, texts, path_texts):
        if self.backend == "e5":
            a = self.st.encode(["query: " + t for t in texts], batch_size=64, normalize_embeddings=True)
            b = self.st.encode(["passage: " + t for t in path_texts], batch_size=64, normalize_embeddings=True)
            return a @ b.T
        v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2)
        v.fit(list(texts) + list(path_texts)); return (v.transform(texts) @ v.transform(path_texts).T).toarray()

def zero_shot(texts, leaves, embedder):
    ptxt = [" ".join(p) for p in leaves]
    S = embedder.scores(list(clean_text(pd.Series(texts))), list(clean_text(pd.Series(ptxt))))
    P = _softmax(S, 20.0); i = P.argmax(1)
    return [leaves[k] for k in i], P.max(1)

def label_assisted(train_text, train_paths, texts, leaves):
    """Fast hierarchical linear classifiers on word + character TF-IDF.

    SGD with modified-Huber loss is used instead of a high-cost one-vs-rest
    solver: it scales to tens of thousands of multilingual procurement lines.
    The level-1 prediction gates leaf scores to a valid branch.
    """
    from scipy.sparse import hstack
    from sklearn.linear_model import SGDClassifier
    wv = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=40000)
    cv = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, sublinear_tf=True, max_features=40000)
    ct = clean_text(pd.Series(train_text)); cq = clean_text(pd.Series(texts))
    X = hstack([wv.fit_transform(ct), cv.fit_transform(ct)]).tocsr()
    Q = hstack([wv.transform(cq), cv.transform(cq)]).tocsr()
    y = np.asarray(["|".join(p) for p in train_paths]); y1 = np.asarray([p[0] for p in train_paths])
    leaf = SGDClassifier(loss="modified_huber", alpha=2e-5, max_iter=90, tol=1e-3,
                         class_weight="balanced", random_state=42, n_jobs=-1).fit(X, y)
    top = SGDClassifier(loss="modified_huber", alpha=2e-5, max_iter=60, tol=1e-3,
                        class_weight="balanced", random_state=42, n_jobs=-1).fit(X, y1)
    D = leaf.decision_function(Q)
    if D.ndim == 1: D = np.column_stack([-D, D])
    p1 = top.predict(Q)
    cls = np.asarray(leaf.classes_); cls1 = np.asarray([c.split("|")[0] for c in cls])
    D = np.where(cls1[None, :] == p1[:, None], D, -1e9)
    P = _softmax(np.where(D < -1e8, -50.0, D), 1.0)
    return [tuple(cls[k].split("|")) for k in P.argmax(1)], P.max(1)
