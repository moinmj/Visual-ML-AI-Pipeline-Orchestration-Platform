import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from backend.app.recipes.base.recipe import BaseRecipe


class TextVectorizerRecipe(BaseRecipe):
    recipe_id = "text_vectorizer"
    name = "Text Vectorizer (TF-IDF / Word2Vec / Count)"
    version = "1.0.0"
    category = "nlp"
    description = "Converts unstructured text columns into numeric feature vectors using TF-IDF, Bag-of-Words, or Word2Vec / Latent Semantic Embeddings."
    input_types = ["dataframe"]
    output_types = ["dataframe"]

    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "title": "Vectorization Technique",
                    "enum": ["tfidf", "count", "word2vec"],
                    "default": "tfidf",
                    "description": "TF-IDF weights words by uniqueness; Count is raw frequency; Word2Vec generates dense semantic vectors."
                },
                "column": {
                    "type": "string",
                    "title": "Target Text Column",
                    "description": "The text column to vectorize. If left empty, first text/string column will be used."
                },
                "max_features": {
                    "type": "integer",
                    "title": "Max Feature Dimensions",
                    "default": 50,
                    "minimum": 5,
                    "maximum": 5000,
                    "description": "Number of top word vector columns to produce."
                },
                "ngram_range": {
                    "type": "string",
                    "title": "N-gram Range",
                    "enum": ["unigram (1,1)", "bigram (1,2)", "trigram (1,3)"],
                    "default": "unigram (1,1)"
                },
                "drop_original": {
                    "type": "boolean",
                    "title": "Drop Original Text Column",
                    "default": True,
                    "description": "Drop raw text string column so output dataframe is ready for ML training."
                }
            },
            "required": ["method"]
        }

    def execute(self, inputs: Dict[str, Any], config: Dict[str, Any], context: Optional[Any] = None) -> Dict[str, Any]:
        df: pd.DataFrame = inputs.get("dataframe")
        if df is None:
            raise ValueError("TextVectorizer expects 'dataframe' in inputs.")

        df = df.copy()
        method = config.get("method", "tfidf")
        target_col = config.get("column")
        max_feat = int(config.get("max_features", 50))
        ngram_str = config.get("ngram_range", "unigram (1,1)")
        drop_orig = config.get("drop_original", True)

        # Parse ngrams
        if "trigram" in ngram_str:
            ngram_tuple = (1, 3)
        elif "bigram" in ngram_str:
            ngram_tuple = (1, 2)
        else:
            ngram_tuple = (1, 1)

        # Find target column
        if not target_col or target_col not in df.columns:
            text_candidates = [c for c in df.columns if df[c].dtype == "object" or str(df[c].dtype) == "string"]
            if not text_candidates:
                return {"dataframe": df, "metrics": {"warning": "No text column found for vectorization."}}
            target_col = text_candidates[0]

        corpus = df[target_col].fillna("").astype(str).tolist()

        if method == "tfidf":
            vectorizer = TfidfVectorizer(max_features=max_feat, ngram_range=ngram_tuple)
            matrix = vectorizer.fit_transform(corpus)
            feature_names = [f"tfidf_{f}" for f in vectorizer.get_feature_names_out()]
            feat_df = pd.DataFrame(matrix.toarray(), columns=feature_names, index=df.index)

        elif method == "count":
            vectorizer = CountVectorizer(max_features=max_feat, ngram_range=ngram_tuple)
            matrix = vectorizer.fit_transform(corpus)
            feature_names = [f"count_{f}" for f in vectorizer.get_feature_names_out()]
            feat_df = pd.DataFrame(matrix.toarray(), columns=feature_names, index=df.index)

        elif method == "word2vec":
            # Attempt Gensim Word2Vec; fallback to SVD Latent Semantic Analysis (LSA)
            gensim_w2v = False
            try:
                from gensim.models import Word2Vec
                tokens_list = [text.lower().split() for text in corpus]
                # Filter non-empty
                train_tokens = [t for t in tokens_list if len(t) > 0]
                if len(train_tokens) > 0:
                    vec_size = min(max_feat, 100)
                    w2v_model = Word2Vec(sentences=train_tokens, vector_size=vec_size, window=5, min_count=1, workers=1)
                    
                    doc_vectors = []
                    for tokens in tokens_list:
                        valid_vecs = [w2v_model.wv[t] for t in tokens if t in w2v_model.wv]
                        if valid_vecs:
                            doc_vectors.append(np.mean(valid_vecs, axis=0))
                        else:
                            doc_vectors.append(np.zeros(vec_size))
                    
                    feature_names = [f"w2v_dim_{i}" for i in range(vec_size)]
                    feat_df = pd.DataFrame(doc_vectors, columns=feature_names, index=df.index)
                    gensim_w2v = True
            except Exception:
                gensim_w2v = False

            if not gensim_w2v:
                # Dense Latent Semantic Analysis (LSA) dense embedding fallback
                tfidf_vec = TfidfVectorizer(max_features=max(max_feat * 2, 100))
                tfidf_mat = tfidf_vec.fit_transform(corpus)
                n_components = min(max_feat, tfidf_mat.shape[1] - 1, tfidf_mat.shape[0] - 1)
                if n_components > 0:
                    svd = TruncatedSVD(n_components=n_components, random_state=42)
                    dense_vecs = svd.fit_transform(tfidf_mat)
                    feature_names = [f"w2v_emb_{i}" for i in range(n_components)]
                    feat_df = pd.DataFrame(dense_vecs, columns=feature_names, index=df.index)
                else:
                    feat_df = pd.DataFrame(tfidf_mat.toarray(), columns=[f"w2v_bow_{i}" for i in range(tfidf_mat.shape[1])], index=df.index)

        # Merge vectors into original DataFrame
        if drop_orig:
            df = df.drop(columns=[target_col])

        out_df = pd.concat([df, feat_df], axis=1)

        return {
            "dataframe": out_df,
            "vectorizer": vectorizer,
            "text_column": target_col,
            "metrics": {
                "method_applied": method,
                "target_column": target_col,
                "vector_features_created": len(feat_df.columns),
                "total_columns": len(out_df.columns)
            }
        }
