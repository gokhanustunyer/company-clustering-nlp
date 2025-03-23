import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os
import joblib
import re

class TextClusteringPipeline:
    def __init__(self, n_clusters, random_state=42, sbert_model_name='paraphrase-MiniLM-L6-v2', use_pca=False, pca_components=50):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.sbert_model_name = sbert_model_name
        self.sbert_model = SentenceTransformer(sbert_model_name)
        self.use_pca = use_pca
        self.pca_components = pca_components
        self.kmeans_instance = None
        self.pca_instance = None
        self.normalized_embeddings = None

    def preprocess_text(self, text):
        text = str(text)
        return re.sub(r'[^\w\s]', ' ', text).lower()

    def get_or_create_embeddings(self, cleaned_texts, embeddings_path='embeddings/embeddings.npy'):
        os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
        
        if os.path.exists(embeddings_path):
            embeddings = np.load(embeddings_path)
        else:
            embeddings = self.sbert_model.encode(cleaned_texts, show_progress_bar=True, convert_to_numpy=True)
            np.save(embeddings_path, embeddings)
        
        return embeddings
    
    def preprocess_data(self, df, text_columns, embeddings_path='embeddings/embeddings.npy'):
        combined_texts = df[text_columns].fillna('').apply(lambda x: ' '.join(x.astype(str)), axis=1)
        cleaned_texts = combined_texts.apply(self.preprocess_text).tolist()
        embeddings = self.get_or_create_embeddings(cleaned_texts, embeddings_path)
        self.normalized_embeddings = normalize(embeddings)
        
        if self.use_pca:
            self.pca_instance = PCA(n_components=self.pca_components)
            self.normalized_embeddings = self.pca_instance.fit_transform(self.normalized_embeddings)
        
        return self.normalized_embeddings
            
    def fit(self, df, text_columns, embeddings_path='embeddings/embeddings.npy'):
        if self.normalized_embeddings is None:
            self.normalized_embeddings = self.preprocess_data(df, text_columns, embeddings_path)

        self.kmeans_instance = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init='auto')
        self.cluster_assignments = self.kmeans_instance.fit_predict(self.normalized_embeddings)

    def save_model(self, directory='models'):
        os.makedirs(directory, exist_ok=True)
        model_path = os.path.join(directory, f'sbert_kmeans_model_k{self.n_clusters}.joblib')
        
        model_components = {
            'sbert_model_name': self.sbert_model_name,
            'kmeans_model': self.kmeans_instance,
            'pca_model': self.pca_instance,
            'n_clusters': self.n_clusters,
            'random_state': self.random_state,
            'use_pca': self.use_pca,
            'pca_components': self.pca_components
        }
        
        joblib.dump(model_components, model_path)
        return model_path

    @classmethod
    def load_model(cls, model_path, df, text_columns):
        model_components = joblib.load(model_path)
        pipeline = cls(
            n_clusters=model_components['n_clusters'],
            random_state=model_components['random_state'],
            sbert_model_name=model_components['sbert_model_name'],
            use_pca=model_components['use_pca'],
            pca_components=model_components['pca_components']
        )
        pipeline.kmeans_instance = model_components['kmeans_model']
        pipeline.pca_instance = model_components['pca_model']
        pipeline.preprocess_data(df, text_columns)
        pipeline.cluster_assignments = pipeline.kmeans_instance.fit_predict(pipeline.normalized_embeddings)
        return pipeline

def start_clustering(df, text_columns, cluster_counts, use_pca=False, pca_components=50):
    os.makedirs('results', exist_ok=True)
    pipeline = TextClusteringPipeline(n_clusters=cluster_counts[0], use_pca=use_pca, pca_components=pca_components)
    for k in cluster_counts:
        model_path = f'models/sbert_kmeans_model_k{k}.joblib'
        if os.path.exists(model_path):
            pipeline = TextClusteringPipeline.load_model(model_path, df, text_columns)
        else:
            pipeline.n_clusters = k
            pipeline.fit(df, text_columns)
            pipeline.save_model()

        df_results = df[['id', 'name']].copy()
        df_results['cluster'] = pipeline.cluster_assignments
        df_results.to_excel(f'results/cluster_results_k{k}.xlsx', index=False)

if __name__ == "__main__":
    input_file = "sirket_verileri.csv"
    text_columns = ['about_en', 'professions_en']
    cluster_counts = [170, 180, 200, 210, 220]

    df = pd.read_csv(input_file)
    start_clustering(df, text_columns, cluster_counts, use_pca=True, pca_components=50)
