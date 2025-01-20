import faiss
import os

class Faiss:
    def __init__(self, index_type='FlatL2', dim=128, nlist=100, faiss_index=None):
        """
        Initializes the FAISS index based on the index type and loads it from a file if provided.
        
        Parameters:
            - index_type: Type of the FAISS index ('FlatL2', 'FlatIP', 'IVFFlat', 'IVFPQ', etc.).
            - dim: Dimensionality of the feature vectors.
            - nlist: Number of clusters (only used for IVF-based indices).
            - faiss_index: Path to a saved FAISS index file (optional).
        """
        self.index = None
        self.dim = dim
        self.nlist = nlist
        self.feature_indexes = []
        self.error_indexes = []

        if faiss_index and os.path.exists(faiss_index):
            self.load_index(faiss_index)
        else:
            self._initialize_index(index_type)

    def _initialize_index(self, index_type):
        """Handle initialize different FAISS index types."""
        if index_type == 'FlatL2':
            self.index = faiss.IndexFlatL2(self.dim)
        elif index_type == 'FlatIP':
            self.index = faiss.IndexFlatIP(self.dim)
        elif index_type == 'IVFFlat':
            quantizer = faiss.IndexFlatL2(self.dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.dim, self.nlist)
            self.index.train = quantizer.train
        else:
            raise ValueError(f"Unsupported index type: {index_type}")

    def add_embedding(self, embedding):
        """Add feature vectors to the FAISS index."""
        try:
            self.index.add(embedding)
        except Exception as e:
            self.error_indexes.append((embedding, str(e)))

    def search(self, query_embedding, k=1):
        """Search for the k nearest neighbors."""
        distances, indices = self.index.search(query_embedding, k)
        return distances, indices

    def save_index(self, file_path):
        """
        Save the FAISS index to a file.

        Parameters:
            - file_path: Path to save the index file.
        """
        if self.index is None:
            raise ValueError("The index has not been initialized.")
        try:
            faiss.write_index(self.index, file_path)
            print(f"Index saved successfully to {file_path}")
        except Exception as e:
            print(f"Error saving index: {e}")

    def load_index(self, file_path):
        """
        Load the FAISS index from a file.

        Parameters:
            - file_path: Path to the saved index file.
        """
        try:
            self.index = faiss.read_index(file_path)
            print(f"Index loaded successfully from {file_path}")
        except Exception as e:
            print(f"Error loading index: {e}. Reinitializing index with default index type.")
            self._initialize_index('FlatL2')
