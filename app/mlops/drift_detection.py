import numpy as np
from utils import load_embeddings_from_db, load_baseline_stats, save_baseline_stats

def calculate_mean_vector(embeddings):
    if not embeddings:
        return None
    return np.mean(np.array(embeddings), axis=0).tolist()

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def detect_drift(threshold=0.95):
    embeddings = load_embeddings_from_db()
    if not embeddings:
        print("EmbeddingがDBにありません")
        return False
    
    current_mean = calculate_mean_vector(embeddings)
    baseline = load_baseline_stats()
    
    if baseline is None:
        print("基準値がありません。初回登録します。")
        save_baseline_stats({"mean_vector": current_mean})
        return False
    
    similarity = cosine_similarity(current_mean, baseline["mean_vector"])
    print(f"現在の平均ベクトルと基準値の類似度: {similarity}")

    if similarity < threshold:
        print("ドリフト検出！")
        save_baseline_stats({"mean_vector": current_mean})
        return True
    else:
        print("ドリフトなし")
        return False
