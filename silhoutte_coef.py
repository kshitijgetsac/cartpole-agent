import numpy as np

def silhouette_score(X, labels):
    unique_labels = np.unique(labels)
    n = X.shape[0]
    silhouette_vals = np.zeros(n)
    
    for i in range(n):
        same_cluster = X[labels == labels[i]]
        # Average distance within cluster (a)
        a = np.mean(np.linalg.norm(same_cluster - X[i], axis=1))
        
        # Compute average distance to the nearest cluster (b)
        b = np.inf
        for label in unique_labels:
            if label != labels[i]:
                other_cluster = X[labels == label]
                distance = np.mean(np.linalg.norm(other_cluster - X[i], axis=1))
                b = min(b, distance)
                
        silhouette_vals[i] = (b - a) / max(a, b)
    return np.mean(silhouette_vals)

# Example usage:
# score = silhouette_score(X, labels)
# print("Silhouette Coefficient:", score)
