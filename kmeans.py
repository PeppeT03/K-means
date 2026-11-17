
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


# ==========================================================
# 1. CARICAMENTO DEL DATASET IRIS
# ==========================================================

# Carichiamo il dataset Iris di Scikit-Learn
iris = load_iris()

# X contiene le caratteristiche dei fiori
# y contiene le rispettive specie reali
X, y = iris.data, iris.target

# Recuperiamo i nomi delle quattro caratteristiche
feature_names = iris.feature_names

# Visualizziamo le dimensioni del dataset e le caratteristiche
print("Forma di X:", X.shape)
print("Forma di y:", y.shape)
print("Caratteristiche:", feature_names)


# ==========================================================
# 2. STANDARDIZZAZIONE DEI DATI
# ==========================================================

# Creiamo lo standardizzatore
scaler = StandardScaler()

# Calcoliamo media e deviazione standard
# e trasformiamo tutte le caratteristiche
X_std = scaler.fit_transform(X)

# Confrontiamo la prima osservazione prima e dopo
# la standardizzazione
print("\nPrima:", X[0])
print("Dopo:", X_std[0])


# ==========================================================
# 3. ELBOW METHOD - RICERCA DEL NUMERO DI CLUSTER
# ==========================================================

# Lista che conterrà i valori WCSS per ogni K
wcss = []

# Proviamo un numero di cluster da 1 a 10
k_values = range(1, 11)

# Ripetiamo K-Means per ogni numero di cluster
for k in k_values:

    # Creiamo un modello K-Means
    kmeans = KMeans(
        n_clusters=k,
        init="k-means++",
        random_state=42,
        n_init=10
    )

    # Addestriamo il modello sui dati standardizzati
    kmeans.fit(X_std)

    # Salviamo l'inerzia (WCSS) ottenuta
    wcss.append(kmeans.inertia_)


# Creiamo il grafico dell'Elbow Method
plt.figure(figsize=(8, 6))

# Disegniamo i valori WCSS per ogni K
plt.plot(
    k_values,
    wcss,
    "o-",
    color="blue",
    markersize=8
)

# Impostiamo i nomi degli assi
plt.xlabel("Numero di cluster (K)")
plt.ylabel("WCSS (Inertia)")

# Mostriamo tutti i valori di K sull'asse orizzontale
plt.xticks(k_values)

# Titolo del grafico
plt.title("Elbow Method")

# Aggiungiamo la griglia
plt.grid(True)

# Sistemiamo gli spazi
plt.tight_layout()

# Salviamo il grafico in alta qualità
plt.savefig("elbow_method.png", dpi=300)

# Mostriamo il grafico
plt.show()


# ==========================================================
# 4. ADDESTRAMENTO DEL MODELLO K-MEANS DEFINITIVO
# ==========================================================

# Dopo aver analizzato l'Elbow Method scegliamo K = 3
kmeans = KMeans(
    n_clusters=3,
    init="k-means++",
    n_init=10,
    random_state=42
)

# Addestriamo K-Means e assegniamo ogni fiore a un cluster
clusters = kmeans.fit_predict(X_std)

# Recuperiamo le coordinate dei tre centroidi
centroids = kmeans.cluster_centers_

# Recuperiamo il numero di iterazioni effettuate
n_iter = kmeans.n_iter_

# Stampiamo le informazioni del modello
print("\nIterazioni:", n_iter)
print("Centroidi:\n", centroids)


# ==========================================================
# 5. RIDUZIONE DELLE DIMENSIONI CON PCA
# ==========================================================

# Creiamo una PCA che mantiene solamente due componenti
pca = PCA(
    n_components=2,
    random_state=42
)

# Trasformiamo i dati standardizzati da 4 a 2 dimensioni
X_pca = pca.fit_transform(X_std)

# Visualizziamo le dimensioni prima e dopo la PCA
print("\nForma originale:", X_std.shape)
print("Forma dopo PCA:", X_pca.shape)


# ==========================================================
# 6. PROIEZIONE DEI CENTROIDI NELLO SPAZIO PCA
# ==========================================================

# I centroidi sono ancora espressi nelle 4 dimensioni originali
# standardizzate.

# Li trasformiamo nello stesso spazio bidimensionale dei fiori
centroids_pca = pca.transform(centroids)


# ==========================================================
# 7. CREAZIONE DEL SUBPLOT CON DUE GRAFICI AFFIANCATI
# ==========================================================

# Creiamo una figura con una riga e due colonne
fig, axes = plt.subplots(
    1, 2,
    figsize=(16, 7)
)

# Titolo generale della figura
fig.suptitle(
    "Confronto tra K-Means e specie reali di Iris",
    fontsize=16,
    fontweight="bold"
)


# ==========================================================
# 8. PRIMO GRAFICO: CLUSTER TROVATI DA K-MEANS
# ==========================================================

# Disegniamo i fiori utilizzando le coordinate PCA
# Il colore dipende dal cluster assegnato da K-Means
scatter_cluster = axes[0].scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=clusters,
    cmap="viridis",
    s=50,
    alpha=0.7
)

# Disegniamo i centroidi con grandi X rosse
centroidi_plot = axes[0].scatter(
    centroids_pca[:, 0],
    centroids_pca[:, 1],
    c="red",
    s=220,
    marker="X",
    edgecolor="black",
    linewidth=1,
    label="Centroidi"
)

# Recuperiamo automaticamente i colori dei tre cluster
handles_cluster, _ = scatter_cluster.legend_elements()

# Creiamo i nomi della legenda
nomi_cluster = [
    f"Cluster {i}" for i in range(kmeans.n_clusters)
]

# Aggiungiamo anche il simbolo dei centroidi alla legenda
axes[0].legend(
    handles=handles_cluster + [centroidi_plot],
    labels=nomi_cluster + ["Centroidi"],
    title="Gruppi individuati",
    loc="best"
)

# Impostiamo titolo e nomi degli assi
axes[0].set_title("K-Means: cluster e centroidi")
axes[0].set_xlabel("Prima componente principale (PC1)")
axes[0].set_ylabel("Seconda componente principale (PC2)")

# Aggiungiamo la griglia
axes[0].grid(alpha=0.3)


# ==========================================================
# 9. SECONDO GRAFICO: SPECIE REALI DI IRIS
# ==========================================================

# Disegniamo gli stessi fiori utilizzando sempre la PCA
# Questa volta i colori dipendono dalle specie reali (y)
scatter_reali = axes[1].scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=y,
    cmap="plasma",
    s=50,
    alpha=0.7
)

# Recuperiamo gli elementi della legenda
handles_reali, _ = scatter_reali.legend_elements()

# Utilizziamo i nomi scientifici delle tre specie
axes[1].legend(
    handles=handles_reali,
    labels=iris.target_names,
    title="Specie reali",
    loc="best"
)

# Impostiamo titolo e nomi degli assi
axes[1].set_title("Distribuzione delle specie reali")
axes[1].set_xlabel("Prima componente principale (PC1)")
axes[1].set_ylabel("Seconda componente principale (PC2)")

# Aggiungiamo la griglia
axes[1].grid(alpha=0.3)


# ==========================================================
# 10. VISUALIZZAZIONE E SALVATAGGIO FINALE
# ==========================================================

# Sistemiamo automaticamente gli spazi tra i grafici
plt.tight_layout(rect=[0, 0, 1, 0.94])

# Salviamo entrambi i grafici nella stessa immagine
plt.savefig(
    "confronto_kmeans_iris.png",
    dpi=300,
    bbox_inches="tight"
)

# Mostriamo la figura completa
plt.show()
