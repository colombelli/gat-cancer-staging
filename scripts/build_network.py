from sklearn.neighbors import kneighbors_graph
import pandas as pd
import networkx as nx

class NearestNeighborsNet:

    def __init__(self, feature_df, class_df, k):
        self.feature_df = feature_df
        self.class_df = class_df
        self.k = k


    # def _get_snf_network(self, t=20, K=20):
    #     dfs_values_matrices = []
    #     for df in self.feature_dfs:
    #         dfs_values_matrices.append(df.loc[self.class_df.index, :].values)

    #     affinity_networks = snf.make_affinity(dfs_values_matrices, K=K)
    #     fused_network = snf.snf(affinity_networks, t=t)
    #     np.fill_diagonal(fused_network, 1)
    #     return fused_network


    def all_data_net(self):
        adj_matrix = kneighbors_graph(self.feature_df, n_neighbors=self.k, 
            mode='connectivity', include_self=True).toarray()

        adj_df = pd.DataFrame(data=adj_matrix, index=self.class_df.index.values,
                            columns=self.class_df.index.values)
        G = nx.from_pandas_adjacency(adj_df, create_using=nx.Graph())
        return nx.to_pandas_edgelist(G)

