import numpy as np
import pandas as pd
import hdbscan


class ClusteredResult(object):
    """ A cluster results object used to hold clustering method and
        frequencies.

        Usage:
        print("[Info] Starting Cluster Ananlysis")
        ofa_df = load_ofa_csv(args.csv_file, add_build=True)
        print("[Info] Loaded CSV " + args.csv_file)
        ofa_df = build_param_table(ofa_df, kmeans_fmt=True)

        clustered = ClusteredResult(
            ofa_df,
            min_cluster_size=20,
            min_samples=20,
            cluster_selection_epsilon=args.cluster_selection_epsilon,
        )
        print("[Info] Clustering Features" + repr(clustered))
        clustered.cluster_features()
        
        print("[Info] Number of Clusters " + str(len(set(clustered.labels))))

        print("[Info] Saving pickles")
        clustered.cluster_analysis("popdb_freqs.pkl") 
    """

    def __init__(
        self,
        ofa_df,
        method="hdbscan",
        min_cluster_size=20,
        min_samples=20,
        cluster_selection_epsilon=0.5
    ) -> None:
        super().__init__()

        self.df = ofa_df
        self.features = ofa_df.drop(columns=["build", "Latency", "Accuracy"])
        self.clusterer = None

        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon

        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            allow_single_cluster=True,
            cluster_selection_method='leaf',
        )

    @property
    def labels(self):
        return self.clusterer.labels_

    def cluster_features(self):
        """ Execute clustering algorithm

            Returns: None
        """
        self.clusterer.fit(self.features)
        return None

    def calc_rel_freqs(self):
        """ Calculate relative frequencies of elastic parameter values
            
            Returns: Dataframe of relative frequencies
        """
        relfreqs = []

        # Create mask of non-noise points
        clustered_points = self.labels != -1

        # For each elastic parameter in a non-noise cluster 
        # calculate and store relative frequencies of values
        for col_ix, column in enumerate(self.features.columns):
            clustered_elastic_params = self.features[clustered_points].iloc[:, col_ix]
            relfreqs.append(clustered_elastic_params.value_counts(normalize=True))

        return pd.concat(relfreqs, keys=self.features.columns)

    def cluster_analysis(self, out_file=None):
        """ Runs the clustering on the elastic parameter data
    
            Args:
            out_file = optional pickle of elastic paramemter relative frequencies

            Returns:
            Dataframe of the relative frequencies of the elastic parameter values
            in the non-noise clusters
        """
        cluster_freqs = self.calc_rel_freqs()
        cluster_freqs_df = pd.DataFrame(cluster_freqs)

        if out_file is not None:
            cluster_freqs_df.to_pickle(out_file)
        
        return cluster_freqs_df
