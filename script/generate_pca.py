import scipy.io
import pandas as pd
import numpy as np
import sys
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import cluster, datasets


import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, normalize
import matplotlib.pyplot as plt
from sklearn import preprocessing


gene_df = pd.read_csv('../data/cancer_full_expression.tsv', sep='\t')

for i in [2]:
    pca = PCA(n_components=i)
    pca_data = gene_df.drop(['sample'], axis=1)

    scaler = preprocessing.StandardScaler()

    Xn = scaler.fit_transform(pca_data)
    
    reduced_matrix = pca.fit_transform(Xn)

#    minmax = preprocessing.MinMaxScaler()
#    X_pca_norm = minmax.fit_transform(reduced_matrix)


    a = pd.DataFrame(reduced_matrix)

    a.index = gene_df['sample']

    a.to_csv("../data/cancer_full_expression_pc"+str(i) + ".csv")

