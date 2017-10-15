import numpy as np

from MultinomialMixture import MultinomialMixture

C = 100
K = 20
dim_dataset = 10000

Y = np.empty(dim_dataset, dtype='int32')

for i in range(0, dim_dataset):
    Y[i] = int(np.random.normal(loc=(K/2 + 1), scale=1.0)) % K
    #Y[i] = np.random.uniform(K+1) - 1
    #Y[i] = (i+2) % K

print("End of preprocessing...")

mixture = MultinomialMixture(C, K)

mixture.generate(1000, plot=True)

mixture.train(Y, threshold=0.01, plot=True)

Y_gen = mixture.generate(1000, plot=True)
