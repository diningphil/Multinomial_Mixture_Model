#################################################################################
#   										#
#    A simple Multinomial Mixture Model, with fancy plotting utility		#
#    Copyright (C) 2017  Federico Errica					#
#										#
#    This program is free software: you can redistribute it and/or modify	#
#    it under the terms of the GNU General Public License as published by	#
#    the Free Software Foundation, either version 3 of the License, or		#
#    (at your option) any later version.					#
#									     	#
#    This program is distributed in the hope that it will be useful,	     	#
#    but WITHOUT ANY WARRANTY; without even the implied warranty of 	     	#
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the	     	#
#    GNU General Public License for more details.			     	#
#									     	#
#    You should have received a copy of the GNU General Public License	     	#
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.   	#
#										#
#################################################################################

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# Do not remove this import
from mpl_toolkits.mplot3d import Axes3D

class MultinomialMixture:
    def __init__(self, c, k):
        """
        Multinomial mixture model
        :param c: the number of hidden states
        :param k: dimension of emission alphabet, which goes from 0 to k-1
        """
        self.C = c
        self.K = k
        self.smoothing = 0.001  # Laplace smoothing

        # Initialisation of the model's parameters.
        # Notice: the sum-to-1 requirement has been naively satisfied.
        pr = np.random.uniform(size=self.C)
        pr = pr / np.sum(pr)
        self.prior = pr

        self.emission = np.empty((self.K, self.C))
        for i in range(0, self.C):
            em = np.random.uniform(size=self.K)
            em = em / np.sum(em)
            self.emission[:, i] = em

    def train(self, dataset, threshold=0, max_epochs=10, plot=False):
        """
        Training with Expectation Maximisation (EM) Algorithm
        :param dataset: the target labels in a single array
        :param threshold: stopping criterion based on the variation off the likelihood
        :param max_epochs: maximum number of epochs
        :param plot: True if you want to see real-time visualisation of likelihood and parameters' histograms
        """

        if plot:
            gs = gridspec.GridSpec(2, 2,
                                   width_ratios=[0.5, 0.5], height_ratios=[0.5, 0.5])
            ax1 = plt.subplot(gs[0])
            ax2 = plt.subplot(gs[1])
            ax3 = plt.subplot(gs[2])
            ax1.set_xticks(range(self.K))
            ax1.set_title("Data Histogram"); ax1.set_xlabel("Value"); ax1.set_ylabel("Frequency")
            ax2.set_title("Likelihood"); ax2.set_xlabel("Epoch")
            ax3.set_title("Prior"); ax3.set_xlabel("State"); ax3.set_ylabel("Probability")

        likelihood_list = []

        # EM Algorithm
        current_epoch = 1
        old_likelihood = - np.inf
        delta = np.inf

        while current_epoch <= max_epochs and delta > threshold:

            # E-step
            # todo batch the dataset, hence the posterior estimate
            numerator = np.multiply(self.emission[dataset, :], np.reshape(self.prior, (1, self.C)))  # len(dataset)xC
            denominator = np.dot(self.emission[dataset, :], np.reshape(self.prior, (self.C, 1)))  # Ux1
            posterior_estimate = np.divide(numerator, denominator) # todo how to ensure correct broadcasting when U = C?

            # Compute the likelihood
            likelihood = np.sum(np.log(self.emission[dataset, :]*np.reshape(self.prior, (1, self.C))))
            likelihood_list.append(likelihood)
            print("Mixture model training: epoch ", current_epoch, ",  likelihood = ", likelihood)

            delta = likelihood - old_likelihood
            old_likelihood = likelihood

            if plot:

                # Draw a histogram for a uniform distribution with discrete labels
                bins = np.arange(self.K + 1) - 0.5
                ax1.hist(dataset, bins, color='blue', edgecolor='black', linewidth=1.2)

                # Draw the likelihood
                ax2.plot(range(1, current_epoch + 1), likelihood_list, color='red')

                # Draw bar chart for the prior distribution
                ax3.bar(range(0, self.C), self.prior, color='green')

                # Draw the emission
                ax4 = plt.subplot(gs[3], projection='3d')
                ax4.set_title("Emission"); ax4.set_xlabel("K"); ax4.set_ylabel("C")
                ax4.set_zlim3d(0, 1)
                meshX, meshY = np.meshgrid(np.arange(0, self.K), np.arange(0, self.C))
                ax4.plot_surface(meshX, meshY, self.emission[meshX, meshY], cmap=cm.coolwarm)

                plt.tight_layout()
                plt.show(block=False)
                plt.pause(3)

            # M-step
            numerator = self.smoothing + np.sum(posterior_estimate, axis=0)
            denominator = self.smoothing * self.C + np.sum(posterior_estimate)
            self.prior = np.divide(numerator, denominator)

            numerator = self.smoothing + np.zeros((self.K, self.C))
            np.add.at(numerator, dataset, posterior_estimate)  # KxC
            denominator = self.smoothing * self.K + np.sum(posterior_estimate[:, :], axis=0)  # 1xC
            self.emission = np.divide(numerator, np.reshape(denominator, (1, self.C)))

            current_epoch += 1

        return likelihood_list

    def predict(self, prediction_set):
        """
        Takes a set and returns the most likely hidden state assignment for each node
        :param prediction_set: the target labels in a single array
        :returns: most likely hidden state labels
        """
        prods = self.emission[prediction_set, :]*np.reshape(self.prior, (1, self.C))  # len(prediction_set)xC
        return np.argmax(prods, axis=1)

    def generate(self, size, plot=False):
        """
        Generates labels
        :param size: the number of labels to be generated
        :param plot: True if you want to plot the generated histogram
        :returns: a 1-D numpy array of generated labels
        """
        Y_gen = []
        for _ in range(0, size):
            state = np.random.choice(np.arange(0, self.C), p=self.prior)
            emitted_label = np.random.choice(np.arange(0, self.K), p=self.emission[:, state])
            Y_gen.append(emitted_label)

        Y_gen = np.array(Y_gen)

        if plot:
            plt.figure()
            bins = np.arange(self.K + 1) - 0.5
            plt.title("Generated labels"); plt.xlabel("Labels"); plt.ylabel("Frequency")
            plt.hist(Y_gen, bins, color='blue', edgecolor='black', linewidth=1.2)
            plt.show()

        return Y_gen
