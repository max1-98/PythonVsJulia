import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import *


class LinearModel(object):
    def __init__(self, step_size=0.2, max_iter=100, eps=1e-5,
                 theta=None, verbose=True):
        
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    
    def fit(self, x, y):
        """
        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        raise NotImplementedError('Subclass of LinearModel must implement fit method.')


class LogisticRegression(LinearModel):

    def fit(self, x, y):
        """
        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        m,n=np.shape(x)
        self.theta = np.zeros((n,1)).flatten()

        while True:
            oldtheta = np.copy(self.theta)
            ht = 1/(1+np.exp(-x.dot(self.theta)))
            gradJ = 1/m*(x.T).dot(y - ht)
            H = (1/m) * x.T.dot(np.diagflat(ht*(1-ht))).dot(x)
            self.theta -= -np.linalg.solve(H, gradJ)

            if np.linalg.norm(self.theta-oldtheta,ord=1) < self.eps:
                break

    def plotfit(self, x, y):
        x1 = x[:,1]
        x2 = x[:,2]
        plt.scatter(x1,x2,c=y)
        inp = np.linspace(x1.min(),x1.max(),100)
        out = -(self.theta[1]*inp+self.theta[0])/self.theta[2]
        plt.plot(inp,out,"-r")
        plt.show()


def csv_to_matrix(filename):
  """
  Converts a CSV file into a NumPy matrix.

  Args:
    filename: Path to the CSV file.

  Returns:
    A NumPy matrix representing the data from the CSV file.
  """

  # Read the CSV file into a Pandas DataFrame
  df = pd.read_csv(filename)

  # Convert the DataFrame to a NumPy matrix
  matrix = df.to_numpy()

  return matrix

# Example usage:
filename = 'file_name.csv'
data_matrix = csv_to_matrix(filename)
n = np.shape(data_matrix)[1]
x = data_matrix[:,0:n-1]
y = data_matrix[:,n-1]

# Adding a column of 1s to the matrix x
x = np.insert(x, 0, 1, axis=1) 

###
clf = LogisticRegression()

# Manual benchmarking
store = 0
n = 100
for i in range(n):
    start = perf_counter()
    clf.fit(x, y)
    end = perf_counter()
    store += end - start

print(store/n)

clf.plotfit(x,y)