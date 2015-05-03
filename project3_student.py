#this file contains major functions
import random as ra
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from numpy import linalg as LA
from scipy.stats import norm, multivariate_normal


# Reads a data matrix from file.
# Output: X: data matrix.
def readData(file):
    X = []
    with open(file,"r") as f:
        for line in f:
            X.append(map(float,line.split(" ")))
    return np.array(X)

# plot 2D toy data
# input: X: n*d data matrix;
#        K: number of mixtures;
#        Mu: K*d matrix, each row corresponds to a mixture mean;
#        P: K*1 matrix, each entry corresponds to the weight for a mixture;
#        Var: K*1 matrix, each entry corresponds to the variance for a mixture;
#        Label: n*K matrix, each row corresponds to the soft counts for all mixtures for an example
#        title: a string represents the title for the plot
def plot2D(X,K,Mu,P,Var,Label,title):
    r=0.25
    color=["r","b","k","y","m","c"]
    n,d = np.shape(X)
    per= Label/(1.0*np.tile(np.reshape(np.sum(Label,axis=1),(n,1)),(1,K)))
    fig=plt.figure()
    plt.title(title)
    ax=plt.gcf().gca()
    ax.set_xlim((-20,20))
    ax.set_ylim((-20,20))
    for i in xrange(len(X)):
        angle=0
        for j in xrange(K):
            cir=pat.Arc((X[i,0],X[i,1]),r,r,0,angle,angle+per[i,j]*360,edgecolor=color[j])
            ax.add_patch(cir)
            angle+=per[i,j]*360
    for j in xrange(K):
        sigma = np.sqrt(Var[j])
        circle=plt.Circle((Mu[j,0],Mu[j,1]),sigma,color=color[j],fill=False)
        ax.add_artist(circle)
        text=plt.text(Mu[j,0],Mu[j,1],"mu=("+str("%.2f" %Mu[j,0])+","+str("%.2f" %Mu[j,1])+"),stdv="+str("%.2f" % np.sqrt(Var[j])))
        ax.add_artist(text)
    plt.axis('equal')
    plt.show()

# initialization for k means model for toy data
# input: X: n*d data matrix;
#        K: number of mixtures;
#        fixedmeans: is an optional variable which is
#        used to control whether Mu is generated from a deterministic way
#        or randomized way
# output: Mu: K*d matrix, each row corresponds to a mixture mean;
#         P: K*1 matrix, each entry corresponds to the weight for a mixture;
#         Var: K*1 matrix, each entry corresponds to the variance for a mixture;
def init(X,K,fixedmeans=False):
    n, d = np.shape(X)
    P=np.ones((K,1))/float(K)

    if (fixedmeans):
        assert(d==2 and K==3)
        Mu = np.array([[4.33,-2.106],[3.75,2.651],[-1.765,2.648]])
    else:
        # select K random points as initial means
        rnd = np.random.rand(n,1)
        ind = sorted(range(n),key = lambda i: rnd[i])
        Mu = np.zeros((K,d))
        for i in range(K):
            Mu[i,:] = np.copy(X[ind[i],:])

    Var=np.mean( (X-np.tile(np.mean(X,axis=0),(n,1)))**2 )*np.ones((K,1))
    return (Mu,P,Var)

# K Means method
# input: X: n*d data matrix;
#        K: number of mixtures;
#        Mu: K*d matrix, each row corresponds to a mixture mean;
#        P: K*1 matrix, each entry corresponds to the weight for a mixture;
#        Var: K*1 matrix, each entry corresponds to the variance for a mixture;
# output: Mu: K*d matrix, each row corresponds to a mixture mean;
#         P: K*1 matrix, each entry corresponds to the weight for a mixture;
#         Var: K*1 matrix, each entry corresponds to the variance for a mixture;
#         post: n*K matrix, each row corresponds to the soft counts for all mixtures for an example
def kMeans(X, K, Mu, P, Var):
    prevCost=-1.0; curCost=0.0
    n=len(X)
    d=len(X[0])
    while abs(prevCost-curCost)>1e-4:
        post=np.zeros((n,K))
        prevCost=curCost
        #E step
        for i in xrange(n):
            post[i,np.argmin(np.sum(np.square(np.tile(X[i,],(K,1))-Mu),axis=1))]=1
        #M step
        n_hat=np.sum(post,axis=0)
        P=n_hat/float(n)
        curCost = 0
        for i in xrange(K):
            Mu[i,:]= np.dot(post[:,i],X)/float(n_hat[i])
            # summed squared distance of points in the cluster from the mean
            sse = np.dot(post[:,i],np.sum((X-np.tile(Mu[i,:],(n,1)))**2,axis=1))
            curCost += sse
            Var[i]=sse/float(d*n_hat[i])
        print curCost
    # return a mixture model retrofitted from the K-means solution
    return (Mu,P,Var,post)

# RMSE criteria
# input: X: n*d data matrix;
#        Y: n*d data matrix;
# output: RMSE
def rmse(X,Y):
    return np.sqrt(np.mean((X-Y)**2))

def dist(X, axis=0):
    return np.sqrt((X**2).sum(axis=axis))

# E step of EM algorithm
# input: X: n*d data matrix;
#        K: number of mixtures;
#        Mu: K*d matrix, each row corresponds to a mixture mean;
#        P: K*1 matrix, each entry corresponds to the weight for a mixture;
#        Var: K*1 matrix, each entry corresponds to the variance for a mixture;
# output:post: n*K matrix, each row corresponds to the soft counts for all mixtures for an example
#        LL: a Loglikelihood value

def Estep(X, K, Mu, P, Var):
    n, d = np.shape(X)
    post = np.zeros((n, K))
    LL = 0

    X = X.copy()
    Mu = Mu.copy()

    for i in xrange(n):
        stored_values = np.zeros(K)
        for j in xrange(K):
            exp_term = -1.0/(2*Var[j])*(dist(X[i]-Mu[j])**2)

            stored_values[j] = P[j] * (1.0/(2*np.pi*Var[j])**(d/2.0))*np.exp(exp_term)
        LL += np.log(stored_values.sum())
        post[i,:] = stored_values/stored_values.sum()

    return (post, LL)

# def Estep(X,K,Mu,P,Var):
#     n,d = np.shape(X) # n data points of dimension d
#     post = np.zeros((n,K)) # posterior probabilities tbd
#
#     X_copies = np.expand_dims(X.copy(), 0) #(1, n, d)
#     Mu_copies = np.expand_dims(Mu.copy(), 1) #(K, 1, d)
#
#     distances = dist(X_copies - Mu_copies, axis=2) #(K, n)
#
#     std_devs = np.sqrt(Var)
#
#     probabilities = (1.0/(2*np.pi*Var)**(d/2.0)) * norm.pdf(distances, scale=std_devs) #(K, n)
#
#     weighted_probs = P * probabilities #(K, n)
#
#     post = (weighted_probs.T/np.expand_dims(weighted_probs.sum(axis=0), 1)) #(n, K)
#
#     LL = (post * np.log(weighted_probs).T).sum()
#     LL = (np.log(weighted_probs.sum(axis=0))).sum()
#
#     return (post, LL)



# M step of EM algorithm
# input: X: n*d data matrix;
#        K: number of mixtures;
#        Mu: K*d matrix, each row corresponds to a mixture mean;
#        P: K*1 matrix, each entry corresponds to the weight for a mixture;
#        Var: K*1 matrix, each entry corresponds to the variance for a mixture;
#        post: n*K matrix, each row corresponds to the soft counts for all mixtures for an example
# output:Mu: updated Mu, K*d matrix, each row corresponds to a mixture mean;
#        P: updated P, K*1 matrix, each entry corresponds to the weight for a mixture;
#        Var: updated Var, K*1 matrix, each entry corresponds to the variance for a mixture;

def Mstep(X, K, Mu, P, Var, post):
    n, d = np.shape(X)

    n_hats = post.sum(axis=0)
    P = n_hats/float(n)

    # Mu = np.empty((K, d))
    # for k in xrange(K):
    #     Mu[k,:] = (1.0/n_hats[k]) * (np.expand_dims(post[:, k], 1) * X).sum(axis=0)

    weighted_mean_of_points = (np.expand_dims(X, 1) * np.expand_dims(post, 2)).sum(axis=0) #(K, d)
    Mu = 1.0/np.expand_dims(n_hats, 1) * weighted_mean_of_points

    Var = np.empty((K, 1))
    for k in xrange(K):
        Var[k, :] = (1.0/(d*n_hats[k])) * (post[:, k] * dist(X-Mu[k, :], axis=1)**2).sum()

    # DOESNT WORK YET
    # displacement_vectors = (np.expand_dims(X, 1) - np.expand_dims(Mu, 0))
    # mss = dist(displacement_vectors, axis=2)**2
    # weighted_mss = (mss * post).sum(axis=0)
    # Var_hat = 1.0/(d * n_hats) * weighted_mss
    # assert np.allclose(Var_hat, Var)


    return (Mu, P, Var)

# def Mstep(X,K,Mu,P,Var,post):
#     n,d = np.shape(X) # n data points of dimension d
#
#     n_hat = post.sum(axis=0)
#     P = n_hat/float(n)
#
#     weighted_mean_of_points = (np.expand_dims(X, 1) * np.expand_dims(post, 2)).sum(axis=0) #(K, d)
#
#     Mu = 1.0/np.expand_dims(n_hat, 1) * weighted_mean_of_points
#
#
#     mean_squared_spread = np.sqrt((np.expand_dims(X, 1) - np.expand_dims(Mu, 0))**2).sum(axis=2) #(n, K)
#     weighted_mss = (mean_squared_spread * post).sum(axis=0)
#
#     Var = 1.0/(d * n_hat) * weighted_mss
#
#     P = np.expand_dims(P, 1)
#     Var = np.expand_dims(Var, 1)
#
#     return (Mu,P,Var)

# ----------------------------------------------------------------------------------------------------
# mixture of Gaussians
# input: X: n*d data matrix;
#        K: number of mixtures;
#        Mu: K*d matrix, each row corresponds to a mixture mean;
#        P: K*1 matrix, each entry corresponds to the weight for a mixture;
#        Var: K*1 matrix, each entry corresponds to the variance for a mixture;
# output: Mu: updated Mu, K*d matrix, each row corresponds to a mixture mean;
#         P: updated P, K*1 matrix, each entry corresponds to the weight for a mixture;
#         Var: updated Var, K*1 matrix, each entry corresponds to the variance for a mixture;
#         post: updated post, n*K matrix, each row corresponds to the soft counts for all mixtures for an example
#         LL: Numpy array for Loglikelihood values

def mixGauss(X,K,Mu,P,Var):
    n,d = np.shape(X) # n data points of dimension d
    post = np.zeros((n,K)) # posterior probs tbd

    LL = []

    post, new_LL = Estep(X, K, Mu, P, Var)
    Mu, P, Var = Mstep(X, K, Mu, P, Var, post)
    LL.append(new_LL)

    old_LL = False
    i = 0
    while np.abs(new_LL - old_LL) > 1e-6*np.abs(new_LL):
        old_LL = new_LL
        post, new_LL = Estep(X, K, Mu, P, Var)
        Mu, P, Var = Mstep(X, K, Mu, P, Var, post)
        LL.append(new_LL)



    return (Mu, np.squeeze(P), Var, post, LL)


# fill incomplete Matrix
# input: X: n*d incomplete data matrix;
#        K: number of mixtures;
#        Mu: K*d matrix, each row corresponds to a mixture mean;
#        P: K*1 matrix, each entry corresponds to the weight for a mixture;
#        Var: K*1 matrix, each entry corresponds to the variance for a mixture;
# output: Xnew: n*d data matrix with unrevealed entries filled
def fillMatrix(X,K,Mu,P,Var):
    n,d = np.shape(X)
    Xnew = np.copy(X)

    #Write your code here

    return Xnew

# Bayesian Information Criterion (BIC) for selecting the number of mixture components
# input:  n*d data matrix X, a list of K's to try
# output: the highest scoring choice of K
def BICmix(X,Kset):
    n, d = np.shape(X)
    BICs = np.empty(len(Kset))
    Ks = np.empty(len(Kset))

    for i, K in enumerate(Kset):
        Mu, P, Var = init(X, K)
        _, _, _, _, LL = mixGauss(X, K, Mu, P, Var)
        BICs[i] = LL - (1.0/2 * (K*d + 2*K - 1) * np.log(n))
        Ks[i] = K

    K = Ks[np.argmax(BICs)]

    return K
