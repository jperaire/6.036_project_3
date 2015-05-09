#this file contains major functions
import random as ra
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from numpy import linalg as LA
from scipy.stats import norm


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
    return (Mu,P,Var,post,curCost)

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
def Estep(X,K,Mu,P,Var):
    n,d = np.shape(X) # n data points of dimension d
    post = np.zeros((n,K)) # posterior probabilities to tbd

    LL = 0
    X_copies = np.expand_dims(X.copy(), 0) #(1, n, d)
    Mu_copies = np.expand_dims(Mu.copy(), 1) #(K, 1, d)

    X = X.copy()
    Mu = Mu.copy()
    P = P.copy()

    X_mask = X != 0

    for i in xrange(n):
        stored_values = np.zeros(K)
        for j in xrange(K):
            mask = np.argwhere(X[i] != 0)
            distance = dist((X[i]-Mu[j])[mask])
            exp_term = (-1.0/(2*Var[j]))*(distance**2)
            stored_values[j] = np.log(P[j]) - \
                                (X_mask[i, :].sum()/2.0)*np.log(2*np.pi*Var[j]) + \
                                exp_term

        m = stored_values.max()
        prob_diff = np.exp(stored_values - m)

        denom = prob_diff.sum()
        post[i,:] = prob_diff/denom
        LL += np.log(denom) + m


    return (post, LL)


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

def Mstep(X,K,Mu,P,Var,post):
    n,d = np.shape(X) # n data points of dimension d

    n_hats = post.sum(axis=0) #(K)
    P = n_hats/float(n)

    Mu_hat = np.empty((K, d))

    delta = X != 0
    post_mask = np.expand_dims(post, 2)
    mod_n_hats = (np.expand_dims(delta, 1) * post_mask).sum(axis=0)
    enough_data = mod_n_hats >= 1 #(K, d)

    numerator = (np.expand_dims(post, 2) * np.expand_dims(delta * X, 1)).sum(axis=0)

    Mu_hat = numerator/mod_n_hats
    Mu_hat = np.where(enough_data, Mu_hat, Mu)

    Var_init = np.empty(K)
    for k in xrange(K):
        distances = np.empty(n)
        for v in xrange(n):
            distances[v] = dist((X[v, :]-Mu_hat[k, :])[delta[v, :]], axis=0)

        Var_init[k] =  (post[:, k] * distances**2).sum()

    Var_modifier = 1.0/(np.expand_dims((X != 0).sum(axis=1), 1) * post).sum(axis=0)
    Var = np.where((Var_modifier * Var_init)>0.25, Var_modifier*Var_init, 0.25)
    Var = np.expand_dims(Var, 1)




    P = np.expand_dims(P, 1)
    return (Mu_hat, P, Var)




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
    while np.abs(new_LL - old_LL) > 1e-6*np.abs(new_LL):
        old_LL = new_LL
        post, new_LL = Estep(X, K, Mu, P, Var)
        Mu, P, Var = Mstep(X, K, Mu, P, Var, post)
        LL.append(new_LL)

    return (Mu, P, Var, post, LL)


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

    delta = X != 0
    post, LL = Estep(X, K, Mu, P, Var)
    predictions = np.dot(post, Mu)

    Xnew = np.where(delta, X, predictions)

    return Xnew

# Bayesian Information Criterion (BIC) for selecting the number of mixture components
# input:  n*d data matrix X, a list of K's to try
# output: the highest scoring choice of K
#         BIC_score: The BIC score of the highest choice of K
def BICmix(X, Kset):
    n, d = np.shape(X)
    BICs = np.empty(len(Kset))
    Ks = np.empty(len(Kset))

    for i, K in enumerate(Kset):
        Mu, P, Var = init(X, K)
        _, _, _, _, LL = mixGauss(X, K, Mu, P, Var)
        BICs[i] = LL[-1] - (1.0/2 * (K*d + 2*K - 1) * np.log(n))

        for _ in xrange(5):
            Mu, P, Var = init(X, K)
            _, _, _, _, LL = mixGauss(X, K, Mu, P, Var)
            BICs[i] = max(LL[-1] - (1.0/2 * (K*d + 2*K - 1) * np.log(n)), BICs[i])
        Ks[i] = K

    K = Ks[np.argmax(BICs)]
    BIC_score = BICs.max()
    return K, BIC_score
