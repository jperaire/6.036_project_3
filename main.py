from project3_student import *

def one_a():
    X = readData('toy_data.txt')
    for K in [1,2,3,4]:
        Mu, P, Var = init(X, K)
        Mu, P, Var, post, cost = kMeans(X, K, Mu, P, Var)

        for i in range(4):

            Mu_new, P_new, Var_new = init(X, K)
            Mu_new, P_new, Var_new, post_new, cost_new = kMeans(X, K,
                                                                Mu_new, P_new, Var_new)
            if cost_new < cost:
                Mu, P, Var, post, cost = Mu_new, P_new, Var_new, post_new, cost_new

        plot2D(X, K, Mu, P, Var, post, "kMeans with K=" + str(K))

def test_EM():
    # BUG in M step maybe? Mu converges for some reason

    X = readData('toy_data.txt')

    K=3

    Mu, P, Var = init(X, K, fixedmeans=True)
    Mu, P, Var, post, LL = mixGauss(X, K, Mu, P, Var)
    print LL[0]
    print LL[-1]
    plot2D(X, K, Mu, P, Var, post, "EM with K=" + str(K))

def one_d():

    X = readData('toy_data.txt')

    for K in [3]:#[1,2,3,4]:
        Mu, P, Var = init(X, K, fixedmeans=True)
        Mu, P, Var, post, LL = mixGauss(X, K, Mu, P, Var)

        # for i in range(30):
        #
        #     Mu_new, P_new, Var_new = init(X, K)
        #     Mu_new, P_new, Var_new, post_new, LL_new = mixGauss(X, K,
        #                                                         Mu_new, P_new, Var_new)
        #     if LL_new[-1] > LL[-1]:
        #         Mu, P, Var, post, LL = Mu_new, P_new, Var_new, post_new, LL_new
        print LL[0], LL[-1]
        #plot2D(X, K, Mu, P, Var, post, "mixGauss with K=" + str(K))

def one_e():
    X = readData('toy_data.txt')

    K, BIC_score = BICmix(X, Kset=[1,2,3,4])
    print K, BIC_score

def two_d():
    X = readData('netflix_incomplete.txt')

    K=12

    Mu, P, Var = init(X, K)
    Mu, P, Var, post, LL = mixGauss(X, K, Mu, P, Var)
    print LL[-1]
    for i in range(4):

        Mu_new, P_new, Var_new = init(X, K)
        Mu_new, P_new, Var_new, post_new, LL_new = mixGauss(X, K,
                                                            Mu_new, P_new, Var_new)
        print LL_new[-1]
        if LL_new[-1] > LL[-1]:
            Mu, P, Var, post, LL = Mu_new, P_new, Var_new, post_new, LL_new
    print "Log likely hood of best mixture (initial, final): ", LL[0], LL[-1]


def two_f():
    X = readData('netflix_incomplete.txt')

    K = 12

    Mu, P, Var = init(X, K)
    Mu, P, Var, post, LL = mixGauss(X, K, Mu, P, Var)

    for i in range(5):

        Mu_new, P_new, Var_new = init(X, K)
        Mu_new, P_new, Var_new, post_new, LL_new = mixGauss(X, K,
                                                            Mu_new, P_new, Var_new)
        if LL_new[-1] > LL[-1]:
            Mu, P, Var, post, LL = Mu_new, P_new, Var_new, post_new, LL_new

    print "LL: ", LL
    Xpred = fillMatrix(X, K, Mu, P, Var)
    Xc = readData('netflix_complete.txt')

    print "RMSE: ", rmse(Xpred, Xc)

if __name__ == "__main__":
    #one_a()
    #test_EM()
    #one_d()
    #one_e()
    two_d()
    #two_f()
    pass

# PART ONE CODE
# def Estep(X, K, Mu, P, Var):
#     n, d = np.shape(X)
#     post = np.zeros((n, K))
#     LL = 0
#
#     X = X.copy()
#     Mu = Mu.copy()
#     P = P.copy()
#
#     for i in xrange(n):
#         stored_values = np.zeros(K)
#         for j in xrange(K):
#             exp_term = -1.0/(2*Var[j])*(dist(X[i]-Mu[j])**2)
#             stored_values[j] = P[j] * (1.0/(2*np.pi*Var[j])**(d/2.0))*np.exp(exp_term)
#         LL += np.log(stored_values.sum())
#         post[i,:] = stored_values/stored_values.sum()
#
# #Alternative method - DOESNT WORK YET
#
#     # X_copies = np.expand_dims(X.copy(), 0) #(1, n, d)
#     # Mu_copies = np.expand_dims(Mu.copy(), 1) #(K, 1, d)
#     #
#     # distances = dist(X_copies - Mu_copies, axis=2) #(K, n)
#     #
#     # std_devs = np.sqrt(Var)
#     #
#     # probabilities = (1.0/(2*np.pi*Var)**(d/2.0)) * norm.pdf(distances, scale=std_devs) #(K, n)
#     #
#     # weighted_probs = P * probabilities #(K, n)
#     #
#     # post_copy = (weighted_probs.T/np.expand_dims(weighted_probs.sum(axis=0), 1)) #(n, K)
#     # assert np.allclose(post, post_copy)
#
#     return (post, LL)
#
# # def Estep(X,K,Mu,P,Var):
# #     n,d = np.shape(X) # n data points of dimension d
# #     post = np.zeros((n,K)) # posterior probabilities tbd
# #
# #     X_copies = np.expand_dims(X.copy(), 0) #(1, n, d)
# #     Mu_copies = np.expand_dims(Mu.copy(), 1) #(K, 1, d)
# #
# #     distances = dist(X_copies - Mu_copies, axis=2) #(K, n)
# #
# #     std_devs = np.sqrt(Var)
# #
# #     probabilities = (1.0/(2*np.pi*Var)**(d/2.0)) * norm.pdf(distances, scale=std_devs) #(K, n)
# #
# #     weighted_probs = P * probabilities #(K, n)
# #
# #     post = (weighted_probs.T/np.expand_dims(weighted_probs.sum(axis=0), 1)) #(n, K)
# #
# #     LL = (post * np.log(weighted_probs).T).sum()
# #     LL = (np.log(weighted_probs.sum(axis=0))).sum()
# #
# #     return (post, LL)
#
#
#
# # M step of EM algorithm
# # input: X: n*d data matrix;
# #        K: number of mixtures;
# #        Mu: K*d matrix, each row corresponds to a mixture mean;
# #        P: K*1 matrix, each entry corresponds to the weight for a mixture;
# #        Var: K*1 matrix, each entry corresponds to the variance for a mixture;
# #        post: n*K matrix, each row corresponds to the soft counts for all mixtures for an example
# # output:Mu: updated Mu, K*d matrix, each row corresponds to a mixture mean;
# #        P: updated P, K*1 matrix, each entry corresponds to the weight for a mixture;
# #        Var: updated Var, K*1 matrix, each entry corresponds to the variance for a mixture;
#
# def Mstep(X, K, Mu, P, Var, post):
#     n, d = np.shape(X)
#
#     n_hats = post.sum(axis=0)
#     P = n_hats/float(n)
#
#     # Mu = np.empty((K, d))
#     # for k in xrange(K):
#     #     Mu[k,:] = (1.0/n_hats[k]) * (np.expand_dims(post[:, k], 1) * X).sum(axis=0)
#
#     weighted_mean_of_points = (np.expand_dims(X, 1) * np.expand_dims(post, 2)).sum(axis=0) #(K, d)
#     Mu = 1.0/np.expand_dims(n_hats, 1) * weighted_mean_of_points
#
#     Var = np.empty((K, 1))
#     for k in xrange(K):
#         Var[k, :] = (1.0/(d*n_hats[k])) * (post[:, k] * dist(X-Mu[k, :], axis=1)**2).sum()
#
#     # DOESNT WORK YET
#     # displacement_vectors = (np.expand_dims(X, 1) - np.expand_dims(Mu, 0))
#     # mss = dist(displacement_vectors, axis=2)**2
#     # weighted_mss = (mss * post).sum(axis=0)
#     # Var_hat = 1.0/(d * n_hats) * weighted_mss
#     # assert np.allclose(Var_hat, Var)
#     #     Var = np.expand_dims(Var, 1)
#
#     P = np.expand_dims(P, 1)
#     return (Mu, P, Var)
#
# # def Mstep(X,K,Mu,P,Var,post):
# #     n,d = np.shape(X) # n data points of dimension d
# #
# #     n_hat = post.sum(axis=0)
# #     P = n_hat/float(n)
# #
# #     weighted_mean_of_points = (np.expand_dims(X, 1) * np.expand_dims(post, 2)).sum(axis=0) #(K, d)
# #
# #     Mu = 1.0/np.expand_dims(n_hat, 1) * weighted_mean_of_points
# #
# #
# #     mean_squared_spread = np.sqrt((np.expand_dims(X, 1) - np.expand_dims(Mu, 0))**2).sum(axis=2) #(n, K)
# #     weighted_mss = (mean_squared_spread * post).sum(axis=0)
# #
# #     Var = 1.0/(d * n_hat) * weighted_mss
# #
# #     P = np.expand_dims(P, 1)
# #     Var = np.expand_dims(Var, 1)
# #
# #     return (Mu,P,Var)
#
# # ----------------------------------------------------------------------------------------------------
# # mixture of Gaussians
# # input: X: n*d data matrix;
# #        K: number of mixtures;
# #        Mu: K*d matrix, each row corresponds to a mixture mean;
# #        P: K*1 matrix, each entry corresponds to the weight for a mixture;
# #        Var: K*1 matrix, each entry corresponds to the variance for a mixture;
# # output: Mu: updated Mu, K*d matrix, each row corresponds to a mixture mean;
# #         P: updated P, K*1 matrix, each entry corresponds to the weight for a mixture;
# #         Var: updated Var, K*1 matrix, each entry corresponds to the variance for a mixture;
# #         post: updated post, n*K matrix, each row corresponds to the soft counts for all mixtures for an example
# #         LL: Numpy array for Loglikelihood values
#
# def mixGauss(X,K,Mu,P,Var):
#     n,d = np.shape(X) # n data points of dimension d
#     post = np.zeros((n,K)) # posterior probs tbd
#
#     LL = []
#
#     post, new_LL = Estep(X, K, Mu, P, Var)
#     Mu, P, Var = Mstep(X, K, Mu, P, Var, post)
#     LL.append(new_LL)
#
#     old_LL = False
#     i = 0
#     while np.abs(new_LL - old_LL) > 1e-6*np.abs(new_LL):
#         old_LL = new_LL
#         post, new_LL = Estep(X, K, Mu, P, Var)
#         Mu, P, Var = Mstep(X, K, Mu, P, Var, post)
#         LL.append(new_LL)
#
#     return (Mu, P, Var, post, LL)
