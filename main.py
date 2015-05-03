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

    for K in [4]:#[1,2,3,4]:
        Mu, P, Var = init(X, K)
        Mu, P, Var, post, LL = mixGauss(X, K, Mu, P, Var)

        for i in range(30):

            Mu_new, P_new, Var_new = init(X, K)
            Mu_new, P_new, Var_new, post_new, LL_new = mixGauss(X, K,
                                                                Mu_new, P_new, Var_new)
            if LL_new[-1] > LL[-1]:
                Mu, P, Var, post, LL = Mu_new, P_new, Var_new, post_new, LL_new
        print LL[-1]
        plot2D(X, K, Mu, P, Var, post, "mixGauss with K=" + str(K))

def one_e():
    X = readData('toy_data.txt')

    K, BIC_score = BICmix(X, Kset=[1,2,3,4])
    print K, BIC_score


if __name__ == "__main__":
    #one_a()
    #test_EM()
    #one_d()
    #one_e()
    pass
