from project3_student import *

def warm_up():
    X = readData('toy_data.txt')
    for K in [3]:# [1,2,3,4]:
        Mu, P, Var = init(X, K)
        Mu, P, Var, post = kMeans(X, K, Mu, P, Var)
        print Mu.shape, P.shape, Var.shape, post.shape

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


if __name__ == "__main__":
    #warm_up()
    test_EM()
    pass
