from project3_student import *

def warm_up():
    X = readData('toy_data.txt')
    for K in [1,2,3,4]:
        Mu, P, Var = init(X, K)
        Mu, P, Var, post = kMeans(X, K, Mu, P, Var)
        plot2D(X, K, Mu, P, Var, post, "kMeans with K=" + str(K))

def test_EM():
    X = readData('toy_data.txt')

    K=3

    Mu, P, Var = init(X, K, fixedmeans=True)
    Mu, P, Var, post, LL = mixGauss(X, K, Mu, P, Var)
    print LL
    plot2D(X, K, Mu, P, Var, post, "EM with K=" + str(K))


if __name__ == "__main__":
    test_EM()
    pass
