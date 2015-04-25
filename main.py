from project3_student import *

def warm_up():
    X = readData('toy_data.txt')
    for K in [1,2,3,4]:
        Mu, P, Var = init(X, K)
        Mu, P, Var, post = kMeans(X, K, Mu, P, Var)
        plot2D(X, K, Mu, P, Var, post, "kMeans with K=" + str(K))

if __name__ == "__main__":
    warm_up()
    pass
