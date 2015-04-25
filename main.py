from project3_student import *

def warm_up():
    print "in warm_up"
    X = readData('toy_data.txt')
    for K in [1,2,3,4]:
        Mu, P, Var = init(X, K)
        Mu, P, Var, post = kmeans(X, K, Mu, P, Var)
        plot2D(X, K, Mu, P, Var, "Label", "K-Means with %d" % K)
if __name__ == "__main__":
    warm_up()
    pass
