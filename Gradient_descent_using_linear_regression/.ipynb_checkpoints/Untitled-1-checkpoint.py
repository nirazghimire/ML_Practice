
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import math


def gradient_descent(x,y):
    m_curr,b_curr =0,0
    n = len(x)
    iterations = 1000000
    cost_previous = 0
    learning_rate = 0.0002
    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        cost = (1/n) * sum(val**2 for val in (y- y_predicted))

        md = (-2/n) * sum(x *(y - y_predicted))
        bd =  (-2/n) * sum(y - y_predicted)

        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd

        if math.isclose(cost,cost_previous, rel_tol=1e-20):
            break
        cost_previous = cost


        print(f'm is {m_curr}, b is {b_curr}, cost is {cost}, iterations is {i}')

        return m_curr, b_curr




def predicted_from_sklearn():
    df = pd.read_csv("test_scores.csv")
    reg = LinearRegression()
    reg.fit(df[['math']],df['cs'])

    return reg.coef_, reg.intercept_


def main():
    df = pd.read_csv("test_scores.csv")
    m_from_orig, b_from_orig = gradient_descent(np.array(df['math']),np.array(df['cs']))
    m_from_model, b_from_model =  predicted_from_sklearn()

    print(f'm from original is : {m_from_orig} and m from model is: {m_from_model}')
    print(f'b from original is : {b_from_orig} and b from model is: {b_from_model}')



if __name__ == "__main__":
    main()



