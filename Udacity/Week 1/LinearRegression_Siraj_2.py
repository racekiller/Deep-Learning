import numpy as np
import matplotlib.pyplot as plt

# y = mx + b
# m is slope, b is y-intercept
def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))

def step_gradient(b_current, m_current, points, learningRate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b, new_m]

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    b_list = []
    m_list = []
    error_list = []
    for i in range(num_iterations):
        b, m = step_gradient(b, m, array(points), learning_rate)
        b_list.append(b)
        m_list.append(m)
        error_list.append(compute_error_for_line_given_points(b,m,points))
    return [b, m,b_list,m_list,error_list]

def run():
    points = genfromtxt("data.csv", delimiter=",")
    learning_rate = 0.0001
    initial_b = 0 # initial y-intercept guess
    initial_m = 0 # initial slope guess
    num_iterations = 100
    x2 = points[:,0]
    y2 = points[:,1]
    print ("Starting gradient descent at b = {0}, m = {1}, error = {2}".format(initial_b, initial_m, compute_error_for_line_given_points(initial_b, initial_m, points)))
    print ("Running...")
    [b, m, b_list, m_list, error_list] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print ("After {0} iterations b = {1}, m = {2}, error = {3}".format(num_iterations, b, m, compute_error_for_line_given_points(b, m, points)))
    b_array = np.array(b_list)
    m_array = np.array(m_list)
    error_array = np.array(error_list)
    # print (b_array,m_array, error_array)

    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.plot(b_array,m_array, label="b vs m")
    ax.legend()

    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.plot(range(0,num_iterations), error_array, label="Error Function")
    ax.legend()

    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    plt.scatter(x2, y2)
    plt.plot(x2, m*x2 + b)

if __name__ == '__main__':
    run()
