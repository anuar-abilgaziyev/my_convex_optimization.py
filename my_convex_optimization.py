import matplotlib.pyplot as plt
import numpy as np

"""Introduction"""
x = np.arange(-10, 10, 0.001)
f = lambda x : (x - 1)**4 + x**2

plt.plot(x,f(x))
plt.show()


"""Bisection method"""
print("Bisection method")
def find_root(f, a, b):
    c = a
    while (b-a) >= 0.001:
        c = (a + b)/2
        if f(c) < 0.001 and f(c) > -0.001:
            break
        else:
            if f(a)*f(c) < 0:
                b = c
            else:
                a = c
    return c

f_prime = lambda x : 4*(x - 1)**3 + 2*x
    
print(find_root(f_prime, -10, 10)) # x value
print(f(find_root(f_prime, -10, 10))) # y value

"""Brendt's method"""
print("Brendt's method")
from scipy.optimize import minimize_scalar

res = minimize_scalar(f, method='brent')
print('x_min: %.02f, f(x_min): %.02f' % (res.x, res.fun))

# plot curve
x = np.linspace(res.x - 1, res.x + 1, 100)
y = [f(val) for val in x]
plt.plot(x, y, color='blue', label='f')

# plot optima
plt.scatter(res.x, res.fun, color='red', marker='x', label='Minimum')

plt.grid()
plt.legend(loc = 1)

"""Gradient Descent"""
print("Gradient Descent")
def gradient_descent(f, f_prime, start, learning_rate = 0.1):
    x = start
    while f_prime(x) > 0.001 or f_prime(x) < -0.001:
        x = x - learning_rate*f_prime(x)
    return x

f = lambda x : (x - 1) ** 4 + x ** 2
f_prime = lambda x : 4*((x-1)**3) + 2*x
start = -1
x_min = gradient_descent(f, f_prime, start, 0.01)
f_min = f(x_min)

print("x_min: %0.2f, f(x_min): %0.2f" % (x_min, f_min))

"""Simplex method"""
print("Simplex method")
from scipy.optimize import linprog
def solve_linear_problem(A, b, c):
    x0_bounds = (0, None)
    x1_bounds = (0, None)
    res = linprog(c, A_ub=A, b_ub=b, bounds=[x0_bounds, x1_bounds], method = 'simplex')
    opt_val = res.fun*-1
    opt_x = []
    for x in res.x:
        opt_x.append(round(x))
    return opt_val, opt_x
    
A = np.array([[2, 1], [-4, 5], [1, -2]])
b = np.array([10, 8, 3])
c = np.array([-1, -2])

optimal_value, optimal_arg = solve_linear_problem(A, b, c)

print("The optimal value is: ", optimal_value, " and is reached for x = ", optimal_arg)

"""Simplex algorithm from scratch"""
print("Simplex algorithm from scratch")
def solve_linear_problem(A, b, c):
    matrix = A
    var_number = np.size(A, 1)
    matrix = np.vstack([matrix, c])
    i = np.identity(len(matrix))
    matrix = np.append(matrix, i, axis=1)
    b = np.append(b, 0)
    b = np.transpose([b])
    matrix = np.append(matrix, b, axis=1)
    last_row = matrix[-1]
    
    while (all(i >= 0 for i in matrix[-1])) == False:
        pivot_col = np.argmin(last_row)
        check_array = []
        for row in matrix[:-1]:
            indicator = row[-1]/row[pivot_col]
            check_array.append(indicator)

        min_val = 1000
        pivot_row = 0
        for idx, val in enumerate(check_array):
            if min_val > val and val >= 0:
                min_val = val
                pivot_row = idx

        matrix[pivot_row] = np.divide(matrix[pivot_row,:], matrix[pivot_row, pivot_col])
        for idx, row in enumerate(matrix):
            if idx != pivot_row:
                matrix[idx] = np.subtract(row, np.multiply(matrix[pivot_row,:], np.divide(row[pivot_col], matrix[pivot_row, pivot_col])))
           
    opt_x = []
    for i_col in  range(var_number):
        for i_row in range(np.size(A, 0)):
            if matrix[i_row, i_col] == 1:
                opt_x.append(round(matrix[i_row, -1]))
    opt_val = matrix[-1,-1]
    return round(opt_val), opt_x
    
A = np.array([[2, 1], [-4, 5], [1, -2]])
b = np.array([10, 8, 3])
c = np.array([-1, -2])

solve_linear_problem(A, b, c)
optimal_value, optimal_arg = solve_linear_problem(A, b, c)

print("The optimal value by Simplex from scratch is: ", optimal_value, " and is reached for x = ", optimal_arg)