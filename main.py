import numpy as np
from scipy import optimize
from sympy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

x_values = []

def list_to_array(x):
    return np.array(x, dtype = np.float64).reshape(2, 1)

def calculate_function_value(fx, xvars, xcurr):
    return fx.subs(zip(xvars,xcurr))

def find_dk(curr_fx, curr_hx, curr_gx, curr_grad_fx, curr_grad_hx, curr_grad_gx, H):
    curr_grad_fx = list_to_array(curr_grad_fx)
    curr_grad_gx = list_to_array(curr_grad_gx)
    curr_grad_hx = list_to_array(curr_grad_hx)

    new_hx = lambda d : curr_hx + np.matmul(np.transpose(curr_grad_hx), list_to_array(d))
    new_gx = lambda d : curr_gx + np.matmul(np.transpose(curr_grad_gx), list_to_array(d))
    objective = lambda d :  np.matmul(np.transpose(curr_grad_fx), list_to_array(d)) + (np.matmul(np.transpose(list_to_array(d)), np.matmul(H, list_to_array(d))))/2

    constraints = ({'type': 'eq', 'fun': new_hx},
                   {'type': 'ineq', 'fun': new_gx})

    result = optimize.minimize(objective, [1.0, 1.0], constraints = constraints)
    return result.x

def find_lagrange_multipliers(curr_fx, curr_hx, curr_gx, curr_grad_fx, curr_grad_hx, curr_grad_gx, H, d):
    A = np.array( ( [ curr_grad_hx[0], curr_grad_gx[0] ], [ curr_grad_hx[1], curr_grad_gx[1]] ), dtype = np.float64)
    B = np.array( [ curr_grad_fx[0] + H[0][0]*d[0] + (H[0][1] + H[1][0])*d[1] , curr_grad_fx[1] + H[1][1]*d[1] + (H[0][1] + H[1][0])*d[0] ], dtype = np.float64 ).reshape(2,1)
    A_inverse = np.linalg.inv(A)
    X = np.matmul(A_inverse,B)
    return X[0][0], X[1][0]

def calculate_mu_sigma(k, u, v, mu_k_1, sigma_k_1):
    if k==1:
        mu_k = abs(v)
        sigma_k = abs(u)
    else:
        mu_k = max(abs(v),(mu_k_1+abs(v))/2)
        sigma_k = max(abs(u), (sigma_k_1+abs(u))/2)
    return mu_k, sigma_k


def minimize_alpha_through_penalty_function(fx, hx, gx, mu_k, sigma_k, x_k_1, d_k):
    alpha = symbols('alpha')
    Px = fx + mu_k*abs(hx) - sigma_k*Min(0, gx)
    P_alpha = Px.subs([(x1,x_k_1[0]+alpha*d_k[0]), (x2,x_k_1[1]+alpha*d_k[1])])
    call_P = lambda alpha : P_alpha.subs([('alpha',alpha)])
    constraints = ({'type': 'ineq', 'fun': lambda alpha: alpha},
                   {'type': 'ineq', 'fun': lambda alpha: 1-alpha})
    alpha_k = optimize.minimize(call_P, 0, constraints = constraints).x
    return alpha_k

def calculate_y(grad_L, x_k, x_k_1):
    return np.array([i.subs([(x1,x_k[0]),(x2,x_k[1])]) for i in grad_L]) - np.array([i.subs([(x1,x_k_1[0]),(x2,x_k_1[1])]) for i in grad_L])

def calculate_theta(z, y, H):
    z = z.reshape(2,1).astype(np.float64)
    y = y.reshape(2,1).astype(np.float64)
    a1 = np.matmul(np.transpose(z),y)
    a2 = 0.2*np.matmul(np.transpose(z), np.matmul(H,z))
    if(a1>=a2):
        return np.array([[1]])
    else:
        return (0.8*np.matmul(np.transpose(z), np.matmul(H,z)))/(np.matmul(np.transpose(z), np.matmul(H,z)) - np.matmul(np.transpose(z),y))

def calculate_w(theta, H, z, y):
    z = z.reshape(2,1).astype(np.float64)
    y = y.reshape(2,1).astype(np.float64)
    theta = theta[0][0]
    return theta*y + (1-theta)*np.matmul(H,z)

def updateH(H, z, w):
    z = z.reshape(2,1).astype(np.float64)
    w = w.reshape(2,1).astype(np.float64)
    a1 = np.matmul(H , np.matmul(z, np.matmul(np.transpose(z) , H ))) / np.matmul(np.transpose(z) , np.matmul(H, z) )
    a2 = np.matmul(w, np.transpose(w)) / np.matmul(np.transpose(z), w)
    return H - a1 + a2

def constrained_variable_metric_method(fx, hx, gx, x_0, H_0, xvars, no_of_iterations):
    d1, d2 = symbols('d1 d2')
    dvars = [d1, d2]

    grad_fx = np.array([ diff(fx, x) for x in xvars ])
    grad_hx = np.array([ diff(hx, x) for x in xvars ])
    grad_gx = np.array([ diff(gx, x) for x in xvars ])

    x_k_1 = x_0
    H_k_1 = H_0

    mu_k_1 = 0
    sigma_k_1 = 0

    for k in range(1,no_of_iterations+1):

        xcurr = x_k_1
        H_k = H_k_1

        curr_fx = np.array([ fx.subs(zip(xvars,xcurr)) ])
        curr_hx = np.array([ hx.subs(zip(xvars,xcurr)) ])
        curr_gx = np.array([ gx.subs(zip(xvars,xcurr)) ])

        curr_grad_fx = np.array([ dfx.subs(zip(xvars,xcurr)) for dfx in grad_fx ])
        curr_grad_hx = np.array([ dhx.subs(zip(xvars,xcurr)) for dhx in grad_hx ])
        curr_grad_gx = np.array([ dgx.subs(zip(xvars,xcurr)) for dgx in grad_gx ])

        d_k = find_dk(curr_fx, curr_hx, curr_gx, curr_grad_fx, curr_grad_hx, curr_grad_gx, H_k)

        v, u = find_lagrange_multipliers(curr_fx, curr_hx, curr_gx, curr_grad_fx, curr_grad_hx, curr_grad_gx, H_k, d_k)

        mu_k, sigma_k = calculate_mu_sigma(k, u, v, mu_k_1, sigma_k_1)

        alpha_k = minimize_alpha_through_penalty_function(fx, hx, gx, mu_k, sigma_k, x_k_1, d_k)

        x_k = x_k_1 + alpha_k*d_k.reshape(2)

        z = x_k - x_k_1

        grad_L = grad_fx - v*grad_hx - u*grad_gx

        y = calculate_y(grad_L, x_k, x_k_1)

        theta = calculate_theta(z, y, H_k)

        w = calculate_w(theta, H_k, z, y)

        H_k_1 = updateH(H_k, z, w)

        print('Iteration: '+str(k)+' '+str(x_k)+' '+str(calculate_function_value(fx, xvars, x_k)))
        x_values.append(list(x_k))

        x_k_1 = x_k
        mu_k_1 = mu_k
        sigma_k_1 = sigma_k

    print('***************************************')
    print('Final x: '+str(x_k))
    print('f(x): '+str(calculate_function_value(fx, xvars, x_k)))
    print('***************************************')
    return




x1, x2 = symbols('x1 x2')
xvars = [x1, x2]

case = 1 # 1 or 2

if case == 1:
    fx = 6*x1*(x2**-1) + x2*(x1**-2)
    hx = x1*x2 - 2
    gx = x1 + x2 -1

    x_0 = np.array([2.0,1.0])
    H_0 = np.eye(2)

    num_iterations = 27
elif case == 2:
    fx = 3*x1**2 - 4*x2
    hx = 2*x1 + x2 -4
    gx = 37 - x1**2 - x2**2

    x_0 = np.array([50,50])
    H_0 = np.eye(2)

    num_iterations = 7

constrained_variable_metric_method(fx,hx,gx,x_0,H_0,xvars,num_iterations)

X_list1 = [i[0] for i in x_values]
Y_list1 = [i[1] for i in x_values]
print(X_list1)
print(Y_list1)
