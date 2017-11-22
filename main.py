import numpy as np
from scipy import optimize
from sympy import *

def find_dk(curr_fx, curr_hx, curr_gx, curr_grad_fx, curr_grad_hx, curr_grad_gx, H):
    a = H[0][0]/2 + (H[1][1]*(curr_grad_hx[0]/curr_grad_hx[1])**2)/2 - (curr_grad_hx[0]/curr_grad_hx[1])*(H[0][1]+H[1][0])
    b = -1*(curr_grad_fx[1]*curr_grad_hx[0])/curr_grad_hx[1] + (H[1][1]*curr_hx*curr_grad_hx[0])/(curr_grad_hx[1]**2) + curr_grad_fx[0] - (curr_hx/curr_grad_hx[1])*(H[0][1]+H[1][0])
    c = -1*(curr_hx*curr_grad_fx[1])/curr_grad_hx[1] + (H[1][1]*(curr_hx)**2)/(curr_grad_hx[1]**2)
    constraint_d1 = -1*(curr_gx*curr_grad_hx[1] - curr_hx*curr_grad_gx[1])/(curr_grad_gx[0]*curr_grad_hx[1] - curr_grad_gx[1]*curr_grad_hx[0])
    global_min = (-1*b)/(2*a)
    if(constraint_d1 > global_min):
        dk = constraint_d1
    else:
        dk = global_min
    return np.array([dk, (-curr_hx - curr_grad_hx[0]*dk)/curr_grad_hx[1]])

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
    call_P = lambda x: P_alpha.subs([('alpha',x)])
    alpha_k = optimize.brent(call_P)
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

def constrained_variable_metric_method(fx, hx, gx, x_0, H_0, xvars):
    d1, d2 = symbols('d1 d2')
    dvars = [d1, d2]

    grad_fx = np.array([ diff(fx, x) for x in xvars ])
    grad_hx = np.array([ diff(hx, x) for x in xvars ])
    grad_gx = np.array([ diff(gx, x) for x in xvars ])

    x_k_1 = x_0
    H_k_1 = H_0

    mu_k_1 = 0
    sigma_k_1 = 0

    for k in range(1,5):

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

        print('Iteration: '+str(k)+'   '+str(x_k))

        x_k_1 = x_k
        mu_k_1 = mu_k
        sigma_k_1 = sigma_k






x1, x2 = symbols('x1 x2')
xvars = [x1, x2]

fx = 6*x1*(x2**-1) + x2*(x1**-2)
hx = x1*x2 - 2
gx = x1 + x2 -1

x_0 = np.array([2.0,1.0])
H_0 = np.eye(2)


constrained_variable_metric_method(fx,hx,gx,x_0,H_0,xvars)

# print('f(x): '+str(fx))
# print('h(x): '+str(hx)+'=0')
# print('g(x): '+str(gx)+'<=0')

# d_k = np.array([-4,2])
#
# x_k_1=np.array([2,1])
#
#
#
# x_k = np.array([1.46051, 1.26974])
