# %% 
import numpy as np
from casadi import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.interpolate import make_interp_spline
from scipy.interpolate import make_interp_spline, BSpline
import seaborn as sns

#Plotting settings:
sns.set()
sns.set_style("whitegrid")


##Define the system/plant equations (subscript s) and model equations (subscript m)
Fa = 1.8275
Mt = 2105.2
# kinetic parameters
phi1 = - 3.
psi1 = -17.
phi2 = - 4.
psi2 = -29.
# Reference temperature
Tref = 110. + 273.15  # [=] K.
# Define vectors with names of states
# system:
states_s = ['x']
nd_s = len(states_s)
# model:
states_m = ['x']
nd_m = len(states_m)
# Define vectors with names of algebraic variables
# system:
algebraics_s = ['Xas', 'Xbs', 'Xcs', 'Xes', 'Xps', 'Xgs']
na_s = len(algebraics_s)
xa_s = SX.sym('xa_s', na_s)
for i in range(na_s):
    globals()[algebraics_s[i]] = xa_s[i]
# model:
algebraics_m= ['Xam', 'Xbm', 'Xem', 'Xpm', 'Xgm'] 
na_m = len(algebraics_m)
xa_m = SX.sym('xa_m', na_m)
for i in range(na_m):
    globals()[algebraics_m[i]] = xa_m[i]
# Define vectors with banes of input variables (same for system and model)
inputs = ['Fb', 'Tr']
nu = len(inputs)
u = SX.sym("u", nu)
for i in range(nu):
    globals()[inputs[i]] = u[i]
# Reparametrization system
k1s = 1.6599e6 * np.exp(-6666.7 / (Tr + 273.15))
k2s = 7.2117e8 * np.exp(-8333.3 / (Tr + 273.15))
k3s = 2.6745e12 * np.exp(-11111. / (Tr + 273.15))
# model:
k1m = np.exp(phi1) * np.exp((Tref / (Tr + 273.15) - 1) * psi1)
k2m = np.exp(phi2) * np.exp((Tref / (Tr + 273.15) - 1) * psi2)


# reaction rate
Fr = Fa + Fb
# system:
r1s = k1s * Xas * Xbs * Mt
r2s = k2s * Xbs * Xcs * Mt
r3s = k3s * Xcs * Xps * Mt
# model:
r1m = k1m * Xam * Xbm * Xbm * Mt
r2m = k2m * Xam * Xbm * Xpm * Mt
# Declare algebraic equations for the system
# system:
Aeqs = []
Aeqm = []
Aeqs = [(Fa - r1s - Fr * Xas) / Mt, (Fb - r1s - r2s - Fr * Xbs) / Mt , (+ 2 * r1s - 2 * r2s - r3s - Fr * Xcs) / Mt ,(+ 2 * r2s - Fr * Xes) / Mt , (+   r2s - 0.5 * r3s - Fr * Xps) / Mt ,(+ 1.5 * r3s - Fr * Xgs) / Mt]
Aeqm = [(Fa - r1m - r2m - Fr * Xam),(Fb - 2 * r1m - r2m - Fr * Xbm),(+ 2 * r1m - Fr * Xem),(+   r1m - r2m - Fr * Xpm),(+ 3 * r2m - Fr * Xgm)]
Fb = u[0]
Tr = u[1]
Fr = Fa + Fb
objs = -(1043.38 * xa_s[4] * Fr + 20.92 * xa_s[3] * Fr - 79.23 * Fa -118.34 * Fb)
objm = -(1043.38 * xa_m[4] * Fr + 20.92 * xa_m[3] * Fr - 79.23 * Fa -118.34 * Fb)
VVs = Function('vfcn', [xa_s, u], [vertcat(*Aeqs)], ['w0s', 'u'], ['ws']) # Root-finding function, implicitly defines x as a function of u
solvers = rootfinder('solver', 'newton', VVs)
VVm = Function('vfcn', [xa_m, u], [vertcat(*Aeqm)], ['w0m', 'u'], ['wm']) # Root-finding function, implicitly defines x as a function of u
solver = rootfinder('solver', 'newton', VVm)
#Initial point
u0 = np.array([7,70])
#sensitivities for the model
# make the NLP
# Start with an empty NLP
ws=[] # decision variables
w0s = []
wm=[] # decision variables
w0m = []
lbws = []
ubws = []
lbwm = []
ubwm = []

ws2 = []
w0s2 = []
lbws2 = []
ubws2 = []
lbgs2 = []
ubgs2 = []
gs2 = []

Js = 0
gs=[]
lbgs = []
ubgs = []
Jm = 0
gm =[]
lbgm = []
ubgm = []
gm_grad = []
lbgm_grad = []
ubgm_grad = []
wm_grad = []
w0m_grad = []



# states
ws.append(xa_s) #må ikke det legges til u som en del av desicion variables????
lbws.append([0, 0, 0, 0 ,0 ,0])
ubws.append([np.inf, np.inf,np.inf,np.inf,np.inf,np.inf])
w0s.append([0, 0, 0, 0 ,0 ,0])
wm.append(xa_m) #cannot invert??
wm.append(u)
lbwm.append([0, 0, 0, 0 ,0 ,4,70])
ubwm.append([np.inf, np.inf,np.inf,np.inf,np.inf,7,100])
w0m.append([0, 0, 0, 0 ,0,u0[0],u0[1]])

ws2.append(xa_s)
ws2.append(u)
lbws2.append([0, 0, 0, 0 ,0 ,0,4,70])
ubws2.append([np.inf, np.inf,np.inf,np.inf,np.inf,np.inf,7,100])
w0s2.append([0, 0, 0, 0 ,0 ,0,u0[0],u0[1]])

#ws2 = ws
#ws2.append(u)
#objective function
Js = -(1043.38 * xa_s[4] * (1.8275 + u[0]) +
                20.92 * xa_s[3] * (1.8275 + u[0]) -
                79.23 * 1.8275 -
                118.34 * u[0]) 
Jm = -(1043.38 * xa_m[3] * (1.8275 + u[0]) +
                20.92 * xa_m[2] * (1.8275 + u[0]) -
                79.23 * 1.8275 -
                118.34 * u[0]) 
#Jm_modified = Jm + eps + mod_grad
# Add equality constraint
Aineq = [xa_m[0]-0.12,xa_m[-1]-0.08]
Aineq2 = [xa_s[0]-0.12,xa_s[-1]-0.08]

grad_hack_1s = xa_s[0]-0.12
grad_hack_2s = xa_s[-1]-0.08
grad_hack_1m = xa_m[0]-0.12
grad_hack_2m = xa_m[-1]-0.08
#consts = Aeqs+Aineq
gs.append(vcat(Aeqs))
lbgs.append([0, 0, 0, 0 ,0 ,0])
ubgs.append([0, 0, 0, 0 ,0 ,0])

gs2.append(vcat(Aeqs))
gs2.append(vcat(Aineq2))
lbgs2.append([0, 0, 0, 0 ,0 ,0,-np.inf,-np.inf])
ubgs2.append([0, 0, 0, 0 ,0 ,0, 0, 0])

gm.append(vcat(Aeqm))
gm.append(vcat(Aineq))

lbgm.append([0, 0, 0, 0 ,0 , -np.inf,-np.inf])
ubgm.append([0, 0, 0, 0 ,0 , 0, 0])

gm_grad.append(vcat(Aeqm))
gm_grad.append(vcat(Aineq))
lbgm_grad.append([0, 0, 0, 0 ,0 , -np.inf,-np.inf])
ubgm_grad.append([0, 0, 0, 0 ,0 , 0, 0])

# Concatenate vectors
ws = vertcat(*ws)
ws2 = vertcat(*ws2)
gs = vertcat(*gs)
gs2 = vertcat(*gs2)
w0s = np.concatenate(w0s)
lbws = np.concatenate(lbws)
ubws = np.concatenate(ubws)
lbgs = np.concatenate(lbgs)
ubgs = np.concatenate(ubgs)
wm = vertcat(*wm)
gm = vertcat(*gm)
w0m = np.concatenate(w0m)
lbwm = np.concatenate(lbwm)
ubwm = np.concatenate(ubwm)
lbgm = np.concatenate(lbgm)
ubgm = np.concatenate(ubgm)

w0s2 = np.concatenate(w0s2)
lbws2 = np.concatenate(lbws2)
ubws2 = np.concatenate(ubws2)
lbgs2 = np.concatenate(lbgs2)
ubgs2 = np.concatenate(ubgs2)

#for the constraint gradients
lbwm_const=lbwm[0:5]
ubwm_const=ubwm[0:5]
lbgm_const=lbgm[0:5]
ubgm_const=ubgm[0:5]

wm_grad = vertcat(*wm_grad)
gm_grad = vertcat(*gm_grad)
#w0m_grad = np.concatenate(w0m_grad)
#lbm_grad = np.concatenate(lbm_grad)
#ubm_grad = np.concatenate(ubm_grad)
#lbgm_grad = np.concatenate(lbgm_grad)
#ubgm_grad = np.concatenate(ubgm_grad)
# Create an NLP solver
probs = {'f': Js, 'x': ws, 'g': gs,'p': u}
probs_grad_g_1 = {'f': grad_hack_1s, 'x': ws, 'g': gs,'p': u}
probs_grad_g_2 = {'f': grad_hack_2s, 'x': ws, 'g': gs,'p': u}

probs2 = {'f': Js, 'x': ws2, 'g': gs2}

probm = {'f': Jm, 'x': wm, 'g': gm}

wm2 = wm[0:5]
gm2 = gm[0:5]
probm2={'f': Jm, 'x': wm2, 'g': gm2,'p': u} #to find dJmdu
prob_grad = {'f': Jm, 'x': wm_grad, 'g': gm_grad,'p': u}
probm_grad_g_1 = {'f': grad_hack_1m, 'x': wm2, 'g': gm2,'p': u}
probm_grad_g_2 = {'f': grad_hack_2m, 'x': wm2, 'g': gm2,'p': u}
# Create an NLP solver, using ipopt
solvers = nlpsol('solver', 'ipopt', probs)
solverm = nlpsol('solver', 'ipopt', probm)
solverm2 = nlpsol('solver','ipopt',probm2)
solvers_grad_1 = nlpsol('solver', 'ipopt', probs_grad_g_1)
solvers_grad_2 = nlpsol('solver', 'ipopt', probs_grad_g_2)
solverm_grad_1 = nlpsol('solver', 'ipopt', probm_grad_g_1)
solverm_grad_2 = nlpsol('solver', 'ipopt', probm_grad_g_2)

solvers2 = nlpsol('solver', 'ipopt', probs2)

# Solve the NLP
solm = solverm(x0=w0m, lbx=lbwm, ubx=ubwm, lbg=lbgm, ubg=ubgm)
w0_m = solm['x'].full()
u_test = u0 # want to use this as starting point,

sols = solvers(x0=w0s, lbx=lbws, ubx=ubws, lbg=lbgs, ubg=ubgs,p=u_test) #make a sols2 where u_test is a part of w0s
sols2 = solvers2(x0=w0s2, lbx=lbws2, ubx=ubws2, lbg=lbgs2, ubg=ubgs2) #make a sols2 where u_test is a part of w0s

solm2 = solverm2(x0=w0m[0:5], lbx=lbwm[0:5], ubx=ubwm[0:5], lbg=lbgm[0:5], ubg=ubgm[0:5], p = u_test)
sols_grad_1 = solvers_grad_1(x0=sols['x'].full(), lbx=lbws, ubx=ubws, lbg=lbgs, ubg=ubgs,p=u_test)
sols_grad_2 = solvers_grad_2(x0=sols['x'].full(), lbx=lbws, ubx=ubws, lbg=lbgs, ubg=ubgs,p=u_test)
#define x0
x0gm = solm['x'].full()[0:5]
solm_grad_1 = solverm_grad_1(x0=x0gm, lbx=lbwm_const, ubx=ubwm_const, lbg=lbgm_const, ubg=ubgm_const,p=u_test)
solm_grad_2 = solverm_grad_2(x0=x0gm, lbx=lbwm_const, ubx=ubwm_const, lbg=lbgm_const, ubg=ubgm_const,p=u_test)

# quick check on the gradient
sols_grad_2_2= solvers_grad_2(x0=sols['x'].full(), lbx=lbws, ubx=ubws, lbg=lbgs, ubg=ubgs,p=[u_test[0],(u_test[1]+1e-6)] )

(sols_grad_2_2['f'].full()- sols_grad_2['f'].full())/(1e-6)

#FINITE DIFFERENCES
# for conveniance
def system_constraint(u_point):
    sols_grad_1 = solvers_grad_1(x0=w0s, lbx=lbws, ubx=ubws, lbg=lbgs, ubg=ubgs,p=u_point)
    sols_grad_2 = solvers_grad_2(x0=w0s, lbx=lbws, ubx=ubws, lbg=lbgs, ubg=ubgs,p=u_point)
    return np.array([sols_grad_1['f'].full()[0][0], sols_grad_2['f'].full()[0][0]])

def finite_dif_constraits(fun,u_point):
    centeral_point = fun(u_point)
    step_u1= fun(np.array([u_point[0]+1e-4,u_point[1]])) #add noise
    step_u2= fun(np.array([u_point[0],u_point[1]+1e-4]))
    gradient_u1 = -(step_u1-centeral_point)/(1e-4) # gradient of fun w.r.t u1
    gradient_u2 = -(step_u2-centeral_point)/(1e-4) # gradient of fun w.r.t u2
    # the minus signs are to match the gradient from lam_p!!
    
    return [np.array([[gradient_u1[0]],[gradient_u2[0]]]), np.array([[gradient_u1[1]],[gradient_u2[1]]])]

'''def finite_dif_constraits(fun,u_point,centeral_point):
    step_u1= fun(np.array([u_point[0]+1e-4,u_point[1]]))
    step_u2= fun(np.array([u_point[0],u_point[1]+1e-4]))
    gradient_u1 = -(step_u1-centeral_point)/(1e-4) # gradient of fun w.r.t u1
    gradient_u2 = -(step_u2-centeral_point)/(1e-4) # gradient of fun w.r.t u2
    # the minus signs are to match the gradient from lam_p!!
    
    return [np.array([[gradient_u1[0]],[gradient_u2[0]]]), np.array([[gradient_u1[1]],[gradient_u2[1]]])]'''

def system_costfunc(u_point):
    sols = solvers(x0=w0s, lbx=lbws, ubx=ubws, lbg=lbgs, ubg=ubgs,p=u_point)
    return np.array(sols['f'].full())

def finite_dif_cost(fun,u_point):
    centeral_point = fun(u_point)
    step_u1= fun(np.array([u_point[0]+1e-4,u_point[1]]))
    step_u2= fun(np.array([u_point[0],u_point[1]+1e-4]))
    gradient_u1 = -(step_u1-centeral_point)/(1e-4) # gradient of fun w.r.t u1
    gradient_u2 = -(step_u2-centeral_point)/(1e-4) # gradient of fun w.r.t u2
    # the minus signs are to match the gradient from lam_p!!
    
    return [np.array([[gradient_u1],[gradient_u2]])]

#EVALUATION
#cost functions
j_s = sols['f'].full()
j_m = solm['f'].full()
print("System cost function:", j_s)
print("Model cost function:", j_m)

#constraints
g1_s = sols_grad_1['f'].full()
g2_s = sols_grad_2['f'].full()
Gs = [g1_s,g2_s]
g1_m = solm_grad_1['f'].full()
g2_m = solm_grad_2['f'].full()
Gm = [g1_m,g2_m]
print("System constraints:", Gs)
print("Model constraints:", Gm)

#cost function gradients
djsdp = sols['lam_p'].full() #Exact
djmdp = solm2['lam_p'].full()
djsdp_fdiff = finite_dif_cost(system_costfunc,u0)
print("System cost function gradient for u:", djsdp, djsdp_fdiff)
print("Model cost function gradient for u:", djmdp)

#constraint gradients
dGsdp = [sols_grad_1['lam_p'].full(),sols_grad_2['lam_p'].full()] #gradient of constraints system, exact
dGsdp_fdiff = finite_dif_constraits(system_constraint,u0) #with finite differences
dGmdp = [solm_grad_1['lam_p'].full(),solm_grad_2['lam_p'].full()] #gradient of constraints model
print("System constraint gradients:", dGsdp, dGsdp_fdiff)
print("Model constraint gradients:", dGmdp)

#optimal x
x_opts = sols['x'].full()
x_optm = solm['x'].full() 

p=u # just doing this for clarity of explanation
#dgps = Function('dgsdp',[ws,p],[jacobian(gs,p)])  # this creates a function with input w,p that gives as output dgdp
# the function should then be called at the steady state values (found by the optimisation) 
#dgsdp= dgps(sols['x'].full(),u0)
#print(dgsdp)

#dgdu_model_f = Function('dgdu_model_f',[wm_grad,p],[jacobian(gm_grad,p)])
#dgdu_model= dgdu_model_f(solm_grad_1['x'],solm['x'].full()[5:6])
#print(dgdu_model)
#jacobian(gm_grad,p)

#MODIFIER CALCULATION - set ut with parameters
modifier_param = []

eps_k =SX.sym("eps_k", 2)

lamb_cost_k = SX.sym("lamb_cost_k", 2) #djsdp-djmdp
lamb_const_k = SX.sym("lamb_const_k", 4) # make into array 
modifier_param.append(vertcat(eps_k))
modifier_param.append(vertcat(lamb_cost_k))
modifier_param.append(vertcat(lamb_const_k))

u_current = SX.sym("eps_k", 2)

modifier_param.append(vertcat(u_current))


modifier_param = vertcat(*modifier_param)


#OPTIMIZATION: use model to set the next uk
w=[] # decision variables
w0 = []
lbw = []
ubw= []

J = 0
g=[]
lbg = []
ubg = []

# states
w.append(xa_m)  
w.append(u)
lbw.append([0, 0, 0, 0 ,0 ,4,70])
ubw.append([np.inf, np.inf,np.inf,np.inf,np.inf,7,100])
w0.append([0, 0, 0, 0 ,0,u_test[0],u_test[1]])



J = Jm - lamb_cost_k.T@(u-u_current)
Aineq_MA = [xa_m[0]-0.12+eps_k[0] - lamb_const_k[0]*(u[0]-u_current[0])-lamb_const_k[1]*(u[1]-u_current[1]),
xa_m[-1]-0.08+eps_k[1] -lamb_const_k[2]*(u[0]-u_current[0])-lamb_const_k[3]*(u[1]-u_current[1])]

g.append(vcat(Aeqm))
g.append(vcat(Aineq_MA))

lbg.append([0, 0, 0, 0 ,0 , -np.inf,-np.inf]) 
ubg.append([0, 0, 0, 0 ,0 , 0, 0])

w = vertcat(*w)
g = vertcat(*g)
w0_opt = np.concatenate(w0)
lbw = np.concatenate(lbw)
ubw = np.concatenate(ubw)
lbg = np.concatenate(lbg)
ubg = np.concatenate(ubg)

prob = {'f': J, 'x': w, 'g': g, 'p': modifier_param}
solver = nlpsol('solver', 'ipopt', prob)


eps_k0 = Gs[0]-Gm[0]
eps_k1 = Gs[1]-Gm[1]

lamb_cost_k0 = (djsdp[0]-djmdp[0])
lamb_cost_k1 = (djsdp[1]-djmdp[1])
lamb_const_k0 = (dGsdp[0][0]-dGmdp[0][0])
lamb_const_k1 = (dGsdp[0][1]-dGmdp[0][1])
lamb_const_k2 = (dGsdp[1][0]-dGmdp[1][0])
lamb_const_k3 = (dGsdp[1][1]-dGmdp[1][1])



lamb_cost_k0_fd = (djsdp_fdiff[0][0]-djmdp[0]) #with finite diff for the system
lamb_cost_k1_fd =  (djsdp_fdiff[0][1]-djmdp[1])
lamb_const_k0_fd = (dGsdp_fdiff[0][0]-dGmdp[0][0])
lamb_const_k1_fd = (dGsdp_fdiff[0][1]-dGmdp[0][1])
lamb_const_k2_fd = (dGsdp_fdiff[1][0]-dGmdp[1][0])
lamb_const_k3_fd = (dGsdp_fdiff[1][1]-dGmdp[1][1])

mod_param = [eps_k0,eps_k1,lamb_cost_k0,lamb_cost_k1,lamb_const_k0,lamb_const_k1,lamb_const_k2,lamb_const_k3, u_test[0],  u_test[1]]
mod_param_fd = [eps_k0,eps_k1,lamb_cost_k0_fd,lamb_cost_k1_fd,lamb_const_k0_fd,lamb_const_k1_fd,lamb_const_k2_fd,lamb_const_k3_fd, u_test[0],  u_test[1]]

mod_param = vertcat(*mod_param)
mod_param_fd = vertcat(*mod_param_fd)

sol = solver(x0=w0_opt, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=mod_param_fd)
u_temp = sol['x'][-2:].full()
w_opt = sol['x'].full()
J_opt = sols['f'].full()

u_list = []

u_list.append(np.vstack(u0))
u_opt = np.vstack(u0)
# set the initial modifiers

#For plottingƒ
G1_list = []
Xa_list = []
G2_list = []
Xg_list = []
J_list = []
iter = []

J_list.append(-float(J_opt))
xa_opt = np.array(x_opts[:1])
xg_opt = np.array(x_opts[5:6])
Xa_list.append(float(xa_opt))
Xg_list.append(float(xg_opt))
iter.append(0)

w0m2 = w0m
for i in range(20):
    # at this stage we already have a u_opt.

   # sol_model = solver(x0=w0_opt, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=[0,0,0,0,0,0,0,0]) #update in a loop
    #u_opt =  # model predicts optimal u to be
    # w_opt = sol['x'].full()
    # u_opt = w_opt[5:]


    # calculate the difference in cost, objective and gradients
    #EVALUATION
    # Solve the NLP
    
    solm = solverm(x0=w0m, lbx=lbwm, ubx=ubwm, lbg=lbgm, ubg=ubgm)

    solm2 = solverm(x0=w0m2, lbx=lbwm, ubx=ubwm, lbg=lbgm, ubg=ubgm)
    w0m2 = solm2['x'].full()
    w0m2 = w0m2.reshape(-1,1)

    # sol = solver(x0=w0_opt, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=[eps_k0,eps_k1,lamb_cost_k0,lamb_cost_k1,lamb_const_k0,lamb_const_k1,lamb_const_k2,lamb_const_k3])
    
    # evaluate the system @ u_opt

    sols = solvers(x0=w0s, lbx=lbws, ubx=ubws, lbg=lbgs, ubg=ubgs,p=u_opt)


    #to find model optimum
    sols2 = solvers2(x0=w0s2, lbx=lbws2, ubx=ubws2, lbg=lbgs2, ubg=ubgs2)
    w0s2 = sols2['x'].full()
    w0s2 = w0s2.reshape(-1,1)
    # evaluate the system and model gradients @ u_opt

    solm2 = solverm2(x0=w0m[0:5], lbx=lbwm[0:5], ubx=ubwm[0:5], lbg=lbgm[0:5], ubg=ubgm[0:5], p = u_opt)
    sols_grad_1 = solvers_grad_1(x0=sols['x'].full(), lbx=lbws, ubx=ubws, lbg=lbgs, ubg=ubgs,p=u_opt)
    sols_grad_2 = solvers_grad_2(x0=sols['x'].full(), lbx=lbws, ubx=ubws, lbg=lbgs, ubg=ubgs,p=u_opt)
    solm_grad_1 = solverm_grad_1(x0=x0gm, lbx=lbwm_const, ubx=ubwm_const, lbg=lbgm_const, ubg=ubgm_const,p=u_opt)
    solm_grad_2 = solverm_grad_2(x0=x0gm, lbx=lbwm_const, ubx=ubwm_const, lbg=lbgm_const, ubg=ubgm_const,p=u_opt)

    #cost functions
    j_s = sols['f'].full()
    j_m = solm['f'].full()
    print("System cost function:", j_s)
    print("Model cost function:", j_m)

    #constraints
    g1_s = sols_grad_1['f'].full()
    g2_s = sols_grad_2['f'].full()
    Gs = [g1_s,g2_s]
    g1_m = solm_grad_1['f'].full()
    g2_m = solm_grad_2['f'].full()
    Gm = [g1_m,g2_m]
    print("System constraints:", Gs)
    print("Model constraints:", Gm)

    #cost function gradients
    djsdp = sols['lam_p'].full()
    djmdp = solm2['lam_p'].full()
    print("System cost function gradient for u:", djsdp)
    print("Model cost function gradient for u:", djmdp)

    #constraint gradients
    dGsdp = [sols_grad_1['lam_p'].full(),sols_grad_2['lam_p'].full()] #gradient of constraints system
    dGmdp = [solm_grad_1['lam_p'].full(),solm_grad_2['lam_p'].full()] #gradient of constraints model
    print("System constraint gradients:", dGsdp)
    print("Model constraint gradients:", dGmdp)

    djsdp_fdiff = finite_dif_cost(system_costfunc,u_opt)
    #djsdp_fdiff = finite_dif_cost(system_costfunc,u_opt,j_s) #where js is the noisy measurement
    dGsdp_fdiff = finite_dif_constraits(system_constraint,u_opt) #with finite differences
    #dGsdp_fdiff = finite_dif_constraits(system_constraint,u_opt,Gs) # where Gs has noise


    #optimal x
    x_opts = sols['x'].full()
    x_optm = solm['x'].full() 

    # MODIFIERS
    # define a filter for the modifiers
    _filter = 0.6
    _filter_min = 1-_filter

    #zero order MA
    eps_k0 =  _filter*eps_k0 + _filter_min*(Gs[0]-Gm[0]) 
    eps_k1 = _filter*eps_k1 + _filter_min*(Gs[1]-Gm[1]) 

    #first order

    lamb_cost_k0 = _filter*lamb_cost_k0 + _filter_min*(djsdp[0]-djmdp[0])
    lamb_cost_k1 =  _filter*lamb_cost_k1 + _filter_min*(djsdp[1]-djmdp[1])
    lamb_const_k0 = _filter*lamb_const_k0 + _filter_min*(dGsdp[0][0]-dGmdp[0][0])
    lamb_const_k1 = _filter*lamb_const_k1 + _filter_min*(dGsdp[0][1]-dGmdp[0][1])
    lamb_const_k2 = _filter*lamb_const_k2 + _filter_min*(dGsdp[1][0]-dGmdp[1][0])
    lamb_const_k3 = _filter*lamb_const_k3 + _filter_min*(dGsdp[1][1]-dGmdp[1][1])

    lamb_cost_k0_fd = _filter*lamb_cost_k0_fd + _filter_min*(djsdp_fdiff[0][0]-djmdp[0]) #with finite diff for the system
    lamb_cost_k1_fd =  _filter*lamb_cost_k1_fd + _filter_min*(djsdp_fdiff[0][1]-djmdp[1])
    lamb_const_k0_fd = _filter*lamb_const_k0_fd + _filter_min*(dGsdp_fdiff[0][0]-dGmdp[0][0])
    lamb_const_k1_fd = _filter*lamb_const_k1_fd + _filter_min*(dGsdp_fdiff[0][1]-dGmdp[0][1])
    lamb_const_k2_fd = _filter*lamb_const_k2_fd + _filter_min*(dGsdp_fdiff[1][0]-dGmdp[1][0])
    lamb_const_k3_fd = _filter*lamb_const_k3_fd + _filter_min*(dGsdp_fdiff[1][1]-dGmdp[1][1])

    # create vector of modifiers, including the current u_opt
    mod_param = [eps_k0,eps_k1,lamb_cost_k0,lamb_cost_k1,lamb_const_k0,lamb_const_k1,lamb_const_k2,lamb_const_k3, u_opt[0],  u_opt[1]]
    mod_param_fd = [eps_k0,eps_k1,lamb_cost_k0_fd,lamb_cost_k1_fd,lamb_const_k0_fd,lamb_const_k1_fd,lamb_const_k2_fd,lamb_const_k3_fd, u_opt[0],  u_opt[1]]

    mod_param = vertcat(*mod_param)
    mod_param_fd = vertcat(*mod_param_fd)

    sol = solver(x0=w_opt, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=mod_param_fd)
    w_opt = sol['x'].full()
    u_temp = np.array(w_opt[5:])

    J_opt = sols['f'].full()
    J_list.append(float(-J_opt))
    iter.append(int(i+1))

    xa_opt = np.array(x_opts[:1])
    xg_opt = np.array(x_opts[5:6])
    Xa_list.append(float(xa_opt))
    Xg_list.append(float(xg_opt))

    w0m = []
    w0m.append([0,0,0,0,0,x_optm[-2],x_optm[-1]])
    w0m = np.concatenate(w0m)
    # filter u
    kappa=0.4
    u_opt = kappa*u_list[-1] + (1-kappa)*u_temp
    #check convergence
    u_list.append(u_opt)
    if np.sum(np.abs(u_list[-2]-u_list[-1]))<10**(-5):
        break
    #check if u has converged


Fb_list = []
Tr_list = []

for i in range(len(u_list)):
    Fb_list.append(u_list[i][0])
    Tr_list.append(u_list[i][1])

G1_Fb = [4,4.125,4.3,4.39,4.45,4.5,4.75,5,5.25,5.36,5.5,5.75,6,6.1]
G1_Tr = [84,82.65,81.2,80.5,80,79.5,77.6,76,74.5,73.9,73.2,71.85,70.5,70]
a_BSpline = make_interp_spline(G1_Fb, G1_Tr)
G1_Tr_new = a_BSpline(G1_Fb)
# 300 represents number of points to make between T.min and T.max
G1_Fb_new = np.linspace(4, 6.1, 300) 

spl = make_interp_spline(G1_Fb, G1_Tr, k=3)  # type: BSpline
G1_Tr_smooth = spl(G1_Fb_new)

G2_Fb = [4,4.3,4.39,4.5,4.7,5.05,5.25,5.5,7]
G2_Tr = [77.7,80,80.5,81.5,82.5,85,86,88,98.5]
G2_Fb = [4,7]
G2_Tr = [77.605,98.5]
G2_Tr1 = np.linspace(77.605,98.5,400)

x = np.append(G1_Fb_new,np.linspace(6.1,7,100))
G1_Tr_add = np.ones(100)*70
G1_Tr_fill = np.append(G1_Tr_smooth, G1_Tr_add)
y = np.linspace(10,98.5,300)
midTr = u_opt[1]*np.ones(400)
midTr2 = (u_opt[1]+0.1)*np.ones(400)
x2 = np.linspace(4,7,400)
  



#%%
font = {'fontname':'Calibri'}

plt.figure(figsize=(8,6)) 
plt.plot(G1_Fb_new,G1_Tr_smooth,'r',label='$g_1$')
plt.plot(G2_Fb,G2_Tr,'firebrick',label='$g_2$')
plt.plot(Fb_list,Tr_list,'--',color='tab:blue')
plt.plot(Fb_list,Tr_list,'*',label='$u_k$',color='tab:blue')
plt.plot(Fb_list[-1],Tr_list[-1],'r*',label='$u_k^f$')
#plt.contour(Fb_list, Tr_list, , levels=14, linewidths=0.5, colors='k')

plt.grid(color='0.95')
plt.legend()
#plt.title('Standard MA')
plt.xlabel("$F_b$ [kg/s]")
plt.ylabel("$T_R$ [°C]")
plt.axis([4, 7, 70, 100])
plt.fill_between(x,midTr, G1_Tr_fill, where = (x>u_opt[0]),color='lightblue')
plt.fill_between(x2,midTr, G2_Tr1, color='lightblue',where = (x>u_opt[0]))
plt.show()

fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, sharex=True,
                                    figsize=(18, 4))

ax0.plot(iter,J_list)
ax0.set_xlabel('Iterations [-]')
ax0.set_ylabel('Plant profit [$/s]')
ax0.grid(color='0.95')
ax0.set(xlim=(0,len(iter)-1), ylim=(-60, 100))

ax1.plot(iter, Fb_list)
ax1.set_xlabel('Iterations [-]')
ax1.set_ylabel('$F_b$ [kg/s]')
ax1.grid(color='0.95')
ax1.set(xlim=(0,len(iter)-1), ylim=(4, 7))

ax2.plot(iter, Tr_list)
ax2.set_xlabel('Iterations [-]')
ax2.set_ylabel('$T_R$ [°C]')
ax2.grid(color='0.95')
ax2.set(xlim=(0,len(iter)-1), ylim=(70, 100))

fig.suptitle('Plant evaluations as a function of iterations', fontsize=14)
plt.show()


Xa_lim = 0.12*np.ones(len(iter))
Xg_lim = 0.08*np.ones(len(iter))
fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, sharex=True,
                                    figsize=(9.5, 4))

ax0.plot(iter,Xa_list)
ax0.plot(iter,Xa_lim,'r--')
ax0.set_xlabel('Iterations [-]')
ax0.set_ylabel('$x_A$ [-]')
ax0.legend(('$x_A$','$g_1$'),
           loc='lower right')
#ax0.set_label({'$X_A$','$G_1$'},'Location','northeast')
ax0.grid(color='0.95')
ax0.set(xlim=(0,len(iter)-1), ylim=(0.085, 0.122))

ax1.plot(iter, Xg_list)
ax1.plot(iter,Xg_lim,'r--')
ax1.legend(('$x_G$','$g_2$'),
           loc='lower right')
ax1.set_xlabel('Iterations [-]')
ax1.set_ylabel('$x_G$ [-]')
ax1.grid(color='0.95')
ax1.set(xlim=(0,len(iter)-1), ylim=(0.015, 0.11))

fig.suptitle('Constraint evaluations', fontsize=14)
plt.show()

print(Tr_list[-1],Fb_list[-1])


