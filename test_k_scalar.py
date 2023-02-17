from dolfin import *
import ufl
import matplotlib.pyplot as plt
import numpy


# Create mesh and define function space
mesh = UnitSquareMesh(4, 6)
# plot(mesh, title='mesh')
# plt.show()
# V = VectorFunctionSpace(mesh, "Lagrange", 2)
P1 = FiniteElement('Lagrange', mesh.ufl_cell(), 1)
R = FiniteElement('Real', mesh.ufl_cell(), 0)
X = FunctionSpace(mesh, P1*R)

# Define subdomains
subdomains = MeshFunction("size_t", mesh, 2)
class Omega_0(SubDomain):
    def inside(self, x, on_boundary):
        return True if x[1] <= 0.5 else False
class Omega_1(SubDomain):
    def inside(self, x, on_boundary):
        return True if x[1] >= 0.5 else False
# Mark subdomains with numbers 0 and 1
subdomain_0 = Omega_0()
subdomain_1 = Omega_1()
subdomain_0.mark(subdomains, 0)
subdomain_1.mark(subdomains, 1)

# plot(subdomains, title='subdomains')
# plt.show()

# fill cell value in k
V0 = FunctionSpace(mesh,"DG",0)
k = Function(V0)
k_values = [1.5, 50] # values of k in the two subdomains
print(len(subdomains.array()))
for cell_no in range(len(subdomains.array())):
    subdomain_no = subdomains.array()[cell_no]
    k.vector()[cell_no] = k_values[subdomain_no]
# print('k degree of freedoms:', k.vector().get_local())

# # Define Dirichlet conditions for y=0 boundary
# tol = 1E-14   # tolerance for coordinate comparisons
# # u_L = Expression("x[0]", degree=1)
# # u_R = Expression("x[0]", degree=1)
# class BottomBoundary(SubDomain):
#     def inside(self, x, on_boundary):
#         return on_boundary and abs(x[1]) < tol
# Gamma_0 = DirichletBC(V, Constant(0.0), BottomBoundary())

# # Define Dirichlet conditions for y=1 boundary
# class TopBoundary(SubDomain):
#     def inside(self, x, on_boundary):
#         return on_boundary and abs(x[1] - 1) < tol
# Gamma_1 = DirichletBC(V, Constant(1.1), TopBoundary())

# bcs = [Gamma_0, Gamma_1]

# define corrector equation
u, m = TrialFunctions(X)
v, r = TestFunctions(X)
Fc = u*r*dx + m*v*ds
# Fl = dot(k*nabla_grad(u),nabla_grad(v))*dx + k * nabla_grad(v) * dx
Fl1 = inner(nabla_grad(u) + Constant((1,0)), k*nabla_grad(v))*dx
F1 = Fc + Fl1

Fl2 = inner(nabla_grad(u) + Constant((0,1)), k*nabla_grad(v))*dx
F2 = Fc + Fl2

# Compute first order corrector    
x = Function(X)
a1 = lhs(F1)
L1 = rhs(F1)
solve(a1 == L1, x)
uh_x, mh_x = x.split()
plot(uh_x)
plt.show()
# plot(mh_x)
# plt.show()

y = Function(X)
a2 = lhs(F2)
L2 = rhs(F2)
solve(a2 == L2, y)
uh_y, mh_y = y.split()
plot(uh_y)
plt.show()
# plot(mh_y)
# plt.show()

A_bar = assemble((k*nabla_grad(uh_x) + k*nabla_grad(uh_y))*dx)