from dolfin import *
import matplotlib.pyplot as plt
import numpy as np

def solver(kappa):
    # Create mesh and define function space
    mesh = UnitSquareMesh(6, 4)
    V = FunctionSpace(mesh, "Lagrange", 1)

    # Define boundary conditions
    u0 = Expression("1 + x[0]*x[0] + 2*x[1]*x[1]",degree=2)
    def u0_boundary(x, on_boundary):
        return on_boundary
    bc = DirichletBC(V, u0, u0_boundary)

    # Define variational problem
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(-6.0)

    a = kappa*dot(grad(u), grad(v))*dx # a = inner((K*nabla_grad(u)), nabla_grad(v))*dx
    L = f*v*dx
    # Compute solution
    u = Function(V)
    solve(a == L, u, bc)

    # Plot solution and mesh
    plot(u)
    plot(mesh)
    plt.show()

if __name__ == '__main__':
    kappa = Constant(1.0)
    solver(kappa)

    kappa_list = np.random.normal(1, 1, 10)
    for kappa in kappa_list:
        solver(kappa)
