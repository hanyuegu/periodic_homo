from dolfin import *
import matplotlib.pyplot as plt
import numpy

def solver(k_values):
    # Create mesh and define function space
    mesh = UnitSquareMesh(4, 6)
    V = FunctionSpace(mesh, "Lagrange", 1)

    # Define subdomains
    subdomains = MeshFunction('size_t', mesh, 2)

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

    # fill cell value in k
    V0 = FunctionSpace(mesh, 'DG', 0)
    k = Function(V0)
    help = numpy.asarray(subdomains.array(), dtype=numpy.int32)
    k.vector()[:] = numpy.choose(help, k_values)
    print('k degree of freedoms:', k.vector().get_local())

    plot(subdomains, title='subdomains')
    # plt.show()

    # Define Dirichlet conditions for y=0 boundary
    tol = 1E-14   # tolerance for coordinate comparisons
    class BottomBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and abs(x[1]) < tol
    Gamma_0 = DirichletBC(V, Constant(0), BottomBoundary())

    # Define Dirichlet conditions for y=1 boundary
    class TopBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and abs(x[1] - 1) < tol
    Gamma_1 = DirichletBC(V, Constant(1), TopBoundary())

    bcs = [Gamma_0, Gamma_1]

    # define corrector equation
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(0)
    a = inner(k*(nabla_grad(u) + Constant((1,0))),nabla_grad(v))*dx
    L = f*v*dx
    # F = k * inner(nabla_grad(u), nabla_grad(v)) * dx + k * nabla_grad(v) * dx
    # a, L = lhs(F), rhs(F)

    # Compute first order corrector
    u = Function(V)
    solve(a == L, u, bcs)

    return u,mesh

def show_plot(u,mesh):
    plot(u)
    plot(mesh)
    plt.show()


if __name__ == '__main__':
    k_values = [1.5, 50]  # values of k in the two subdomains
    chi,mesh = solver(k_values) 
    print(numpy.array(chi.vector()).shape)
    show_plot(chi,mesh)

    # k_list = numpy.random.normal(0, 1, (1000,2))
    # k_list = k_list + k_values
    # K_eff_sum = numpy.zeros((1,2))
    # for k_values in k_list:
    #     k_values = k_values.tolist()
    #     u, mesh, K_eff= solver(k_values)
    #     K_eff_sum = K_eff_sum + K_eff
    # print("K_eff", K_eff_sum/1000)