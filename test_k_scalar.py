from dolfin import *
import ufl
import matplotlib.pyplot as plt
import numpy

def solver(k_values):
    # Create mesh and define function space
    mesh = UnitSquareMesh(4, 6)
    # plot(mesh, title='mesh')
    # plt.show()
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
    kvalues = k_values # values of k in the two subdomains
    # print(len(subdomains.array()))
    for cell_no in range(len(subdomains.array())):
        subdomain_no = subdomains.array()[cell_no]
        k.vector()[cell_no] = kvalues[subdomain_no]
    # print('k degree of freedoms:', k.vector().get_local())

    # Define Dirichlet conditions for y=0 boundary
    tol = 1E-14   # tolerance for coordinate comparisons
    # u_L = Expression("x[0]", degree=1)
    # u_R = Expression("x[0]", degree=1)
    class BottomBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and abs(x[1]) < tol
    Gamma_0 = DirichletBC(X, Constant((0,0)), BottomBoundary())

    # Define Dirichlet conditions for y=1 boundary
    class TopBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and abs(x[1] - 1) < tol
    Gamma_1 = DirichletBC(X, Constant((0,0)), TopBoundary())

    bcs = [Gamma_0, Gamma_1]

    # define corrector equation
    u, m = TrialFunctions(X)
    v, r = TestFunctions(X)
    Fc = u*r*dx + m*v*dx
    # Fl = dot(k*nabla_grad(u),nabla_grad(v))*dx + k * nabla_grad(v) * dx
    Fl1 = inner(nabla_grad(u) + Constant((1,0)), k*nabla_grad(v))*dx
    F1 = Fc + Fl1

    Fl2 = inner(nabla_grad(u) + Constant((0,1)), k*nabla_grad(v))*dx
    F2 = Fc + Fl2

    # Compute first order corrector    
    x = Function(X)
    a1 = lhs(F1)
    L1 = rhs(F1)
    solve(a1 == L1, x, bcs)
    uh_x, mh_x = x.split()
    # print(uh_x.ufl_shape)
    # print(uh_x.vector().get_local())
    # plot(uh_x)
    # plt.show()
    # plot(mh_x)
    # plt.show()

    y = Function(X)
    a2 = lhs(F2)
    L2 = rhs(F2)
    solve(a2 == L2, y)
    uh_y, mh_y = y.split()
    # print(uh_y.vector().get_local())
    # plot(uh_y)
    # plt.show()
    # plot(mh_y)
    # plt.show()

    # print(nabla_grad(uh_x).ufl_shape)
    # print(nabla_grad(uh_y).ufl_shape)
    A_bar_11 = assemble((k*nabla_grad(uh_x)[0] + k)*dx)
    A_bar_12 = assemble((k*nabla_grad(uh_x)[1])*dx)
    A_bar_21 = assemble((k*nabla_grad(uh_y)[0])*dx)
    A_bar_22 = assemble((k*nabla_grad(uh_y)[1] + k)*dx)
    A_bar = numpy.array([[A_bar_11, A_bar_12], [A_bar_21, A_bar_22]])
    return A_bar

if __name__ == '__main__':
    k_values = [1.5, 50]  # values of k in the two subdomains
    # A_bar = solver(k_values)
    # print("A_bar", A_bar)
    k_list = numpy.random.normal(0, 1, (10000,2))
    k_list = k_list + k_values
    # print(k_list)
    A_bar_11 = []
    A_bar_12 = []
    A_bar_21 = []
    A_bar_22 = []
    for k_values in k_list:
        k_values = k_values.tolist()
        A_bar= solver(k_values)
        A_bar_11.append(A_bar[0,0])
        A_bar_12.append(A_bar[0,1])
        A_bar_21.append(A_bar[1,0])
        A_bar_22.append(A_bar[1,1])
        # print("k_values", k_values)
        # print("A_bar", A_bar)
    plt.subplot(2,2,1)
    plt.hist(A_bar_11,label="The mean of A_bar_11: %f\n" %(numpy.array(A_bar_11).mean()) + 
             "The variance of A_bar_11: %f" %(numpy.array(A_bar_11).var()))
    plt.legend()
    plt.title("results of A_bar_11")
    # plt.plot(numpy.array(A_bar_11).mean(), 0, c="orange", marker="o", label="The mean of A_bar_11: %f" %(numpy.array(A_bar_11).mean()))
    # plt.plot(numpy.array(A_bar_11).var(), 0, c="green", marker="d", label="The variance of A_bar_11: %f" %(numpy.array(A_bar_11).var()))
    
    plt.subplot(2,2,2)
    plt.hist(A_bar_12,label="The mean of A_bar_12: %f\n" %(numpy.array(A_bar_12).mean()) + 
             "The variance of A_bar_12: %f" %(numpy.array(A_bar_12).var()))
    plt.legend()
    plt.title("results of A_bar_12")

    plt.subplot(2,2,3)
    plt.hist(A_bar_21,label="The mean of A_bar_21: %f\n" %(numpy.array(A_bar_21).mean()) + 
             "The variance of A_bar_21: %f" %(numpy.array(A_bar_21).var()))
    plt.legend()
    plt.title("results of A_bar_21")

    plt.subplot(2,2,4)
    plt.hist(A_bar_22,label="The mean of A_bar_22: %f\n" %(numpy.array(A_bar_22).mean()) + 
             "The variance of A_bar_22: %f" %(numpy.array(A_bar_22).var()))
    plt.legend()
    plt.title("results of A_bar_22")
    plt.show()
