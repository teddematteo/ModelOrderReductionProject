import numpy as np
import scipy.sparse
from tqdm import tqdm
from pypolydim import polydim, gedim
import other_utilities as other_ut
from pypolydim.export_vtk_utilities import ExportVTKUtilities

def speed_x_exact(x, y, z):
    return 0.0

def speed_y_exact(x, y, z):
    return 0.0

def pressure_exact(x, y, z):
    return 0.0

def generate_snapshots(snapshot_num, geometry_utilities, vtk_utilities, mesh, mesh_geometric_data, tot_dofs, export_solution_path,
                       speed_n_dofs, speed_mesh_dofs_info, speed_dofs_data, speed_reference_element_data, speed_n_strongs,
                       pressure_n_dofs, pressure_mesh_dofs_info, pressure_dofs_data, pressure_reference_element_data, pressure_n_strongs):
    
    mu0_range = [0.1, 10.]
    mu1_range = [1., 3.]
    P = np.array([mu0_range, mu1_range])
    training_set = np.random.uniform(low=P[:, 0], high=P[:, 1], size=(snapshot_num, P.shape[0]))

    snapshot_matrix_u = []
    snapshot_matrix_s = [] ### supremizer snapshot
    snapshot_matrix_p = []

    for mu_arr in tqdm(training_set, desc="Generating Snapshots"):
        mu0 = mu_arr[0]
        mu1 = mu_arr[1]

        def f_x_function(x, y, z):
            return -(mu1**3*np.pi**2*np.cos(mu1**2*np.pi*x)-mu1**2*np.pi**2)*np.sin(mu1*np.pi*y)*np.cos(mu1*np.pi*y)+(mu1*np.pi*np.cos(mu1*np.pi*x)*np.cos(mu1*np.pi*y))

        def f_y_function(x, y, z):
            return -(-mu1**3*(np.pi)**2*np.cos(mu1**2*np.pi*y)+mu1**2*np.pi**2)*np.sin(mu1*np.pi*x)*np.cos(mu1*np.pi*x)+(-mu1*np.pi*np.sin(mu1*np.pi*x)*np.sin(mu1*np.pi*y))
        
        f_x = polydim.pde_tools.assembler_utilities.pcc_2_d.assemble_source_term(geometry_utilities,
                                                                                mesh,
                                                                                mesh_geometric_data,
                                                                                speed_dofs_data,
                                                                                speed_reference_element_data,
                                                                                speed_reference_element_data,
                                                                                f_x_function)
        f_y = polydim.pde_tools.assembler_utilities.pcc_2_d.assemble_source_term(geometry_utilities,
                                                                                mesh,
                                                                                mesh_geometric_data,
                                                                                speed_dofs_data,
                                                                                speed_reference_element_data,
                                                                                speed_reference_element_data,
                                                                                f_y_function)
        f_S = np.concatenate([f_x, f_y, np.zeros(pressure_n_dofs)])

        def pressure_strong_function(marker, x, y, z):
            if marker==1:
                return 0.0 #vertice basso-sinistra
            else:
                return None
            
        def speed_x_strong_function(marker, x, y, z):
            return 0.0 #tutto il bordo
        def speed_y_strong_function(marker, x, y, z):
            return 0.0 #tutto il bordo

        p_strong = polydim.pde_tools.assembler_utilities.pcc_2_d.assemble_strong_solution(geometry_utilities,
                                                                            mesh,
                                                                            mesh_geometric_data,
                                                                            pressure_mesh_dofs_info,
                                                                            pressure_dofs_data,
                                                                            pressure_reference_element_data,
                                                                            pressure_strong_function)
            
        u_x_strong = polydim.pde_tools.assembler_utilities.pcc_2_d.assemble_strong_solution(geometry_utilities,
                                                                            mesh,
                                                                            mesh_geometric_data,
                                                                            speed_mesh_dofs_info,
                                                                            speed_dofs_data,
                                                                            speed_reference_element_data,
                                                                            speed_x_strong_function)
        u_y_strong = polydim.pde_tools.assembler_utilities.pcc_2_d.assemble_strong_solution(geometry_utilities,
                                                                            mesh,
                                                                            mesh_geometric_data,
                                                                            speed_mesh_dofs_info,
                                                                            speed_dofs_data,
                                                                            speed_reference_element_data,
                                                                            speed_y_strong_function)
        
        def mu_term(x, y, z):
            return mu0
        def b_x_term(x, y, z):
            return np.array([1.0,0.0,0.0])
        def b_y_term(x, y, z):
            return np.array([0.0,1.0,0.0])

        A_operator = polydim.pde_tools.assembler_utilities.pcc_2_d.assemble_diffusion_operator(geometry_utilities,
                                                                                            mesh,
                                                                                            mesh_geometric_data,
                                                                                            speed_dofs_data,
                                                                                            speed_dofs_data,
                                                                                            speed_reference_element_data,
                                                                                            speed_reference_element_data,
                                                                                            mu_term)


        J_A_x = other_ut.make_np_sparse(A_operator.operator_dofs, [tot_dofs, tot_dofs], [0, 0])
        J_A_y = other_ut.make_np_sparse(A_operator.operator_dofs, [tot_dofs, tot_dofs], [speed_n_dofs, speed_n_dofs])

        B_x_operator = polydim.pde_tools.assembler_utilities.pcc_2_d.assemble_advection_operator(geometry_utilities,
                                                                            mesh,
                                                                            mesh_geometric_data,
                                                                            speed_dofs_data,
                                                                            pressure_dofs_data,
                                                                            speed_reference_element_data,
                                                                            pressure_reference_element_data,
                                                                            b_x_term)
        B_y_operator = polydim.pde_tools.assembler_utilities.pcc_2_d.assemble_advection_operator(geometry_utilities,
                                                                            mesh,
                                                                            mesh_geometric_data,
                                                                            speed_dofs_data,
                                                                            pressure_dofs_data,
                                                                            speed_reference_element_data,
                                                                            pressure_reference_element_data,
                                                                            b_y_term)

        J_B_x = other_ut.make_np_sparse(B_x_operator.operator_dofs, [tot_dofs, tot_dofs], [2 * speed_n_dofs, 0])
        J_B_y = other_ut.make_np_sparse(B_y_operator.operator_dofs, [tot_dofs, tot_dofs], [2 * speed_n_dofs, speed_n_dofs])
        J_BT_x = other_ut.make_np_sparse(B_x_operator.operator_dofs, [tot_dofs, tot_dofs], [2 * speed_n_dofs, 0], True)
        J_BT_y = other_ut.make_np_sparse(B_y_operator.operator_dofs, [tot_dofs, tot_dofs], [2 * speed_n_dofs, speed_n_dofs], True)

        J_S = J_A_x + J_A_y - J_B_x - J_B_y - J_BT_x - J_BT_y
        X_1 = other_ut.make_np_sparse(A_operator.operator_dofs, [2 * speed_n_dofs, 2 * speed_n_dofs], [0, 0])
        X_2 = other_ut.make_np_sparse(A_operator.operator_dofs, [2 * speed_n_dofs, 2 * speed_n_dofs], [speed_n_dofs, speed_n_dofs])
        B_1 = other_ut.make_np_sparse(B_x_operator.operator_dofs, [pressure_n_dofs, 2 * speed_n_dofs], [0, 0])
        B_2 = other_ut.make_np_sparse(B_y_operator.operator_dofs, [pressure_n_dofs, 2 * speed_n_dofs], [0, speed_n_dofs])

        def speed_x_initial_condition(x, y, z):
            return 0.0
        def speed_y_initial_condition(x, y, z):
            return 0.0
        def pressure_initial_condition(x, y, z):
            return 0.0

        u_x_numeric = polydim.pde_tools.assembler_utilities.pcc_2_d.evaluate_function_on_dofs(geometry_utilities,
                                                                                        mesh,
                                                                                        mesh_geometric_data,
                                                                                        speed_dofs_data,
                                                                                        speed_reference_element_data,
                                                                                        speed_x_initial_condition).function_dofs
        u_y_numeric = polydim.pde_tools.assembler_utilities.pcc_2_d.evaluate_function_on_dofs(geometry_utilities,
                                                                                        mesh,
                                                                                        mesh_geometric_data,
                                                                                        speed_dofs_data,
                                                                                        speed_reference_element_data,
                                                                                        speed_y_initial_condition).function_dofs
        p_numeric = polydim.pde_tools.assembler_utilities.pcc_2_d.evaluate_function_on_dofs(geometry_utilities,
                                                                                        mesh,
                                                                                        mesh_geometric_data,
                                                                                        pressure_dofs_data,
                                                                                        pressure_reference_element_data,
                                                                                        pressure_initial_condition).function_dofs

        u_k = np.concatenate([u_x_numeric, u_y_numeric, p_numeric])
        du_x_strong = np.zeros(speed_n_strongs)
        du_y_strong = np.zeros(speed_n_strongs)
        dp_strong = np.zeros(pressure_n_strongs)
        residual_norm = 1.0
        solution_norm = 1.0
        newton_tol = 1.0e-6
        max_iterations = 10
        num_iteration = 1

        while num_iteration < max_iterations and residual_norm > newton_tol * solution_norm:
            c_operator = polydim.pde_tools.assembler_utilities.pcc_2_d.assemble_ns_operators(geometry_utilities,
                                                                                            mesh,
                                                                                            mesh_geometric_data,
                                                                                            speed_dofs_data,
                                                                                            speed_reference_element_data,
                                                                                            u_x_numeric,
                                                                                            u_y_numeric,
                                                                                            u_x_strong,
                                                                                            u_y_strong)

            J_C = other_ut.make_np_sparse(c_operator.convective_operator.operator_dofs, [tot_dofs, tot_dofs], [0, 0])
            f_C = np.concatenate([c_operator.convective_rhs, np.zeros(pressure_n_dofs)])

            J_f = f_S - f_C - J_S @ u_k
            du = scipy.sparse.linalg.spsolve(J_S + J_C, J_f)

            u_k = u_k + du

            du_x = du[0:speed_n_dofs]
            du_y = du[speed_n_dofs:2 * speed_n_dofs]
            dp = du[2 * speed_n_dofs:]

            u_x_numeric = u_k[0:speed_n_dofs]
            u_y_numeric = u_k[speed_n_dofs:2 * speed_n_dofs]
            u_numeric = u_k[0:2*speed_n_dofs]
            p_numeric = u_k[2 * speed_n_dofs:]

            

            du_x_error_L2 = polydim.pde_tools.assembler_utilities.pcc_2_d.compute_error_l2(geometry_utilities,
                                                                                        mesh,
                                                                                        mesh_geometric_data,
                                                                                        speed_dofs_data,
                                                                                        speed_reference_element_data,
                                                                                        du_x,
                                                                                        du_x_strong)
            du_y_error_L2 = polydim.pde_tools.assembler_utilities.pcc_2_d.compute_error_l2(geometry_utilities,
                                                                                        mesh,
                                                                                        mesh_geometric_data,
                                                                                        speed_dofs_data,
                                                                                        speed_reference_element_data,
                                                                                        du_y,
                                                                                        du_y_strong)
            dp_error_L2 = polydim.pde_tools.assembler_utilities.pcc_2_d.compute_error_l2(geometry_utilities,
                                                                                        mesh,
                                                                                        mesh_geometric_data,
                                                                                        pressure_dofs_data,
                                                                                        pressure_reference_element_data,
                                                                                        dp,
                                                                                        dp_strong)
            u_x_error_L2 = polydim.pde_tools.assembler_utilities.pcc_2_d.compute_error_l2(geometry_utilities,
                                                                                        mesh,
                                                                                        mesh_geometric_data,
                                                                                        speed_dofs_data,
                                                                                        speed_reference_element_data,
                                                                                        u_x_numeric,
                                                                                        u_x_strong,
                                                                                        speed_x_exact)
            u_y_error_L2 = polydim.pde_tools.assembler_utilities.pcc_2_d.compute_error_l2(geometry_utilities,
                                                                                        mesh,
                                                                                        mesh_geometric_data,
                                                                                        speed_dofs_data,
                                                                                        speed_reference_element_data,
                                                                                        u_y_numeric,
                                                                                        u_y_strong,
                                                                                        speed_y_exact)
            p_error_L2 = polydim.pde_tools.assembler_utilities.pcc_2_d.compute_error_l2(geometry_utilities,
                                                                                        mesh,
                                                                                        mesh_geometric_data,
                                                                                        pressure_dofs_data,
                                                                                        pressure_reference_element_data,
                                                                                        p_numeric,
                                                                                        p_strong,
                                                                                        pressure_exact)

            solution_norm = np.sqrt(u_x_error_L2.numeric_norm_l2 * u_x_error_L2.numeric_norm_l2 +
                                u_y_error_L2.numeric_norm_l2 * u_y_error_L2.numeric_norm_l2 +
                                p_error_L2.numeric_norm_l2 * p_error_L2.numeric_norm_l2);
            residual_norm = np.sqrt(du_x_error_L2.numeric_norm_l2 * du_x_error_L2.numeric_norm_l2 +
                                du_y_error_L2.numeric_norm_l2 * du_y_error_L2.numeric_norm_l2 +
                                dp_error_L2.numeric_norm_l2 * dp_error_L2.numeric_norm_l2);

            #print(f"{u_k.shape}")
            #print("dofs", "u_x_errorL2", "u_y_errorL2", "p_errorL2", "residual", "iteration", "max_iteration")
            #print(tot_dofs, '{:.2e}'.format(u_x_error_L2.error_l2 / u_x_error_L2.numeric_norm_l2), '{:.2e}'.format(u_y_error_L2.error_l2 / u_y_error_L2.numeric_norm_l2), '{:.2e}'.format(p_error_L2.error_l2 / p_error_L2.numeric_norm_l2), '{:.16e}'.format(residual_norm / solution_norm), '{:d}'.format(num_iteration), '{:d}'.format(max_iterations))

            num_iteration = num_iteration + 1

        snapshot_matrix_u.append(np.copy(u_k[0:2*speed_n_dofs]))
        snapshot_matrix_p.append(np.copy(u_k[2*speed_n_dofs:]))
        snapshot_s = scipy.sparse.linalg.spsolve(X_1 + X_2, np.transpose(B_1 + B_2) @ u_k[2*speed_n_dofs:])
        snapshot_matrix_s.append(np.copy(snapshot_s)) 

    snapshot_matrix_u = np.array(snapshot_matrix_u)
    snapshot_matrix_s = np.array(snapshot_matrix_s)
    snapshot_matrix_p = np.array(snapshot_matrix_p)
    inner_product_u = X_1 + X_2
    return snapshot_matrix_u,snapshot_matrix_s,snapshot_matrix_p,inner_product_u

"""def eig_analysis(C, N_max=None, tol=1e-9):
    L_e, VM_e = np.linalg.eig(C)
    eigenvalues = []
    eigenvectors = []

    for i in range(len(L_e)):
        eig_real = L_e[i].real
        eig_complex = L_e[i].imag
        assert np.isclose(eig_complex, 0.)
        eigenvalues.append(eig_real)
        eigenvectors.append(VM_e[i].real)


    total_energy = sum(eigenvalues)
    retained_energy_vector = np.cumsum(eigenvalues)
    relative_retained_energy = retained_energy_vector/total_energy


    if all(flag==False for flag in relative_retained_energy>= tol) and N_max != None:
        N = N_max
    else:
        N = np.argmax(relative_retained_energy >= (1.0-tol)) + 1
    
    return N, eigenvectors"""

def eig_analysis(C, N_max=None, tol=1e-9):
    L_e, VM_e = np.linalg.eig(C)
    
    # 1. Ordina gli autovalori e i relativi autovettori in ordine decrescente
    idx = np.argsort(L_e.real)[::-1]
    L_e = L_e[idx]
    VM_e = VM_e[:, idx]

    eigenvalues = []
    eigenvectors = []

    for i in range(len(L_e)):
        eig_real = L_e[i].real
        eig_complex = L_e[i].imag
        assert np.isclose(eig_complex, 0.)
        eigenvalues.append(eig_real)
        
        # 2. Gli autovettori sono le COLONNE, non le righe
        eigenvectors.append(VM_e[:, i].real)

    total_energy = sum(eigenvalues)
    retained_energy_vector = np.cumsum(eigenvalues)
    relative_retained_energy = retained_energy_vector / total_energy

    # 3. Logica della tolleranza corretta (1.0 - tol)
    threshold = 1.0 - tol
    
    if all(flag == False for flag in relative_retained_energy >= threshold) and N_max is not None:
        N = N_max
    else:
        N = np.argmax(relative_retained_energy >= threshold) + 1
    
    # Rispetta l'eventuale limite imposto da N_max
    if N_max is not None and N > N_max:
        N = N_max
        
    return N, eigenvectors

def create_basis_functions_matrix(N, snapshot_matrix, eigenvectors, inner_product=None):
    basis_functions = []
    
    for n in range(N):
        eigenvector =  eigenvectors[n]
        basis = np.transpose(snapshot_matrix)@eigenvector
        if inner_product!= None:
            norm = np.sqrt(np.transpose(basis) @ inner_product @ basis) ## metti inner product
        else:
            norm = np.sqrt(np.transpose(basis) @ basis)
        basis /= norm
        basis_functions.append(np.copy(basis))

    basis_function_matrix = np.transpose(np.array(basis_functions))
    
    return basis_function_matrix

def generate_basis(snapshot_matrix_u,snapshot_matrix_s,snapshot_matrix_p,inner_product_u,N_max,tol):
    C_u = snapshot_matrix_u @ inner_product_u @ np.transpose(snapshot_matrix_u)
    C_s = snapshot_matrix_s @ inner_product_u @ np.transpose(snapshot_matrix_s)
    C_p = snapshot_matrix_p @ np.transpose(snapshot_matrix_p)

    N_u, eigs_u = eig_analysis(C_u, N_max=N_max, tol=tol)
    N_s, eigs_s = eig_analysis(C_s, N_max=N_max, tol=tol)
    N_p, eigs_p = eig_analysis(C_p, N_max=N_max, tol=tol)

    basis_functions_u = create_basis_functions_matrix(N_u, snapshot_matrix_u, eigs_u, inner_product=inner_product_u)
    basis_functions_s = create_basis_functions_matrix(N_s, snapshot_matrix_s, eigs_s, inner_product=inner_product_u)
    basis_functions_p = create_basis_functions_matrix(N_p, snapshot_matrix_p, eigs_p)

    global_basis_function_matrix = np.zeros((basis_functions_u.shape[0] + basis_functions_p.shape[0],N_u + N_s + N_p))
    global_basis_function_matrix[0:basis_functions_u.shape[0], 0:N_u] = basis_functions_u
    global_basis_function_matrix[0:basis_functions_u.shape[0], N_u : N_u + N_s] = basis_functions_s
    global_basis_function_matrix[basis_functions_u.shape[0]:, N_u + N_s:] = basis_functions_p

    return global_basis_function_matrix


    