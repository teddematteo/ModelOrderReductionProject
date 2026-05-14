def generate_snapshot(snapshot_num):
    
    mu_range = [0.1, 10.]
    mu1_range = [1., 3.]
    P = np.array([mu_range, mu1_range])
    training_set = np.random.uniform(low=P[:, 0], high=P[:, 1], size=(snapshot_num, P.shape[0]))

    def f_x_function(x, y, z):
        return -(mu1**3*np.pi**2*np.cos(mu1**2*np.pi*x)-mu1**2*np.pi**2)*np.sin(mu1*np.pi*y)*np.cos(mu1*np.pi*y) + (mu1*np.pi*np.cos(mu1*np.pi*x)*np.cos(mu1*np.pi*y))

    def f_y_function(x, y, z):
        return -(-mu1**3 * (np.pi)**2 * np.cos(mu1**2 * np.pi * y) + mu1**2 * np.pi ** 2) * np.sin(mu1 * np.pi * x) * np.cos(mu1 * np.pi * x) + (-mu1 * np.pi * np.sin(mu1 * np.pi * x) * np.sin(mu1 * np.pi * y))

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

    p_strong = polydim.pde_tools.assembler_utilities.pcc_2_d.assemble_strong_solution(geometry_utilities,
                                                                        mesh,
                                                                        mesh_geometric_data,
                                                                        pressure_mesh_dofs_info,
                                                                        pressure_dofs_data,
                                                                        pressure_reference_element_data,
                                                                        pressure_strong_function)
        
    def speed_x_strong_function(marker, x, y, z):
        return 0.0 #tutto il bordo
    def speed_y_strong_function(marker, x, y, z):
        return 0.0 #tutto il bordo

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
        return mu
    def b_x_term(x, y, z):
        return np.array([\
            1.0,\
            0.0,\
            0.0])
    def b_y_term(x, y, z):
        return np.array([\
            0.0,\
            1.0,\
            0.0])

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

        print(f"{u_k.shape}")
        print("dofs", "u_x_errorL2", "u_y_errorL2", "p_errorL2", "residual", "iteration", "max_iteration")
        print(tot_dofs, '{:.2e}'.format(u_x_error_L2.error_l2 / u_x_error_L2.numeric_norm_l2), '{:.2e}'.format(u_y_error_L2.error_l2 / u_y_error_L2.numeric_norm_l2), '{:.2e}'.format(p_error_L2.error_l2 / p_error_L2.numeric_norm_l2), '{:.16e}'.format(residual_norm / solution_norm), '{:d}'.format(num_iteration), '{:d}'.format(max_iterations))

        num_iteration = num_iteration + 1

    u_x_error_L2 = polydim.pde_tools.assembler_utilities.pcc_2_d.compute_error_l2(geometry_utilities,
                                                                                    mesh,
                                                                                    mesh_geometric_data,
                                                                                    speed_dofs_data,
                                                                                    speed_reference_element_data,
                                                                                    u_x_numeric,
                                                                                    u_x_strong,
                                                                                    speed_x_exact)
    print("dofs", "errorL2")
    print(speed_dofs_data.number_do_fs, '{:.2e}'.format(u_x_error_L2.error_l2 / u_x_error_L2.numeric_norm_l2))

    u_y_error_L2 = polydim.pde_tools.assembler_utilities.pcc_2_d.compute_error_l2(geometry_utilities,
                                                                                    mesh,
                                                                                    mesh_geometric_data,
                                                                                    speed_dofs_data,
                                                                                    speed_reference_element_data,
                                                                                    u_y_numeric,
                                                                                    u_y_strong,
                                                                                    speed_y_exact)
    print("dofs", "errorL2")
    print(speed_dofs_data.number_do_fs, '{:.2e}'.format(u_y_error_L2.error_l2 / u_y_error_L2.numeric_norm_l2))

    p_error_L2 = polydim.pde_tools.assembler_utilities.pcc_2_d.compute_error_l2(geometry_utilities,
                                                                                    mesh,
                                                                                    mesh_geometric_data,
                                                                                    pressure_dofs_data,
                                                                                    pressure_reference_element_data,
                                                                                    p_numeric,
                                                                                    p_strong,
                                                                                    pressure_exact)
    print("dofs", "errorL2")
    print(pressure_dofs_data.number_do_fs, '{:.2e}'.format(p_error_L2.error_l2 / p_error_L2.numeric_norm_l2))

    u_x_on_cell0Ds = polydim.pde_tools.assembler_utilities.pcc_2_d.extract_solution_on_cell0_ds(mesh,
                                                                        speed_dofs_data,
                                                                                            u_x_numeric,
                                                                                            u_x_strong,
                                                                        speed_x_exact)
    u_y_on_cell0Ds = polydim.pde_tools.assembler_utilities.pcc_2_d.extract_solution_on_cell0_ds(mesh,
                                                                        speed_dofs_data,
                                                                                            u_y_numeric,
                                                                                            u_y_strong,
                                                                        speed_y_exact)
    p_on_cell0Ds = polydim.pde_tools.assembler_utilities.pcc_2_d.extract_solution_on_cell0_ds(mesh,
                                                                        pressure_dofs_data,
                                                                                            p_numeric,
                                                                                            p_strong,
                                                                        pressure_exact)
    
    vtk_utilities.export_solution_2(export_solution_path + '/u_x',
                                    mesh,
                                    u_x_on_cell0Ds.numeric_solution,
                                    u_x_on_cell0Ds.exact_solution,
                                    u_x_error_L2.cell2_ds_error_l2,
                                u_x_error_L2.cell2_ds_error_l2)
    vtk_utilities.export_solution_2(export_solution_path + '/u_y',
                                    mesh,
                                    u_y_on_cell0Ds.numeric_solution,
                                    u_y_on_cell0Ds.exact_solution,
                                    u_y_error_L2.cell2_ds_error_l2,
                                u_y_error_L2.cell2_ds_error_l2)
    vtk_utilities.export_solution_2(export_solution_path + '/p',
                                    mesh,
                                    p_on_cell0Ds.numeric_solution,
                                    p_on_cell0Ds.exact_solution,
                                    p_error_L2.cell2_ds_error_l2,
                                p_error_L2.cell2_ds_error_l2)
    other_ut.plot_solution(mesh, u_x_on_cell0Ds.numeric_solution, "u_x")
    other_ut.plot_solution(mesh, u_y_on_cell0Ds.numeric_solution, "u_y")
    other_ut.plot_solution(mesh, np.sqrt(u_x_on_cell0Ds.numeric_solution * u_x_on_cell0Ds.numeric_solution + u_y_on_cell0Ds.numeric_solution * u_y_on_cell0Ds.numeric_solution), "u_mag")
    other_ut.plot_solution(mesh, p_on_cell0Ds.numeric_solution, "p")