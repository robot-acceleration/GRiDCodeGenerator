def gen_forward_dynamics_gradient_inner_temp_mem_size(self, use_qdd_Minv_input = False):
    n = self.robot.get_num_pos()
    minv_temp = self.gen_direct_minv_inner_temp_mem_size()
    id_du_temp = self.gen_inverse_dynamics_gradient_inner_temp_mem_size()
    return max(minv_temp,id_du_temp) if not use_qdd_Minv_input else id_du_temp

def gen_forward_dynamics_gradient_inner_python(self, use_thread_group = False, use_qdd_Minv_input = False, s_df_du_name = "s_df_du"):
    n = self.robot.get_num_pos()
    if not use_qdd_Minv_input:
        #
        # TODO: there is a slightly faster way as s_v does not change -- thus no recompute needed
        #       but that requires a custom function to be written
        #
        self.gen_add_code_line("//TODO: there is a slightly faster way as s_v does not change -- thus no recompute needed")
        self.gen_direct_minv_inner_function_call(use_thread_group)
        # updated_var_names = dict(s_c_name = "s_temp", s_vaf_name = "&s_temp[" + str(n) + "]", s_temp_name = "&s_temp[" + str(19*n) + "]")
        updated_var_names = dict(s_c_name = "s_temp", s_temp_name = "&s_temp[" + str(n) + "]")
        self.gen_inverse_dynamics_inner_function_call(use_thread_group, compute_c = True, use_qdd_input = False, updated_var_names = updated_var_names)
        self.gen_forward_dynamics_finish_function_call(use_thread_group, updated_var_names)
        self.gen_inverse_dynamics_inner_function_call(use_thread_group, compute_c = False, use_qdd_input = True)
    # else just compute vaf
    else:
        self.gen_inverse_dynamics_inner_function_call(use_thread_group, compute_c = False, use_qdd_input = True)
    # then run the gradient code
    self.gen_inverse_dynamics_gradient_inner_function_call(use_thread_group)

    if self.DEBUG_MODE:
        self.gen_add_sync(use_thread_group)
        self.gen_add_serial_ops(use_thread_group)
        self.gen_add_code_lines(["printf(\"Minv\\n\");", \
                                 "printMat<T," + str(n) + "," + str(n) + ">(s_Minv," + str(n) + ");", \
                                 "printf(\"qdd\\n\");", \
                                 "printMat<T,1," + str(n) + ">(s_qdd,1);", \
                                 "printf(\"v\\n\");", \
                                 "printMat<T,6," + str(n) + ">(s_vaf,6);", \
                                 "printf(\"a\\n\");", \
                                 "printMat<T,6," + str(n) + ">(&s_vaf[6*" + str(n) + "],6);", \
                                 "printf(\"f\\n\");", \
                                 "printMat<T,6," + str(n) + ">(&s_vaf[12*" + str(n) + "],6);", \
                                 "printf(\"dc/dq\\n\");", \
                                 "printMat<T," + str(n) + "," + str(n) + ">(&s_dc_du[0]," + str(n) + ");", \
                                 "printf(\"dc/dqd\\n\");", \
                                 "printMat<T," + str(n) + "," + str(n) + ">(&s_dc_du[" + str(n*n) + "]," + str(n) + ");"])
        self.gen_add_end_control_flow()
        self.gen_add_sync(use_thread_group)

    # and finally finish with df/du = -Minv*dc/du
    self.gen_add_parallel_loop("ind",str(n*2*n),use_thread_group)
    self.gen_add_code_line("int row = ind % " + str(n) + "; int dc_col_offset = ind - row;")
    self.gen_add_code_line("// account for the fact that Minv is an SYMMETRIC_UPPER triangular matrix")
    self.gen_add_code_line("T val = static_cast<T>(0);")
    self.gen_add_code_line("for(int col = 0; col < " + str(n) + "; col++) {", True)
    self.gen_add_code_line("int index = (row <= col) * (col * " + str(n) + " + row) + (row > col) * (row * " + str(n) + " + col);")
    self.gen_add_code_line("val += s_Minv[index] * s_dc_du[dc_col_offset + col];")
    self.gen_add_end_control_flow()
    self.gen_add_code_line(s_df_du_name + "[ind] = -val;")
    self.gen_add_end_control_flow()

def gen_forward_dynamics_gradient_device(self, use_thread_group = False, use_qdd_Minv_input = False):
    n = self.robot.get_num_pos()

    # construct the boilerplate and function definition
    func_params = ["s_df_du is a pointer to memory for the final result of size 2*NUM_JOINTS*NUM_JOINTS = " + str(2*n*n), \
                   "s_q is the vector of joint positions", \
                   "s_qd is the vector of joint velocities", \
                   "d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)", \
                   "gravity is the gravity constant"]
    func_def_start = "void forward_dynamics_gradient_device(T *s_df_du, const T *s_q, const T *s_qd, "
    func_def_end = "const robotModel<T> *d_robotModel, const T gravity) {"
    func_notes = ["Uses the fd/du = -Minv*id/du trick as described in Carpentier and Mansrud 'Analytical Derivatives of Rigid Body Dynamics Algorithms'"]
    if use_thread_group:
        func_def_start += "cgrps::thread_group tgrp, "
        func_params.insert(0,"tgrp is the handle to the thread_group running this function")
    if use_qdd_Minv_input:
        func_def_start += "const T *s_qdd, "
        func_params.insert(-2,"s_qdd is the vector of joint accelerations")
        func_def_start += "const T *s_Minv, "
        func_params.insert(-2,"s_Minv is the mass matrix")
    else:
        func_def_start += "const T *s_u, "
        func_params.insert(-2,"s_u is the vector of input torques")
    func_def = func_def_start + func_def_end
    self.gen_add_func_doc("Computes the gradient of forward dynamics",func_notes,func_params,None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)
    # add the shared memory variables
    self.gen_add_code_lines(["__shared__ T s_vaf[" + str(18*n) + "];",
                             "__shared__ T s_dc_du[" + str(n*2*n) + "];"])
    if not use_qdd_Minv_input:
        self.gen_add_code_lines(["__shared__ T s_Minv[" + str(n*n) + "];",
                                 "__shared__ T s_qdd[" + str(n) + "];"])
    shared_mem_size = self.gen_forward_dynamics_gradient_inner_temp_mem_size() if not self.use_dynamic_shared_mem_flag else None
    self.gen_XImats_helpers_temp_shared_memory_code(shared_mem_size)
    # then load/update XI and run the algo
    self.gen_load_update_XImats_helpers_function_call(use_thread_group)
    # then run the computation
    self.gen_forward_dynamics_gradient_inner_python(use_thread_group,use_qdd_Minv_input)
    self.gen_add_end_function()

def gen_forward_dynamics_gradient_kernel_max_temp_mem_size(self):
    n = self.robot.get_num_pos()
    base_size = 2*n + n*2*n + n*2*n + 18*n + n + n*n + n
    temp_mem_size = self.gen_forward_dynamics_gradient_inner_temp_mem_size()
    return base_size + temp_mem_size

def gen_forward_dynamics_gradient_kernel(self, use_thread_group = False, use_qdd_Minv_input = False, single_call_timing = False):
    n = self.robot.get_num_pos()
    # define function def and params
    func_params = ["d_df_du is a pointer to memory for the final result of size 2*NUM_JOINTS*NUM_JOINTS = " + str(2*n*n), \
                   "d_q_dq is the vector of joint positions and velocities", \
                   "stride_q_qd is the stide between each q, qd", \
                   "d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)", \
                   "gravity is the gravity constant", \
                   "num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)"]
    func_notes = []
    func_def_start = "void forward_dynamics_gradient_kernel(T *d_df_du, const T *d_q_qd, const int stride_q_qd, "
    func_def_end = "const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {"
    if use_qdd_Minv_input:
        func_def_start += "const T *d_qdd, "
        func_params.insert(-2,"d_qdd is the vector of joint accelerations")
        func_def_start += "const T *d_Minv, "
        func_params.insert(-2,"d_Minv is the mass matrix")
    else:
        func_def_start = func_def_start.replace("_q_qd","_q_qd_u")
        func_params[1] = "d_q_dq is the vector of joint positions, velocities, and input torques"
        func_params[2] = "stride_q_qd_u is the stide between each q, qd, u"
    func_def = func_def_start + func_def_end
    if single_call_timing:
        func_def = func_def.replace("kernel(", "kernel_single_timing(")
    # then generate the code
    self.gen_add_func_doc("Computes the gradient of forward dynamics",func_notes,func_params,None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__global__")
    self.gen_add_code_line(func_def, True)
    # add shared memory variables
    shared_mem_vars = ["__shared__ T s_q_qd[2*" + str(n) + "]; T *s_q = s_q_qd; T *s_qd = &s_q_qd[" + str(n) + "];", \
                       "__shared__ T s_dc_du[" + str(n*2*n) + "];",
                       "__shared__ T s_vaf[" + str(18*n) + "];",
                       "__shared__ T s_qdd[" + str(n) + "];",
                       "__shared__ T s_Minv[" + str(n*n) + "];"]
    if not use_qdd_Minv_input:
        shared_mem_vars[0] = "__shared__ T s_q_qd_u[3*" + str(n) + "]; T *s_q = s_q_qd_u; T *s_qd = &s_q_qd_u[" + str(n) + "]; T *s_u = &s_q_qd_u[" + str(2*n) + "];"
    self.gen_add_code_lines(shared_mem_vars)
    shared_mem_size = self.gen_forward_dynamics_gradient_inner_temp_mem_size() if not self.use_dynamic_shared_mem_flag else None
    self.gen_XImats_helpers_temp_shared_memory_code(shared_mem_size)
    if use_thread_group:
        self.gen_add_code_line("cgrps::thread_group tgrp = TBD;")
    if not single_call_timing:
        # load to shared mem and loop over blocks to compute all requested comps
        self.gen_add_parallel_loop("k","NUM_TIMESTEPS",use_thread_group,block_level = True)
        if use_qdd_Minv_input:
            self.gen_kernel_load_inputs("q_qd","stride_q_qd",str(2*n),use_thread_group,"qdd",str(n),str(n),"Minv",str(n*n),str(n*n))
        else:
            self.gen_kernel_load_inputs("q_qd_u","stride_q_qd_u",str(3*n),use_thread_group)
        # compute
        self.gen_add_code_line("// compute")
        self.gen_load_update_XImats_helpers_function_call(use_thread_group)
        self.gen_forward_dynamics_gradient_inner_python(use_thread_group,use_qdd_Minv_input,"s_temp") # use the temp mem to store s_df_du
        # save to global
        self.gen_kernel_save_result("df_du",str(n*2*n),str(n*2*n),use_thread_group,"s_temp")
        self.gen_add_end_control_flow()
    else:
        #repurpose NUM_TIMESTEPS for number of timing reps
        if use_qdd_Minv_input:
            self.gen_kernel_load_inputs_single_timing("q_qd",str(2*n),use_thread_group,"qdd",str(n),"Minv",str(n*n))
        else:
            self.gen_kernel_load_inputs_single_timing("q_qd_u",str(3*n),use_thread_group)
        # then compute in loop for timing
        self.gen_add_code_line("// compute with NUM_TIMESTEPS as NUM_REPS for timing")
        self.gen_add_code_line("for (int rep = 0; rep < NUM_TIMESTEPS; rep++){", True)
        self.gen_load_update_XImats_helpers_function_call(use_thread_group)
        self.gen_forward_dynamics_gradient_inner_python(use_thread_group,use_qdd_Minv_input,"s_temp") # use the temp mem to store s_df_du
        self.gen_add_end_control_flow()
        # save to global
        self.gen_kernel_save_result_single_timing("df_du",str(n*2*n),use_thread_group,"s_temp")
    self.gen_add_end_function()

def gen_forward_dynamics_gradient_host(self, mode = 0):
    # default is to do the full kernel call -- options are for single timing or compute only kernel wrapper
    single_call_timing = True if mode == 1 else False
    compute_only = True if mode == 2 else False

    # define function def and params
    func_params = ["hd_data is the packaged input and output pointers", \
                   "d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)", \
                   "gravity is the gravity constant,", \
                   "num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)", \
                   "streams are pointers to CUDA streams for async memory transfers (if needed)"]
    func_notes = []
    func_def_start = "void forward_dynamics_gradient(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,"
    func_def_end =   "                      const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {"
    if single_call_timing:
        func_def_start = func_def_start.replace("(", "_single_timing(")
        func_def_end = "              " + func_def_end
    if compute_only:
        func_def_start = func_def_start.replace("(", "_compute_only(")
        func_def_end = "             " + func_def_end.replace(", cudaStream_t *streams", "")
    # then generate the code
    self.gen_add_func_doc("Compute the RNEA (Recursive Newton-Euler Algorithm)",\
                          func_notes,func_params,None)
    self.gen_add_code_line("template <typename T, bool USE_QDD_MINV_FLAG = false>")
    self.gen_add_code_line("__host__")
    self.gen_add_code_line(func_def_start)
    self.gen_add_code_line(func_def_end, True)
    func_call_start = "forward_dynamics_gradient_kernel<T><<<block_dimms,thread_dimms,FD_DU_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_df_du,hd_data->d_q_qd_u,stride_q_qd,"
    func_call_end = "d_robotModel,gravity,num_timesteps);"
    if single_call_timing:
        func_call_start = func_call_start.replace("kernel<T>","kernel_single_timing<T>")
    self.gen_add_code_line("int stride_q_qd= 3*NUM_JOINTS;")
    if not compute_only:
        # start code with memory transfer
        self.gen_add_code_lines(["// start code with memory transfer", \
                                 "gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd*" + \
                                    ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyHostToDevice,streams[0]));", \
                                 "if (USE_QDD_MINV_FLAG) {" ,\
                                 "    gpuErrchk(cudaMemcpyAsync(hd_data->d_qdd,hd_data->h_qdd,NUM_JOINTS*" + \
                                        ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyHostToDevice,streams[1]));", \
                                 "    gpuErrchk(cudaMemcpyAsync(hd_data->d_Minv,hd_data->h_Minv,NUM_JOINTS*NUM_JOINTS*" + \
                                        ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyHostToDevice,streams[2]));", \
                                 "}", \
                                 "gpuErrchk(cudaDeviceSynchronize());"])
    # then compute
    self.gen_add_code_line("// then call the kernel")
    func_call = func_call_start + func_call_end
    func_call_with_qdd_minv = func_call_start + "hd_data->d_qdd, hd_data->d_Minv, " + func_call_end
    func_call_code = ["if (USE_QDD_MINV_FLAG) {" + func_call_with_qdd_minv + "}", "else {" + func_call + "}", "gpuErrchk(cudaDeviceSynchronize());"]
    # wrap function call in timing (if needed)
    if single_call_timing:
        func_call_code.insert(0,"struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);")
        func_call_code.append("clock_gettime(CLOCK_MONOTONIC,&end);")
    self.gen_add_code_lines(func_call_code)
    if not compute_only:
        # then transfer memory back
        self.gen_add_code_lines(["// finally transfer the result back", \
                                 "gpuErrchk(cudaMemcpy(hd_data->h_df_du,hd_data->d_df_du,NUM_JOINTS*2*NUM_JOINTS*" + \
                                    ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyDeviceToHost));",
                                 "gpuErrchk(cudaDeviceSynchronize());"])
    # finally report out timing if requested
    if single_call_timing:
        self.gen_add_code_line("printf(\"Single Call FD_DU %fus\\n\",time_delta_us_timespec(start,end)/static_cast<double>(num_timesteps));")
    self.gen_add_end_function()

def gen_forward_dynamics_gradient(self, use_thread_group = False):
    # first device wrappers
    self.gen_forward_dynamics_gradient_device(use_thread_group,False)
    self.gen_forward_dynamics_gradient_device(use_thread_group,True)
    # then kernels
    self.gen_forward_dynamics_gradient_kernel(use_thread_group,True,True)
    self.gen_forward_dynamics_gradient_kernel(use_thread_group,True,False)
    self.gen_forward_dynamics_gradient_kernel(use_thread_group,False,True)
    self.gen_forward_dynamics_gradient_kernel(use_thread_group,False,False)
    # finally host wrappers
    self.gen_forward_dynamics_gradient_host(0)
    self.gen_forward_dynamics_gradient_host(1)
    self.gen_forward_dynamics_gradient_host(2)