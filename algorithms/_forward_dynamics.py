def gen_forward_dynamics_inner_temp_mem_size(self):
        n = self.robot.get_num_pos()
        return self.gen_direct_minv_inner_temp_mem_size() + n*n

def gen_forward_dynamics_finish_function_call(self, use_thread_group = False, updated_var_names = None):
    var_names = dict( \
        s_qdd_name = "s_qdd", \
        s_u_name = "s_u", \
        s_c_name = "s_c", \
        s_Minv_name = "s_Minv"
    )
    if updated_var_names is not None:
        for key,value in updated_var_names.items():
            var_names[key] = value
    code = "forward_dynamics_finish<T>(" + var_names["s_qdd_name"] + ", " + var_names["s_u_name"] + ", " + \
                                           var_names["s_c_name"] + ", " + var_names["s_Minv_name"] + ");"
    if use_thread_group:
        code = code.replace("(","(tgrp, ")
    self.gen_add_code_line(code)

def gen_forward_dynamics_finish(self, use_thread_group = False):
    n = self.robot.get_num_pos()
    # construct the boilerplate and function definition
    func_params = ["s_qdd is a pointer to memory for the final result", \
                   "s_u is the vector of joint input torques", \
                   "s_c is the bias vector", \
                   "s_Minv is the inverse mass matrix"]
    func_def = "void forward_dynamics_finish(T *s_qdd, const T *s_u, const T *s_c, const T *s_Minv) {"
    func_notes = ["Assumes s_Minv and s_c are already computed"]
    if use_thread_group:
        func_def = func_def.replace("(","(cgrps::thread_group tgrp, ")
        func_params.insert(0,"tgrp is the handle to the thread_group running this function")
    self.gen_add_func_doc("Finish the forward dynamics computation with qdd = Minv*(u-c)",func_notes,func_params,None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)

    # compute the final answer qdd = Minv * (u - c)
    # remember that Minv is an SYMMETRIC_UPPER triangular matrix
    self.gen_add_parallel_loop("row",str(n),use_thread_group)
    self.gen_add_code_line("T val = static_cast<T>(0);")
    self.gen_add_code_line("for(int col = 0; col < " + str(n) + "; col++) {", True)
    self.gen_add_code_line("// account for the fact that Minv is an SYMMETRIC_UPPER triangular matrix")
    self.gen_add_code_line("int index = (row <= col) * (col * " + str(n) + " + row) + (row > col) * (row * " + str(n) + " + col);")
    self.gen_add_code_line("val += s_Minv[index] * (s_u[col] - s_c[col]);")
    self.gen_add_end_control_flow()
    self.gen_add_code_line("s_qdd[row] = val;")
    self.gen_add_end_control_flow()
    self.gen_add_end_function()

def gen_forward_dynamics_inner_function_call(self, use_thread_group = False, updated_var_names = None):
    var_names = dict( \
        s_q_name = "s_q", \
        s_qd_name = "s_qd", \
        s_qdd_name = "s_qdd", \
        s_u_name = "s_u", \
        s_temp_name = "s_temp", \
        gravity_name = "gravity"
    )
    if updated_var_names is not None:
        for key,value in updated_var_names.items():
            var_names[key] = value
    fd_code_start = "forward_dynamics_inner<T>(" + var_names["s_qdd_name"] + ", " + var_names["s_q_name"] + ", " + \
                                                   var_names["s_qd_name"] + ", " + var_names["s_u_name"] + ", "
    fd_code_end = var_names["s_temp_name"] + ", " + var_names["gravity_name"] + ");"
    fd_code_middle = self.gen_insert_helpers_function_call()
    if use_thread_group:
        fd_code_start = fd_code_start.replace("(","(tgrp, ")
    fd_code = fd_code_start + fd_code_middle + fd_code_end
    self.gen_add_code_line(fd_code)

def gen_forward_dynamics_inner(self, use_thread_group = False):
    n = self.robot.get_num_pos()
    # construct the boilerplate and function definition
    func_params = ["s_qdd is a pointer to memory for the final result", \
                   "s_q is the vector of joint positions", \
                   "s_qd is the vector of joint velocities", \
                   "s_u is the vector of joint input torques", \
                   "s_temp is the pointer to the shared memory needed of size: " + \
                            str(self.gen_forward_dynamics_inner_temp_mem_size()), \
                   "gravity is the gravity constant"]
    func_def_start = "void forward_dynamics_inner(T *s_qdd, const T *s_q, const T *s_qd, const T *s_u, "
    func_def_end = "T *s_temp, const T gravity) {"
    func_def_start, func_params = self.gen_insert_helpers_func_def_params(func_def_start, func_params, -2)
    func_notes = ["Assumes s_XImats is updated already for the current s_q"]
    if use_thread_group:
        func_def_start = func_def_start.replace("(","(cgrps::thread_group tgrp, ")
        func_params.insert(0,"tgrp is the handle to the thread_group running this function")
    func_def = func_def_start + func_def_end
    # then generate the code
    self.gen_add_func_doc("Computes forward dynamics",func_notes,func_params,None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")        
    self.gen_add_code_line(func_def, True)
    updated_var_names = dict(s_Minv_name = "s_temp", s_temp_name = "&s_temp[" + str(n*n) + "]")
    self.gen_direct_minv_inner_function_call(use_thread_group, updated_var_names)
    updated_var_names = dict(s_c_name = "&s_temp[" + str(n*n) + "]", s_vaf_name = "&s_temp[" + str(n*n + n) + "]", s_temp_name = "&s_temp[" + str(n*n + 19*n) + "]")
    self.gen_inverse_dynamics_inner_function_call(use_thread_group, compute_c = True, use_qdd_input = False, updated_var_names = updated_var_names)
    
    if self.DEBUG_MODE:
        self.gen_add_sync(use_thread_group)
        self.gen_add_serial_ops(use_thread_group)
        self.gen_add_code_lines(["printf(\"Minv\\n\"); printMat<T," + str(n) + "," + str(n) + ">(s_temp," + str(n) + ");",
                                 "printf(\"u\\n\"); printMat<T,1," + str(n) + ">(s_u,1);"
                                 "printf(\"c\\n\"); printMat<T,1," + str(n) + ">(&s_temp[" + str(n*n) + "],1);"])
        self.gen_add_end_control_flow()
        self.gen_add_sync(use_thread_group)

    # finally compute the final answer qdd = Minv * (u - c)
    updated_var_names = dict(s_Minv_name = "s_temp", s_c_name = "&s_temp[" + str(n*n) + "]")
    self.gen_forward_dynamics_finish_function_call(use_thread_group, updated_var_names)
    self.gen_add_end_function()

def gen_forward_dynamics_device(self, use_thread_group = False):
    n = self.robot.get_num_pos()
    # construct the boilerplate and function definition
    func_params = ["s_qdd is a pointer to memory for the final result", \
                   "s_q is the vector of joint positions", \
                   "s_qd is the vector of joint velocities", \
                   "s_u is the vector of joint input torques", \
                   "d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)", \
                   "gravity is the gravity constant"]
    func_def_start = "void forward_dynamics_device(T *s_qdd, const T *s_q, const T *s_qd, const T *s_u, "
    func_def_end = "const robotModel<T> *d_robotModel, const T gravity) {"
    func_notes = []
    if use_thread_group:
        func_def_start = func_def_start.replace("(","(cgrps::thread_group tgrp, ")
        func_params.insert(0,"tgrp is the handle to the thread_group running this function")
    func_def = func_def_start + func_def_end
    # then generate the code
    self.gen_add_func_doc("Computes forward dynamics",func_notes,func_params,None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)
    # add the shared memory variables
    shared_mem_size = self.gen_forward_dynamics_inner_temp_mem_size() if not self.use_dynamic_shared_mem_flag else None
    self.gen_XImats_helpers_temp_shared_memory_code(shared_mem_size)
    # then load/update XI and run the algo
    self.gen_load_update_XImats_helpers_function_call(use_thread_group)
    self.gen_forward_dynamics_inner_function_call(use_thread_group)
    self.gen_add_end_function()

def gen_forward_dynamics_kernel(self, use_thread_group = False, single_call_timing = False):
    n = self.robot.get_num_pos()
    # define function def and params
    func_params = ["d_qdd is a pointer to memory for the final result", \
                   "d_q_qd_u is the vector of joint positions, velocities, and input torques", \
                   "d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)", \
                   "gravity is the gravity constant", \
                   "num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)"]
    func_def_start = "void forward_dynamics_kernel(T *d_qdd, const T *d_q_qd_u, const int stride_q_qd_u, "
    func_def_end = "const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {"
    func_notes = []
    func_def = func_def_start + func_def_end
    # then generate the code
    if single_call_timing:
        func_def = func_def.replace("kernel(", "kernel_single_timing(")
    # then generate the code
    self.gen_add_func_doc("Computes forward dynamics",func_notes,func_params,None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__global__")
    self.gen_add_code_line(func_def, True)
    # add shared memory variables
    shared_mem_vars = ["__shared__ T s_q_qd_u[" + str(3*n) + "]; T *s_q = s_q_qd_u; T *s_qd = &s_q_qd_u[" + str(n) + "]; T *s_u = &s_q_qd_u[" + str(2*n) + "];", \
                       "__shared__ T s_qdd[" + str(n) + "];"]
    self.gen_add_code_lines(shared_mem_vars)
    shared_mem_size = self.gen_forward_dynamics_inner_temp_mem_size() if not self.use_dynamic_shared_mem_flag else None
    self.gen_XImats_helpers_temp_shared_memory_code(shared_mem_size)
    if use_thread_group:
        self.gen_add_code_line("cgrps::thread_group tgrp = TBD;")
    if not single_call_timing:
        # load to shared mem and loop over blocks to compute all requested comps
        self.gen_add_parallel_loop("k","NUM_TIMESTEPS",use_thread_group,block_level = True)
        self.gen_kernel_load_inputs("q_qd_u","stride_q_qd_u",str(3*n),use_thread_group)
        # compute
        self.gen_add_code_line("// compute")
        self.gen_load_update_XImats_helpers_function_call(use_thread_group)
        self.gen_forward_dynamics_inner_function_call(use_thread_group)
        self.gen_add_sync(use_thread_group)
        # save to global
        self.gen_kernel_save_result("qdd",str(n),str(n),use_thread_group)
        self.gen_add_end_control_flow()
    else:
        #repurpose NUM_TIMESTEPS for number of timing reps
        self.gen_kernel_load_inputs_single_timing("q_qd_u",str(3*n))
        # then compute in loop for timing
        self.gen_add_code_line("// compute with NUM_TIMESTEPS as NUM_REPS for timing")
        self.gen_add_code_line("for (int rep = 0; rep < NUM_TIMESTEPS; rep++){", True)
        self.gen_load_update_XImats_helpers_function_call(use_thread_group)
        self.gen_forward_dynamics_inner_function_call(use_thread_group)
        self.gen_add_end_control_flow()
        # save to global
        self.gen_kernel_save_result_single_timing("qdd",str(n),use_thread_group)
    self.gen_add_end_function()

def gen_forward_dynamics_host(self, mode = 0):
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
    func_def_start = "void forward_dynamics(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,"
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
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__host__")
    self.gen_add_code_line(func_def_start)
    self.gen_add_code_line(func_def_end, True)
    func_call_start = "forward_dynamics_kernel<T><<<block_dimms,thread_dimms,FD_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_qdd,hd_data->d_q_qd_u,stride_q_qd_u,"
    func_call_end = "d_robotModel,gravity,num_timesteps);"
    if single_call_timing:
        func_call_start = func_call_start.replace("kernel<T>","kernel_single_timing<T>")
    self.gen_add_code_line("int stride_q_qd_u = 3*NUM_JOINTS;")
    if not compute_only:
        # start code with memory transfer
        self.gen_add_code_lines(["// start code with memory transfer", \
                                 "gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd_u*" + \
                                    ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyHostToDevice,streams[0]));", \
                                 "gpuErrchk(cudaDeviceSynchronize());"])
    # then compute
    self.gen_add_code_line("// then call the kernel")
    func_call = func_call_start + func_call_end
    func_call_code = [func_call, "gpuErrchk(cudaDeviceSynchronize());"]
    # wrap function call in timing (if needed)
    if single_call_timing:
        func_call_code.insert(0,"struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);")
        func_call_code.append("clock_gettime(CLOCK_MONOTONIC,&end);")
    self.gen_add_code_lines(func_call_code)
    if not compute_only:
        # then transfer memory back
        self.gen_add_code_lines(["// finally transfer the result back", \
                                 "gpuErrchk(cudaMemcpy(hd_data->h_qdd,hd_data->d_qdd,NUM_JOINTS*" + \
                                    ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyDeviceToHost));",
                                 "gpuErrchk(cudaDeviceSynchronize());"])
    # finally report out timing if requested
    if single_call_timing:
        self.gen_add_code_line("printf(\"Single Call FD %fus\\n\",time_delta_us_timespec(start,end)/static_cast<double>(num_timesteps));")
    self.gen_add_end_function()

def gen_forward_dynamics(self, use_thread_group = False):
    # first helpers
    self.gen_forward_dynamics_finish(use_thread_group)
    self.gen_forward_dynamics_inner(use_thread_group)
    # then device wrapper
    self.gen_forward_dynamics_device(use_thread_group)
    # then kernels
    self.gen_forward_dynamics_kernel(use_thread_group,True)
    self.gen_forward_dynamics_kernel(use_thread_group,False)
    # then host launch
    self.gen_forward_dynamics_host(0)
    self.gen_forward_dynamics_host(1)
    self.gen_forward_dynamics_host(2)