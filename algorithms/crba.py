import numpy as np
import copy
#np.set_printoptions(precision=4, suppress=True, linewidth = 100)

def gen_crba_inner_temp_mem_size(self):
    n = self.robot.get_num_pos()
    return 6*n

def gen_crba_inner_function_call(self, use_thread_group = False, updated_var_names = None):
    var_names = dict( \
        s_H = "s_H", \
        #s_c_name = "s_c", \
        s_q_name = "s_q", \
        s_qd_name = "s_qd", \
        s_tau = "tau", \
        s_XI = "s_XI", \
        s_temp_name = "s_temp", \
        gravity_name = "gravity"
    )
    #s_XI calculated in device and allocated in kernel 
    #s_H, temp, q, qd allocated in kernel
    #gravity allocated in host 
    #where is tau allocated?  
     
    if updated_var_names is not None:
        for key,value in updated_var_names.items():
            var_names[key] = value
    
    id_code = "crba_inner<T>(" + var_names["s_q_name"] + ", " + var_names["s_qd_name"] + ", " + var_names["s_tau"] + ", " + var_names["s_temp_name"] + ", " + var_names["gravity_name"] + ")"

    #what happens if use_thread_group = True

    #self.gen_add_code_line(id_code)


def gen_crba_inner(self, use_thread_group = False):
    
    n = self.robot.get_num_pos()
    n_bfs_levels = self.robot.get_max_bfs_level() + 1

    #construct the boilerplate and function definition
    func_params = [ "s_H is a pointer to shared memory of size NUM_JOINTS*NUM_JOINTS = " + str(n*n), \
                    "s_q is the vector of joint positions", \
                    "s_qd is the vector of joint velocities", \
                    "s_XI is the pointer to the transformation and inertia matricies ", \
                    "s_IC is the pointer to the inertia and force vector matricies", \
                    "s_tau is the pointer to the torque", \
                    "s_temp is a pointer to helper shared memory of size 6*NUM_JOINTS = " + \
                            str(self.gen_crba_inner_temp_mem_size())]
    func_notes = [] #insert notes abt function 
    func_def_start = "void crba_inner(const T *s_q, const T *s_qd, const T *s_tau, "
    func_def_end = "T *s_temp) {"
    if use_thread_group:
        func_def_start = func_def_start.replace("(", "(cgrps::thread_group tgrp, ")
        func_params.insert(0,"tgrp is the handle to the thread_group running this function")

    #insert helpers/other parameters?
    func_def = func_def_start + func_def_end
    self.gen_add_func_doc("Compute the Composite Rigid Body Algorithm", func_notes, func_params, None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)

    
    #deal with like memory for variables --> memory is taken care of in device and host 
    fh_offset = 0
    j_offset = fh_offset + 6*6 #bc fh is 6*6 ints
    s_offset = j_offset + 1 #bc j is 1 int 
    ind_offset = s_offset + 6 #bc S is array of 6 ints
    parent_offset = ind_offset + 1 #bc parent_ind_cpp is 1 int 
    sval_offset = parent_offset + 1 #bc S_ind_cpp is 1 int 

    self.gen_add_code_line("s_fh = $s_temp[" + str(fh_offset) + "];")
    self.gen_add_code_line("s_j = $s_temp[" + str(j_offset) + "];")
    self.gen_add_code_line("s_S = $s_temp[" + str(s_offset) + "];")
    self.gen_add_code_line("ind = $s_temp[" + str(ind_offset) + "];")
    self.gen_add_code_line("parent_ind_cpp = $s_temp[" + str(parent_offset) + "];")
    self.gen_add_code_line("S_ind_cpp = $s_temp[" + str(sval_offset) + "];")

    x_offset = 0
    ic_offset = x_offset + 6*6

    self.gen_add_code_line("s_X = $s_XI[" + str(x_offset) + "];")
    self.gen_add_code_line("s_IC = $s_XI[" + str(ic_offset) + "];")
    
    self.gen_add_code_line("//")
    self.gen_add_code_line("// first loop ")
    self.gen_add_code_line("//")
    self.gen_add_code_line("// each bfs level runs in parallel")
 
    for bfs_level in range(n_bfs_levels-1,0,-1):
        inds = self.robot.get_ids_by_bfs_level(bfs_level)
        ind = str(inds)
        ind = ind[1]
        #self.gen_add_code_line("ind = " + ind)
        #self.gen_add_code_line("inds testing = " + str(inds))
        joint_names = [self.robot.get_joint_by_id(indj).get_name() for indj in inds]
        link_names = [self.robot.get_link_by_id(indl).get_name() for indl in inds]

        self.gen_add_code_line("// pass updates where bfs_level is " + str(bfs_level))
        self.gen_add_code_line("//     joints are: " + ", ".join(joint_names))
        self.gen_add_code_line("//     links are: " + ", ".join(link_names))

        parent_ind_cpp, S_ind_cpp = self.gen_topology_helpers_pointers_for_cpp(inds, NO_GRAD_FLAG = True)
        #self.gen_add_code_line("// S_ind_cpp = " + S_ind_cpp)

        self.gen_add_parallel_loop("jid",str(6*len(inds)),use_thread_group)
        #row = ind % 6   
        #self.gen_add_code_line("rowwww = " + str(row))
        self.gen_add_code_line("int row = ind % 6;")
        if len(inds) > 1:
            select_var_vals = [("int", "jid", [str(jid) for jid in inds])]
            self.gen_add_multi_threaded_select("ind", "<", [str(6*(i+1)) for i in range(len(inds))], select_var_vals)
            jid = "jid"
        else:
            jid = str(inds[0])

        self.gen_add_code_line("int jid6 = 6*" + jid + ";")              
        self.gen_add_code_line("int ind = " + ind + ";")

        #parent_ind = self.robot.get_parent_id(ind)
        comment = "// parent_ind = self.robot.get_parent_id(ind)"
        self.gen_add_code_line(comment)
        self.gen_add_code_line("int parent_ind = " + parent_ind_cpp + ";")
        
        #Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])
        comment = "// Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind]) --> as param so don't need to init it now" 
        self.gen_add_code_line(comment)
        #self.gen_add_code_line("s_Xmat = *s_X;")
    
       #IC[parent_ind] = IC[parent_ind] + np.matmul(Xmat.T@IC[ind],Xmat)
        comment = "// IC[parent_ind] = IC[parent_ind] + (Xmat.T)@IC[ind]@Xmat" 
        self.gen_add_code_line(comment)
        self.gen_add_code_line("&s_IC[" + parent_ind_cpp + "] = &s_IC[" + parent_ind_cpp + "] + dot_prod<T,6,6,1>(s_X[6*jid6 + row], dot_prod<T,6,6,1>(&s_IC[ind],s_X));") 
        self.gen_add_end_control_flow()

    self.gen_add_sync(use_thread_group) 
    
    for ind in range(n-1, -1, -1): # in parallel
        # Calculation of fh and H[ind, ind]
        _, S_ind_cpp = self.gen_topology_helpers_pointers_for_cpp(NO_GRAD_FLAG = True)

        self.gen_add_code_line("//")
        self.gen_add_code_line("// Calculation of fh and H[ind, ind] (second loop)")
        self.gen_add_code_line("//")
        self.gen_add_parallel_loop("jid",str(n),use_thread_group)

        #S = self.robot.get_S_by_id(ind)
        comment = "// S = self.robot.get_S_by_id(ind)" 
        self.gen_add_code_line(comment)
        s_S = np.zeros(6)
        for i in range(6):
            if i == int(S_ind_cpp):
                s_S[i] += 1
        self.gen_add_code_line("s_S = " + str(s_S) + ";")

        #fh = np.matmul(IC[ind], S)
        comment = "// fh = np.matmul(IC[ind], S)" 
        self.gen_add_code_line(comment)
        self.gen_add_code_line("s_fh = dot_prod<T,6,6,1>(&s_IC[ind], &s_S);")

        #H[ind, ind] = np.matmul(S, fh)
        comment = "// H[ind, ind] = np.matmul(S, fh))" 
        self.gen_add_code_line(comment)
        self.gen_add_code_line("&s_H[ind][ind] = dot_prod<T,6,6,1>(&s_S, &s_fh);")

        self.gen_add_end_control_flow()

    self.gen_add_sync(use_thread_group)

    for ind in range(n-1, -1, -1): # in parallel
        # Calculation of H[ind, j] and H[j, ind]
        self.gen_add_code_line("//")
        self.gen_add_code_line("// Calculation of H[ind, j] and H[j, ind] (third loop) ")
        self.gen_add_code_line("//")
        self.gen_add_parallel_loop("jid",str(n),use_thread_group)
        self.gen_add_code_line("int jid6 = 6*jid;")

        #S = self.robot.get_S_by_id(ind)
        comment = "// S = self.robot.get_S_by_id(ind)" 
        self.gen_add_code_line(comment)
        s_S = np.zeros(6)
        for i in range(6):
            if i == int(S_ind_cpp):
                s_S[i] = 1
        self.gen_add_code_line("s_S = " + str(s_S) + ";")

        #fh = np.matmul(IC[ind], S)
        comment = "// fh = np.matmul(IC[ind], S)" 
        self.gen_add_code_line(comment)
        self.gen_add_code_line("&s_fh = dot_prod<T,6,6,1>(&s_IC[ind], &s_S);")

        #j = ind
        comment = "// s_j = jid" 
        self.gen_add_code_line(comment)
        init_j = "int s_j = jid;"
        self.gen_add_code_line(init_j)
        j = ind

        #while loop format
        #while self.robot.get_parent_id(ind) > -1:
        if self.robot.get_parent_id(ind) > -1:
            loop = "while(self.robot.get_parent_id(ind) > -1) {"
            self.gen_add_code_line(loop)
            self.gen_add_code_line("    int row = ind % 6;")
            #row = ind % 6
            #self.gen_add_code_line("rowwww in if = " + str(row))
            #Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])
            comment = "    // Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind]) --> as param so don't need to init it now" 
            self.gen_add_code_line(comment)
            #self.gen_add_code_line("    s_Xmat = *s_X;")
    
            #fh = np.matmul(Xmat.T, fh)
            comment = "    // fh = np.matmul(Xmat.T, fh)" 
            self.gen_add_code_line(comment)
            self.gen_add_code_line("    &s_fh = dot_prod<T,6,6,1>(s_X[6*jid6 + row], &s_fh);")

            #j = self.robot.get_parent_id(j)
            comment = "    // j = self.robot.get_parent_id(j)" 
            self.gen_add_code_line(comment)
            j_list = [j]
            #self.gen_add_code_line("j_list = " + str(j_list))
            j_parent_ind_cpp, j_S_ind_cpp = self.gen_topology_helpers_pointers_for_cpp(j_list, NO_GRAD_FLAG = True)
            init_j = "    j = " + j_parent_ind_cpp + ";"
            self.gen_add_code_line(init_j)

            #S = self.robot.get_S_by_id(j)
            comment = "    // S = self.robot.get_S_by_id(j)" 
            self.gen_add_code_line(comment)
            s_S = np.zeros(6)
            for i in range(6):
                if i == int(j_S_ind_cpp):
                    s_S[i] = 1
            self.gen_add_code_line("    s_S = " + str(s_S) + ";")

            #H[ind, j] = np.matmul(S.T, fh)
            comment = "    // H[ind, j] = np.matmul(S.T, fh)" 
            self.gen_add_code_line(comment)
            self.gen_add_code_line("    &s_H[ind,j] = dot_prod<T,6,6,1>(s_S[6*jid6 + row], &s_fh);")

            #H[j, ind] = H[ind, j]
            comment = "    // H[j, ind] = H[ind, j]" 
            self.gen_add_code_line(comment)
            self.gen_add_code_line("    &s_H[j,ind] = &s_H[ind,j];")

            self.gen_add_code_line("}")

        self.gen_add_end_control_flow()
  
    self.gen_add_sync(use_thread_group)
    self.gen_add_code_line("return &s_H;") 
    self.gen_add_end_function()

def gen_crba_device_temp_mem_size(self):
    n = self.robot.get_num_pos()
    wrapper_size = self.gen_topology_helpers_size() + 72*n # for XImats
    return self.gen_crba_inner_temp_mem_size() + wrapper_size

def gen_crba_device(self, use_thread_group = False):
    n = self.robot.get_num_pos()

    # construct the boilerplate and function definition
    func_params = ["s_q is the vector of joint positions", \
                   "s_qd is the vector of joint velocities", \
                   "s_tau is the pointer to the torque", \
                   "d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)", \
                   "gravity is the gravity constant"]
    func_notes = []
    func_def_start = "void crba_device("
    func_def_middle = "const T *s_q, const T *s_qd, const T *s_tau,"
    func_def_end = "const robotModel<T> *d_robotModel, const T gravity) {"
    if use_thread_group:
        func_def_start += "cgrps::thread_group tgrp, "
        func_params.insert(0,"tgrp is the handle to the thread_group running this function")

    func_def = func_def_start + func_def_middle + func_def_end

    # then generate the code
    self.gen_add_func_doc("Compute the CRBA (Composite Rigid Body Algorithm)",\
                          func_notes,func_params,None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)

    # add the shared memory variables
    shared_mem_size = self.gen_crba_inner_temp_mem_size() if not self.use_dynamic_shared_mem_flag else None 
    self.gen_XImats_helpers_temp_shared_memory_code(shared_mem_size)

    # then load/update XI and run the algo
    self.gen_load_update_XImats_helpers_function_call(use_thread_group)
    self.gen_crba_inner_function_call(use_thread_group)
    self.gen_add_end_function()

def gen_crba_kernel(self, use_thread_group = False, single_call_timing = False):
    n = self.robot.get_num_pos()
    # define function def and params
    func_params = ["d_H is the matrix of output Inertia", \
                   "d_q_dq is the vector of joint positions and velocities", \
                   "stride_q_qd is the stride between each q, qd", \
                   "d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)", \
                   "gravity is the gravity constant,", \
                   "num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)"]
    func_notes = []
    func_def_start = "void crba_kernel(T *d_H, const T *d_q_qd, const int stride_q_qd, "
    func_def_end = "const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {"
    func_def = func_def_start + func_def_end
    if single_call_timing:
        func_def = func_def.replace("kernel(", "kernel_single_timing(")
    # then generate the code
    self.gen_add_func_doc("Compute the CRBA (Composite Rigid Body Algorithm)",\
                          func_notes,func_params,None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__global__")
    self.gen_add_code_line(func_def, True)
    # add shared memory variables
    shared_mem_vars = ["__shared__ T s_q_qd[2*" + str(n) + "]; T *s_q = s_q_qd; T *s_qd = &s_q_qd[" + str(n) + "];", \
                       "__shared__ T s_H[" + str(6*n) + "];"]
    self.gen_add_code_lines(shared_mem_vars)
    shared_mem_size = self.gen_crba_inner_temp_mem_size() if not self.use_dynamic_shared_mem_flag else None
    self.gen_XImats_helpers_temp_shared_memory_code(shared_mem_size)
    if use_thread_group:
        self.gen_add_code_line("cgrps::thread_group tgrp = TBD;")
    if not single_call_timing:
        # load to shared mem and loop over blocks to compute all requested comps
        self.gen_add_parallel_loop("k","NUM_TIMESTEPS",use_thread_group,block_level = True)
        self.gen_kernel_load_inputs("q_qd","stride_q_qd",str(2*n),use_thread_group)
        # compute
        self.gen_add_code_line("// compute")
        self.gen_load_update_XImats_helpers_function_call(use_thread_group)
        self.gen_crba_inner_function_call(use_thread_group)
        self.gen_add_sync(use_thread_group)
        # save to global
        self.gen_kernel_save_result("H",str(6*n),str(6*n),use_thread_group)
        self.gen_add_end_control_flow()
    else:
        #repurpose NUM_TIMESTEPS for number of timing reps
        self.gen_kernel_load_inputs_single_timing("q_qd",str(2*n),use_thread_group)
        # then compute in loop for timing
        self.gen_add_code_line("// compute with NUM_TIMESTEPS as NUM_REPS for timing")
        self.gen_add_code_line("for (int rep = 0; rep < NUM_TIMESTEPS; rep++){", True)
        self.gen_load_update_XImats_helpers_function_call(use_thread_group)
        self.gen_crba_inner_function_call(use_thread_group)
        self.gen_add_end_control_flow()
        # save to global
        self.gen_kernel_save_result_single_timing("H",str(6*n),use_thread_group)
    self.gen_add_end_function()

def gen_crba_host(self, mode = 0):
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
    func_def_start = "void crba(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,"
    func_def_end =   "                      const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {"
    if single_call_timing:
        func_def_start = func_def_start.replace("(", "_single_timing(")
        func_def_end = "              " + func_def_end
    if compute_only:
        func_def_start = func_def_start.replace("(", "_compute_only(")
        func_def_end = "             " + func_def_end.replace(", cudaStream_t *streams", "")
    # then generate the code
    self.gen_add_func_doc("Compute the CRBA (Composite Rigid Body Algorithm)",\
                          func_notes,func_params,None)
    self.gen_add_code_line("template <typename T, bool USE_COMPRESSED_MEM = false>")
    self.gen_add_code_line("__host__")
    self.gen_add_code_line(func_def_start)
    self.gen_add_code_line(func_def_end, True)
    func_call_start = "crba_kernel<T><<<block_dimms,thread_dimms,ID_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_c,hd_data->d_q_qd,stride_q_qd,"
    func_call_end = "d_robotModel,gravity,num_timesteps);"
    if single_call_timing:
        func_call_start = func_call_start.replace("kernel<T>","kernel_single_timing<T>")
    if not compute_only:
        # start code with memory transfer
        self.gen_add_code_lines(["// start code with memory transfer", \
                                 "int stride_q_qd;", \
                                 "if (USE_COMPRESSED_MEM) {stride_q_qd = 2*NUM_JOINTS; " + \
                                    "gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd,hd_data->h_q_qd,stride_q_qd*" + \
                                    ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyHostToDevice,streams[0]));}", \
                                 "else {stride_q_qd = 3*NUM_JOINTS; " + \
                                    "gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd*" + \
                                    ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyHostToDevice,streams[0]));}", \
                                 "if (USE_QDD_FLAG) {gpuErrchk(cudaMemcpyAsync(hd_data->d_qdd,hd_data->h_qdd,NUM_JOINTS*" + \
                                    ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyHostToDevice,streams[1]));}", \
                                 "gpuErrchk(cudaDeviceSynchronize());"])
    else:
        self.gen_add_code_line("int stride_q_qd = USE_COMPRESSED_MEM ? 2*NUM_JOINTS: 3*NUM_JOINTS;")
    self.gen_add_code_line("// then call the kernel")
    func_call = func_call_start + func_call_end
    # add in compressed mem adjusts
    func_call_mem_adjust = "    if (USE_COMPRESSED_MEM) {" + func_call + "}"
    func_call_mem_adjust2 = "    else                    {" + func_call.replace("hd_data->d_q_qd","hd_data->d_q_qd_u") + "}"
    # compule into a set of code
    func_call_code = ["{", func_call_mem_adjust, func_call_mem_adjust2, "}", "gpuErrchk(cudaDeviceSynchronize());"]
    # wrap function call in timing (if needed)
    if single_call_timing:
        func_call_code.insert(0,"struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);")
        func_call_code.append("clock_gettime(CLOCK_MONOTONIC,&end);")
    self.gen_add_code_lines(func_call_code)
    if not compute_only:
        # then transfer memory back
        self.gen_add_code_lines(["// finally transfer the result back", \
                                 "gpuErrchk(cudaMemcpy(hd_data->h_c,hd_data->d_c,NUM_JOINTS*" + \
                                    ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyDeviceToHost));",
                                 "gpuErrchk(cudaDeviceSynchronize());"])
    # finally report out timing if requested
    if single_call_timing:
        self.gen_add_code_line("printf(\"Single Call ID %fus\\n\",time_delta_us_timespec(start,end)/static_cast<double>(num_timesteps));")
    self.gen_add_end_function()

def gen_crba(self, use_thread_group = False):
    # first generate the inner helpers
    self.gen_crba_inner(use_thread_group)
    # then generate the device wrappers
    self.gen_crba_device(use_thread_group)
    # then generate the kernels
    self.gen_crba_kernel(use_thread_group,True)
    self.gen_crba_kernel(use_thread_group,False)
    # then the host launch wrappers
    self.gen_crba_host(0)
    self.gen_crba_host(1)
    self.gen_crba_host(2)


    