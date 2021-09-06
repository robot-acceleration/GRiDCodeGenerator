def gen_inverse_dynamics_inner_temp_mem_size(self):
        n = self.robot.get_num_pos()
        return 6*n

def gen_inverse_dynamics_inner_function_call(self, use_thread_group = False, compute_c = False, use_qdd_input = False, updated_var_names = None):
    var_names = dict( \
        s_c_name = "s_c", \
        s_vaf_name = "s_vaf", \
        s_q_name = "s_q", \
        s_qd_name = "s_qd", \
        s_qdd_name = "s_qdd", \
        s_temp_name = "s_temp", \
        gravity_name = "gravity"
    )
    if updated_var_names is not None:
        for key,value in updated_var_names.items():
            var_names[key] = value
    id_code_start = "inverse_dynamics_inner<T>(" + var_names["s_vaf_name"] + ", " + var_names["s_q_name"] + ", " + var_names["s_qd_name"] + ", "
    id_code_end = var_names["s_temp_name"] + ", " + var_names["gravity_name"] + ");"
    if compute_c:
        id_code_start = id_code_start.replace("(", "(" + var_names["s_c_name"] + ", ")
    else:
        id_code_start = id_code_start.replace("<T>","_vaf<T>")
    # account for thread group and qdd
    if use_thread_group:
        id_code_start = id_code_start.replace("(","(tgrp, ")
    if use_qdd_input:
        id_code_start += var_names["s_qdd_name"] + ", "
    id_code_middle = self.gen_insert_helpers_function_call()
    id_code = id_code_start + id_code_middle + id_code_end
    self.gen_add_code_line(id_code)

def gen_inverse_dynamics_inner(self, use_thread_group = False, compute_c = False, use_qdd_input = False):
    n = self.robot.get_num_pos()
    n_bfs_levels = self.robot.get_max_bfs_level() + 1 # starts at 0
    # construct the boilerplate and function definition
    func_params = ["s_vaf is a pointer to shared memory of size 3*6*NUM_JOINTS = " + str(18*n), \
                   "s_q is the vector of joint positions", \
                   "s_qd is the vector of joint velocities", \
                   "s_XI is the pointer to the transformation and inertia matricies ", \
                   "s_temp is a pointer to helper shared memory of size 6*NUM_JOINTS = " + \
                            str(self.gen_inverse_dynamics_inner_temp_mem_size()), \
                   "gravity is the gravity constant"]
    func_notes = ["Assumes the XI matricies have already been updated for the given q"]
    func_def_start = "void inverse_dynamics_inner("
    func_def_middle = "T *s_vaf, const T *s_q, const T *s_qd, "
    func_def_end = "T *s_temp, const T gravity) {"
    if use_thread_group:
        func_def_start += "cgrps::thread_group tgrp, "
        func_params.insert(0,"tgrp is the handle to the thread_group running this function")
    if compute_c:
        func_def_start += "T *s_c,  "
        func_params.insert(0,"s_c is the vector of output torques")
    else:
        func_def_start = func_def_start.replace("(","_vaf(")
        func_notes.append("used to compute vaf as helper values")
    if use_qdd_input:
        func_def_middle += "const T *s_qdd, "
        func_params.insert(-3,"s_qdd is (optional vector of joint accelerations")
    else:
        func_notes.append("optimized for qdd = 0")
    func_def_middle, func_params = self.gen_insert_helpers_func_def_params(func_def_middle, func_params, -2)
    func_def = func_def_start + func_def_middle + func_def_end
    # now generate the code
    self.gen_add_func_doc("Compute the RNEA (Recursive Newton-Euler Algorithm)",\
                          func_notes,func_params,None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)
    #
    # Initial Debug Prints if Requested
    #
    if self.DEBUG_MODE:
        self.gen_add_sync(use_thread_group)
        self.gen_add_serial_ops(use_thread_group)
        self.gen_add_code_line("printf(\"q\\n\"); printMat<T,1," + str(n) + ">(s_q,1);")
        self.gen_add_code_line("printf(\"qd\\n\"); printMat<T,1," + str(n) + ">(s_qd,1);")
        if use_qdd_input:
            self.gen_add_code_line("printf(\"qdd\\n\"); printMat<T,1,6>(s_qdd,1);")
        self.gen_add_code_line("for (int i = 0; i < " + str(n) + "; i++){printf(\"X[%d]\\n\",i); printMat<T,6,6>(&s_XImats[36*i],6);}")
        self.gen_add_code_line("for (int i = 0; i < " + str(n) + "; i++){printf(\"I[%d]\\n\",i); printMat<T,6,6>(&s_XImats[36*(i+" + str(n) + ")],6);}")
        self.gen_add_end_control_flow()
        self.gen_add_sync(use_thread_group)

    #
    # Forward Pass we are going to go in bfs_level waves
    # 
    self.gen_add_code_line("//")
    self.gen_add_code_line("// Forward Pass")
    self.gen_add_code_line("//")
    for bfs_level in range(n_bfs_levels):
        #
        # v and a need to be computed serially by wave
        #
        inds = self.robot.get_ids_by_bfs_level(bfs_level)
        joint_names = [self.robot.get_joint_by_id(ind).get_name() for ind in inds]
        link_names = [self.robot.get_link_by_id(ind).get_name() for ind in inds]
        parent_ind_cpp, S_ind_cpp = self.gen_topology_helpers_pointers_for_cpp(inds, NO_GRAD_FLAG = True)

        if bfs_level == 0: 
            self.gen_add_code_line("// s_v, s_a where parent is base")
            self.gen_add_code_line("//     joints are: " + ", ".join(joint_names))
            self.gen_add_code_line("//     links are: " + ", ".join(link_names))
            # compute the initial v which is just S*qd
            # compute the initial a which is just X*gravity_vec (aka X_last_col*gravity_const) + S*qdd
            comment = "// s_v[k] = S[k]*qd[k] and s_a[k] = X[k]*gravity"
            if use_qdd_input:
                comment += "S[k]*qdd[k]"
            self.gen_add_code_line(comment)
            # load in 0 to v and X*gravity to a in parallel
            # note that depending on S we need to add qd/qdd to one entry
            if len(inds) > 1:
                self.gen_add_parallel_loop("ind",str(6*len(inds)),use_thread_group)
                self.gen_add_code_line("int row = ind % 6;")
                select_var_vals = [("int", "jid", [str(jid) for jid in inds])]
                self.gen_add_multi_threaded_select("ind", "<", [str(6*(i+1)) for i in range(len(inds))], select_var_vals)
                jid = "jid"
            else:
                self.gen_add_parallel_loop("row",str(6),use_thread_group)
                jid = str(inds[0])
            self.gen_add_code_lines(["int jid6 = 6*" + jid + ";", \
                                     "s_vaf[jid6 + row] = static_cast<T>(0);", \
                                     "s_vaf[" + str(n*6) + " + jid6 + row] = s_XImats[6*jid6 + 30 + row]*gravity;"])
            # then add in qd and qdd
            qd_qdd_code = "if (row == " + S_ind_cpp + "){s_vaf[jid6 + " + S_ind_cpp + "] += s_qd[" + jid + "];}"
            if use_qdd_input:
                qd_qdd_code = qd_qdd_code.replace("}", " s_vaf[" + str(n*6) + " + jid6 + " + S_ind_cpp + "] += s_qdd[" + jid + "];}")
            self.gen_add_code_line(qd_qdd_code)
            self.gen_add_end_control_flow()
            self.gen_add_sync(use_thread_group)
            
            # add debug if requested
            if self.DEBUG_MODE:
                self.gen_add_sync(use_thread_group)
                self.gen_add_serial_ops(use_thread_group)
                for ind in inds:
                    self.gen_add_code_line("printf(\"s_v[" + str(ind) + "]\\n\"); printMat<T,1,6>(&s_vaf[6*" + str(ind) + "],1);")
                    self.gen_add_code_line("printf(\"s_a[" + str(ind) + "]\\n\"); printMat<T,1,6>(&s_vaf[" + str(6*n) + " + 6*" + str(ind) + "],1);")
                self.gen_add_end_control_flow()
                self.gen_add_sync(use_thread_group)

        else:
            self.gen_add_code_line("// s_v and s_a where bfs_level is " + str(bfs_level))
            self.gen_add_code_line("//     joints are: " + ", ".join(joint_names))
            self.gen_add_code_line("//     links are: " + ", ".join(link_names))
            # note the update
            comment = "// s_v[k] = X[k]*v[parent_k] + S[k]*qd[k] and s_a[k] = X[k]*a[parent_k]"
            comment += " + S[k]*qdd[k] + mxS[k](v[k])*qd[k]" if use_qdd_input else " + mxS[k](v[k])*qd[k]"
            self.gen_add_code_line(comment)
            # do in parallel Xmat then add qd/qdd
            self.gen_add_parallel_loop("ind",str(6*2*len(inds)),use_thread_group)
            self.gen_add_code_line("int row = ind % 6; int comp = ind / 6; int comp_mod = comp % " + str(len(inds)) + "; int vFlag = comp == comp_mod;")
            # adjust for only one ind and thus fixed Srow and jid and jid_parent
            if len(inds) > 1:
                select_var_vals = [("int", "jid", [str(jid) for jid in inds])]
                jid = "jid"
                self.gen_add_multi_threaded_select("comp_mod", "==", [str(i) for i in range(len(inds))], select_var_vals)
            else:
                jid = str(inds[0])
            self.gen_add_code_line("int vaOffset = !vFlag * " + str(6*n) + "; int jid6 = 6 * " + jid + ";")
            qd_qdd_val_code = "T qd_qdd_val = (row == " + S_ind_cpp + ") * (vFlag * s_qd[" + jid + "]);"
            if use_qdd_input:
                qd_qdd_val_code = qd_qdd_val_code.replace(");", " + !vFlag * s_qdd[" + jid + "]);")
            self.gen_add_code_line(qd_qdd_val_code)
            self.gen_add_code_line("// compute based on the branch and use bool multiply for no branch")
            self.gen_add_code_line("s_vaf[vaOffset + jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[6*jid6 + row], &s_vaf[vaOffset + 6*" + parent_ind_cpp + "]) + qd_qdd_val;")
            self.gen_add_end_control_flow()

            # add debug if requested
            if self.DEBUG_MODE:
                self.gen_add_sync(use_thread_group)
                self.gen_add_serial_ops(use_thread_group)
                for ind in inds:
                    self.gen_add_code_line("printf(\"s_v[" + str(ind) + "] = X*s_v[" + str(self.robot.get_parent_id(ind)) + "] + S*qd[" + str(ind) + "]\\n\"); printMat<T,1,6>(&s_vaf[6*" + str(ind) + "],1);")
                    if use_qdd_input:
                        self.gen_add_code_line("printf(\"s_a[" + str(ind) + "] = X*s_a[" + str(self.robot.get_parent_id(ind)) + "] + S*qdd[" + str(ind) + "]\\n\"); printMat<T,1,6>(&s_vaf[" + str(6*n) + " + 6*" + str(ind) + "],1);")
                    else:
                        self.gen_add_code_line("printf(\"s_a[" + str(ind) + "] = X*s_a[" + str(self.robot.get_parent_id(ind)) + "]\\n\"); printMat<T,1,6>(&s_vaf[" + str(6*n) + " + 6*" + str(ind) + "],1);")
                self.gen_add_end_control_flow()
                self.gen_add_sync(use_thread_group)
            self.gen_add_code_line("// sync before a += MxS(v)*qd[S] ")
            self.gen_add_sync(use_thread_group)
            
            # attempt to do as much of the Mx in parallel as possible (will branch on different S but that is inevitable)
            self.gen_add_parallel_loop("ind",str(len(inds)),use_thread_group)
            if len(inds) > 1:
                select_var_vals = [("int", "jid", [str(jid) for jid in inds])]
                self.gen_add_multi_threaded_select("ind", "==", [str(i) for i in range(len(inds))], select_var_vals)
                dst_name = "&s_vaf[" + str(6*n) + " + 6*jid]"
                src_name = "&s_vaf[6*jid]"
                scale_name = "s_qd[jid]"
            else:
                jid = inds[0]
                dst_name = "&s_vaf[" + str(6*n + 6*jid) + "]"
                src_name = "&s_vaf[" + str(6*jid) + "]"
                scale_name = "s_qd[" + str(jid) + "]"
            updated_var_names = dict(S_ind_name = S_ind_cpp, s_dst_name = dst_name, s_src_name = src_name, s_scale_name = scale_name)
            self.gen_mx_func_call_for_cpp(inds, PEQ_FLAG = True, SCALE_FLAG = True, updated_var_names = updated_var_names)
            self.gen_add_end_control_flow()
            self.gen_add_sync(use_thread_group)
            # add debug if requested
            if self.DEBUG_MODE:
                self.gen_add_sync(use_thread_group)
                self.gen_add_serial_ops(use_thread_group)
                for ind in inds:
                    self.gen_add_code_line("printf(\"s_a[" + str(ind) + "] += MxS(s_v[" + str(ind) + "])\\n\"); printMat<T,1,6>(&s_vaf[" + str(6*n) + " + 6*" + str(ind) + "],1);")
                self.gen_add_end_control_flow()
                self.gen_add_sync(use_thread_group)
    #
    # then compute all f in parallel
    #
    inds = list(range(n))
    self.gen_add_code_line("//")
    self.gen_add_code_line("// s_f in parallel given all v, a")
    self.gen_add_code_line("//")
    self.gen_add_code_line("// s_f[k] = I[k]*a[k] + fx(v[k])*I[k]*v[k]")
    self.gen_add_code_line("// start with s_f[k] = I[k]*a[k] and temp = *I[k]*v[k]")
    self.gen_add_parallel_loop("ind",str(6*2*n),use_thread_group)
    self.gen_add_code_line("int row = ind % 6; int comp = ind / 6; int jid = comp % " + str(n) + ";")
    self.gen_add_code_line("bool IaFlag = comp == jid; int jid6 = 6*jid; int vaOffset = IaFlag * " + str(6*n) + " + jid6;")
    self.gen_add_code_line("T *dst = IaFlag ? &s_vaf[" + str(12*n) + "] : s_temp;")
    self.gen_add_code_line("// compute based on the branch and save Iv to temp to prep for fx(v)*Iv and then sync")
    self.gen_add_code_line("dst[jid6 + row] = dot_prod<T,6,6,1>(&s_XImats[" + str(36*n) + " + 6*jid6 + row], &s_vaf[vaOffset]);")
    self.gen_add_end_control_flow()
    self.gen_add_sync(use_thread_group)
    if self.DEBUG_MODE:
        self.gen_add_sync(use_thread_group)
        self.gen_add_serial_ops(use_thread_group)
        for ind in inds:
            self.gen_add_code_line("printf(\"s_f[" + str(ind) + "] = I*s_a[" + str(ind) + "])\\n\"); printMat<T,1,6>(&s_vaf[" + str(12*n) + " + 6*" + str(ind) + "],1);")
            self.gen_add_code_line("printf(\"I*s_v[" + str(ind) + "])\\n\"); printMat<T,1,6>(&s_temp[6*" + str(ind) + "],1);")
        self.gen_add_end_control_flow()
        self.gen_add_sync(use_thread_group)
    self.gen_add_code_line("// finish with s_f[k] += fx(v[k])*Iv[k]")
    self.gen_add_parallel_loop("jid",str(len(inds)),use_thread_group)
    self.gen_add_code_line("int jid6 = 6*jid;")
    self.gen_add_code_line("fx_times_v_peq<T>(&s_vaf[" + str(12*n) + " + jid6], &s_vaf[jid6], &s_temp[jid6]);")
    self.gen_add_end_control_flow()
    self.gen_add_sync(use_thread_group)
    if self.DEBUG_MODE:
        self.gen_add_sync(use_thread_group)
        self.gen_add_serial_ops(use_thread_group)
        for ind in inds:
            self.gen_add_code_line("printf(\"s_f[" + str(ind) + "] += fx(v[" + str(ind) + "])*I*v[" + str(ind) + "])\\n\"); printMat<T,1,6>(&s_vaf[" + str(12*n) + " + 6*" + str(ind) + "],1);")
        self.gen_add_code_line("printf(\"s_f forward pass\\n\"); printMat<T,6," + str(n) + ">(&s_vaf[" + str(12*n) + "],6);")
        self.gen_add_end_control_flow()
        self.gen_add_sync(use_thread_group)
    #
    # Then compute the Backward Pass again in bfs waves
    #
    self.gen_add_code_line("//")
    self.gen_add_code_line("// Backward Pass")
    self.gen_add_code_line("//")
    # backward pass start by updating all f by bfs_level
    for bfs_level in range(n_bfs_levels - 1, 0, -1): # don't consider level 0 as parent is root
        inds = self.robot.get_ids_by_bfs_level(bfs_level)
        joint_names = [self.robot.get_joint_by_id(ind).get_name() for ind in inds]
        link_names = [self.robot.get_link_by_id(ind).get_name() for ind in inds]
        parent_ind_cpp, _  = self.gen_topology_helpers_pointers_for_cpp(inds, NO_GRAD_FLAG = True)
        
        self.gen_add_code_line("// s_f update where bfs_level is " + str(bfs_level))
        self.gen_add_code_line("//     joints are: " + ", ".join(joint_names))
        self.gen_add_code_line("//     links are: " + ", ".join(link_names))
        # update f parent from f
        self.gen_add_code_line("// s_f[parent_k] += X[k]^T*f[k]")
        self.gen_add_parallel_loop("ind",str(6*len(inds)),use_thread_group)
        self.gen_add_code_line("int row = ind % 6;")
        if len(inds) > 1:
            select_var_vals = [("int", "jid", [str(jid) for jid in inds])]
            self.gen_add_multi_threaded_select("ind", "<", [str(6*(i+1)) for i in range(len(inds))], select_var_vals)
            jid = "jid"
        else:
            jid = str(inds[0])
        self.gen_add_code_line("T val = dot_prod<T,6,1,1>(&s_XImats[36*" + jid + " + 6*row], &s_vaf[" + str(12*n) + " + 6*" + jid + "]);")
        self.gen_add_code_line("int dstOffset = " + str(12*n) + " + 6*" + parent_ind_cpp + " + row;")
        # be careful to make sure you don't have multiuple parents the same -- else add atomics
        # we use atomics because there could still be some atomic parallelism at this level vs. simply looping
        if self.robot.has_repeated_parents(inds):
            self.gen_add_code_line("// using atomics due to repeated parent")
            self.gen_add_code_line("atomicAdd(&s_vaf[dstOffset], val);")
        else:    
            self.gen_add_code_line("s_vaf[dstOffset] += val;")
        self.gen_add_end_control_flow()
        self.gen_add_sync(use_thread_group)

        if self.DEBUG_MODE:
            self.gen_add_sync(use_thread_group)
            self.gen_add_serial_ops(use_thread_group)
            for ind in inds:
                self.gen_add_code_line("printf(\"s_f[" + str(self.robot.get_parent_id(ind)) + "] += X^T*s_f[" + str(ind) + "]\\n\"); printMat<T,1,6>(&s_vaf[" + str(12*n) + " + 6*" + str(self.robot.get_parent_id(ind)) + "],1);")
            self.gen_add_end_control_flow()
            self.gen_add_sync(use_thread_group)

    if compute_c:
        # then extract all c in parallel
        _, S_ind_cpp = self.gen_topology_helpers_pointers_for_cpp(NO_GRAD_FLAG = True)
        self.gen_add_code_line("//")
        self.gen_add_code_line("// s_c extracted in parallel (S*f)")
        self.gen_add_code_line("//")
        self.gen_add_parallel_loop("jid",str(n),use_thread_group)
        self.gen_add_code_line("s_c[jid] = s_vaf[" + str(12*n) + " + 6*jid + " + S_ind_cpp + "];")
        self.gen_add_end_control_flow()
        self.gen_add_sync(use_thread_group)
    self.gen_add_end_function()

def gen_inverse_dynamics_device_temp_mem_size(self, compute_c = False):
    n = self.robot.get_num_pos()
    wrapper_size = (18*n if compute_c else 0) + self.gen_topology_helpers_size() + 72*n # for XImats
    return self.gen_inverse_dynamics_inner_temp_mem_size() + wrapper_size

def gen_inverse_dynamics_device(self, use_thread_group = False, compute_c = False, use_qdd_input = False):
    n = self.robot.get_num_pos()
    # construct the boilerplate and function definition
    func_params = ["s_q is the vector of joint positions", \
                   "s_qd is the vector of joint velocities", \
                   "d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)", \
                   "gravity is the gravity constant"]
    func_notes = []
    func_def_start = "void inverse_dynamics_device("
    func_def_middle = "const T *s_q, const T *s_qd, "
    func_def_end = "const robotModel<T> *d_robotModel, const T gravity) {"
    if use_thread_group:
        func_def_start += "cgrps::thread_group tgrp, "
        func_params.insert(0,"tgrp is the handle to the thread_group running this function")
    if compute_c:
        func_def_start += "T *s_c,  "
        func_params.insert(0,"s_c is the vector of output torques")
    else:
        func_def_start = func_def_start.replace("_device(","_vaf_device(")
        func_def_start += "T *s_vaf, "
        func_notes.append("used to compute vaf as helper values")
    if use_qdd_input:
        func_def_middle += "const T *s_qdd, "
        func_params.insert(-2,"s_qdd is the vector of joint accelerations")
    else:
        func_notes.append("optimized for qdd = 0")
    func_def = func_def_start + func_def_middle + func_def_end
    # then generate the code
    self.gen_add_func_doc("Compute the RNEA (Recursive Newton-Euler Algorithm)",\
                          func_notes,func_params,None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)
    # add the shared memory variables
    if compute_c:
        self.gen_add_code_line("__shared__ T s_vaf[" + str(18*n) + "];")
    shared_mem_size = self.gen_inverse_dynamics_inner_temp_mem_size() if not self.use_dynamic_shared_mem_flag else None
    self.gen_XImats_helpers_temp_shared_memory_code(shared_mem_size)
    # then load/update XI and run the algo
    self.gen_load_update_XImats_helpers_function_call(use_thread_group)
    self.gen_inverse_dynamics_inner_function_call(use_thread_group,compute_c,use_qdd_input)
    self.gen_add_end_function()

def gen_inverse_dynamics_kernel(self, use_thread_group = False, use_qdd_input = False, single_call_timing = False):
    n = self.robot.get_num_pos()
    compute_c = True
    # define function def and params
    func_params = ["d_c is the vector of output torques", \
                   "d_q_dq is the vector of joint positions and velocities", \
                   "stride_q_qd is the stide between each q, qd", \
                   "d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)", \
                   "gravity is the gravity constant,"
                   "num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)"]
    func_notes = []
    func_def_start = "void inverse_dynamics_kernel(T *d_c, const T *d_q_qd, const int stride_q_qd, "
    func_def_end = "const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {"
    if use_qdd_input:
        func_def_start += "const T *d_qdd, "
        func_params.insert(-3,"d_qdd is the vector of joint accelerations")
    else:
        func_notes.append("optimized for qdd = 0")
    func_def = func_def_start + func_def_end
    if single_call_timing:
        func_def = func_def.replace("kernel(", "kernel_single_timing(")
    # then generate the code
    self.gen_add_func_doc("Compute the RNEA (Recursive Newton-Euler Algorithm)",\
                          func_notes,func_params,None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__global__")
    self.gen_add_code_line(func_def, True)
    # add shared memory variables
    shared_mem_vars = ["__shared__ T s_q_qd[2*" + str(n) + "]; T *s_q = s_q_qd; T *s_qd = &s_q_qd[" + str(n) + "];", \
                       "__shared__ T s_c[" + str(n) + "];",
                       "__shared__ T s_vaf[" + str(18*n) + "];"]
    if use_qdd_input:
        shared_mem_vars.insert(-2,"__shared__ T s_qdd[" + str(n) + "]; ")
    self.gen_add_code_lines(shared_mem_vars)
    shared_mem_size = self.gen_inverse_dynamics_inner_temp_mem_size() if not self.use_dynamic_shared_mem_flag else None
    self.gen_XImats_helpers_temp_shared_memory_code(shared_mem_size)
    if use_thread_group:
        self.gen_add_code_line("cgrps::thread_group tgrp = TBD;")
    if not single_call_timing:
        # load to shared mem and loop over blocks to compute all requested comps
        self.gen_add_parallel_loop("k","NUM_TIMESTEPS",use_thread_group,block_level = True)
        if use_qdd_input:
            self.gen_kernel_load_inputs("q_qd","stride_q_qd",str(2*n),use_thread_group,"qdd",str(n),str(n))
        else:
            self.gen_kernel_load_inputs("q_qd","stride_q_qd",str(2*n),use_thread_group)
        # compute
        self.gen_add_code_line("// compute")
        self.gen_load_update_XImats_helpers_function_call(use_thread_group)
        self.gen_inverse_dynamics_inner_function_call(use_thread_group,compute_c,use_qdd_input)
        self.gen_add_sync(use_thread_group)
        # save to global
        self.gen_kernel_save_result("c",str(n),str(n),use_thread_group)
        self.gen_add_end_control_flow()
    else:
        #repurpose NUM_TIMESTEPS for number of timing reps
        if use_qdd_input:
            self.gen_kernel_load_inputs_single_timing("q_qd",str(2*n),use_thread_group,"qdd",str(n))
        else:
            self.gen_kernel_load_inputs_single_timing("q_qd",str(2*n),use_thread_group)
        # then compute in loop for timing
        self.gen_add_code_line("// compute with NUM_TIMESTEPS as NUM_REPS for timing")
        self.gen_add_code_line("for (int rep = 0; rep < NUM_TIMESTEPS; rep++){", True)
        self.gen_load_update_XImats_helpers_function_call(use_thread_group)
        self.gen_inverse_dynamics_inner_function_call(use_thread_group,compute_c,use_qdd_input)
        self.gen_add_end_control_flow()
        # save to global
        self.gen_kernel_save_result_single_timing("c",str(n),use_thread_group)
    self.gen_add_end_function()

def gen_inverse_dynamics_host(self, mode = 0):
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
    func_def_start = "void inverse_dynamics(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,"
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
    self.gen_add_code_line("template <typename T, bool USE_QDD_FLAG = false, bool USE_COMPRESSED_MEM = false>")
    self.gen_add_code_line("__host__")
    self.gen_add_code_line(func_def_start)
    self.gen_add_code_line(func_def_end, True)
    func_call_start = "inverse_dynamics_kernel<T><<<block_dimms,thread_dimms,ID_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_c,hd_data->d_q_qd,stride_q_qd,"
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
    # then compute but adjust for compressed mem and qdd usage
    self.gen_add_code_line("// then call the kernel")
    func_call = func_call_start + func_call_end
    func_call_with_qdd = func_call_start + "hd_data->d_qdd, " + func_call_end
    # add in compressed mem adjusts
    func_call_mem_adjust = "    if (USE_COMPRESSED_MEM) {" + func_call + "}"
    func_call_mem_adjust2 = "    else                    {" + func_call.replace("hd_data->d_q_qd","hd_data->d_q_qd_u") + "}"
    func_call_with_qdd_mem_adjust = "    if (USE_COMPRESSED_MEM) {" + func_call_with_qdd + "}"
    func_call_with_qdd_mem_adjust2 = "    else                    {" + func_call_with_qdd.replace("hd_data->d_q_qd","hd_data->d_q_qd_u") + "}"
    # compule into a set of code
    func_call_code = ["if (USE_QDD_FLAG) {", func_call_with_qdd_mem_adjust, func_call_with_qdd_mem_adjust2, "}", \
                      "else {", func_call_mem_adjust, func_call_mem_adjust2, "}", "gpuErrchk(cudaDeviceSynchronize());"]
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

def gen_inverse_dynamics(self, use_thread_group = False):
    # first generate the inner helpers
    self.gen_inverse_dynamics_inner(use_thread_group,True,True)
    self.gen_inverse_dynamics_inner(use_thread_group,True,False)
    self.gen_inverse_dynamics_inner(use_thread_group,False,True)
    self.gen_inverse_dynamics_inner(use_thread_group,False,False)
    # then generate the device wrappers
    self.gen_inverse_dynamics_device(use_thread_group,True,True)
    self.gen_inverse_dynamics_device(use_thread_group,True,False)
    self.gen_inverse_dynamics_device(use_thread_group,False,True)
    self.gen_inverse_dynamics_device(use_thread_group,False,False)
    # then generate the kernels
    self.gen_inverse_dynamics_kernel(use_thread_group,True,True)
    self.gen_inverse_dynamics_kernel(use_thread_group,True,False)
    self.gen_inverse_dynamics_kernel(use_thread_group,False,True)
    self.gen_inverse_dynamics_kernel(use_thread_group,False,False)
    # then the host launch wrappers
    self.gen_inverse_dynamics_host(0)
    self.gen_inverse_dynamics_host(1)
    self.gen_inverse_dynamics_host(2)