def gen_inverse_dynamics_gradient_inner_temp_mem_size(self):
        n = self.robot.get_num_pos()
        (dva_cols_per_partial, _, _, df_cols_per_partial, _, _, _) = self.gen_topology_sparsity_helpers_python()
        return 66*n + 6*(4*dva_cols_per_partial + 2*df_cols_per_partial)

def gen_inverse_dynamics_gradient_inner_function_call(self, use_thread_group = False, updated_var_names = None):
    var_names = dict( \
        s_dc_du_name = "s_dc_du", \
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
    id_du_code_start = "inverse_dynamics_gradient_inner<T>(" + var_names["s_dc_du_name"] + ", " + var_names["s_q_name"] + ", " + var_names["s_qd_name"] + ", "
    id_du_code_middle = var_names["s_vaf_name"] + ", " + self.gen_insert_helpers_function_call()
    id_du_code_end = var_names["s_temp_name"] + ", " + var_names["gravity_name"] + ");"
    if use_thread_group:
        id_du_code_start = id_du_code_start.replace("(","(tgrp, ")
    id_du_code = id_du_code_start + id_du_code_middle + id_du_code_end
    self.gen_add_code_line(id_du_code)

def gen_inverse_dynamics_gradient_inner(self, use_thread_group = False):
    n = self.robot.get_num_pos()
    max_bfs_levels = self.robot.get_max_bfs_level()
    n_bfs_levels = max_bfs_levels + 1 # starts at 0

    # construct the boilerplate and function definition
    func_params = ["s_dc_du is a pointer to memory for the final result of size 2*NUM_JOINTS*NUM_JOINTS = " + str(2*n*n), \
                   "s_q is the vector of joint positions", \
                   "s_qd is the vector of joint velocities", \
                   "s_vaf are the helper intermediate variables computed by inverse_dynamics", \
                   "s_temp is a pointer to helper shared memory of size 66*NUM_JOINTS + 6*sparse_dv,da,df_col_needs = " + \
                            str(self.gen_inverse_dynamics_gradient_inner_temp_mem_size()), \
                   "gravity is the gravity constant"]
    func_def_start = "void inverse_dynamics_gradient_inner(T *s_dc_du, const T *s_q, const T *s_qd, const T *s_vaf, "
    func_def_end = "T *s_temp, const T gravity) {"
    func_def_start, func_params = self.gen_insert_helpers_func_def_params(func_def_start, func_params, -2)
    func_notes = ["Assumes s_XImats is updated already for the current s_q"]
    if use_thread_group:
        func_def_start = func_def_start.replace("(", "(cgrps::thread_group tgrp, ")
        func_params.insert(0,"tgrp is the handle to the thread_group running this function")
    func_def = func_def_start + func_def_end
    # then generate the code
    self.gen_add_func_doc("Computes the gradient of inverse dynamics",func_notes,func_params,None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)

    #
    # Optimize memory requirements due to sparsity induced by branching
    # requires more complex pointer math but/and saves a lot of space
    #
    (dva_cols_per_partial, dva_cols_per_jid, running_sum_dva_cols_per_jid, \
      df_cols_per_partial, df_cols_per_jid, running_sum_df_cols_per_jid, df_col_that_is_jid) = self.gen_topology_sparsity_helpers_python()
    self.gen_add_code_line("//")
    self.gen_add_code_line("// dv and da need " + str(dva_cols_per_partial) + " cols per dq,dqd")
    self.gen_add_code_line("// df needs " + str(df_cols_per_partial) + " cols per dq,dqd")
    self.gen_add_code_line("//    out of a possible " + str(n*n) + " cols per dq,dqd")
    self.gen_add_code_line("// Gradients are stored compactly as dv_i/dq_[0...a], dv_i+1/dq_[0...b], etc")
    self.gen_add_code_line("//    where a and b are the needed number of columns")
    self.gen_add_code_line("//")
    # gen som aditional helpers
    running_sum_delta_df_dva_cols_per_jid = [running_sum_df_cols_per_jid[jid] - running_sum_dva_cols_per_jid[jid] for jid in range(n)]

    # add shared memory note
    Offset_dv_dq = 0
    Offset_dv_dqd = Offset_dv_dq + 6*dva_cols_per_partial
    Offset_da_dq = Offset_dv_dqd + 6*dva_cols_per_partial
    Offset_da_dqd = Offset_da_dq + 6*dva_cols_per_partial
    Offset_df_dq = Offset_da_dqd + 6*dva_cols_per_partial
    Offset_df_dqd = Offset_df_dq + 6*df_cols_per_partial
    Offset_FxvI = Offset_df_dqd + 6*df_cols_per_partial
    Offset_MxXv = Offset_FxvI + 36*n
    Offset_MxXa = Offset_MxXv + 6*n
    Offset_Mxv = Offset_MxXa + 6*n
    Offset_Mxf = Offset_Mxv + 6*n
    Offset_Iv = Offset_Mxf + 6*n
    # Offset_dva_cols = Offset_Iv + 6*n
    # Offset_df_cols = Offset_dva_cols + n

    self.gen_add_code_line("// Temp memory offsets are as follows:")
    self.gen_add_code_line("// T *s_dv_dq = &s_temp[" + str(Offset_dv_dq) + "]; " + \
                              "T *s_dv_dqd = &s_temp[" + str(Offset_dv_dqd) + "]; " + \
                              "T *s_da_dq = &s_temp[" + str(Offset_da_dq) + "];")
    self.gen_add_code_line("// T *s_da_dqd = &s_temp[" + str(Offset_da_dqd) + "]; " + \
                              "T *s_df_dq = &s_temp[" + str(Offset_df_dq) + "]; " + \
                              "T *s_df_dqd = &s_temp[" + str(Offset_df_dqd) + "];")
    self.gen_add_code_line("// T *s_FxvI = &s_temp[" + str(Offset_FxvI) + "]; T *s_MxXv = &s_temp[" + str(Offset_MxXv) + "]; " + \
                              "T *s_MxXa = &s_temp[" + str(Offset_MxXa) + "];")
    self.gen_add_code_line("// T *s_Mxv = &s_temp[" + str(Offset_Mxv) + "]; T *s_Mxf = &s_temp[" + str(Offset_Mxf) + "]; " + \
                              "T *s_Iv = &s_temp[" + str(Offset_Iv) + "];")

    if self.DEBUG_MODE:
        self.gen_add_sync(use_thread_group)
        self.gen_add_serial_ops(use_thread_group)
        self.gen_add_code_lines(["printf(\"-------------------------\\n\");", \
                                 "printf(\"Validating Function Inputs\\n\");", \
                                 "printf(\"-------------------------\\n\");", \
                                 "printf(\"q\\n\"); printMat<T,1," + str(n) + ">(s_q,1);", \
                                 "printf(\"qd\\n\"); printMat<T,1," + str(n) + ">(s_qd,1);", \
                                 "printf(\"vaf-v\\n\"); printMat<T,6," + str(n) + ">(s_vaf,6);", \
                                 "printf(\"vaf-a\\n\"); printMat<T,6," + str(n) + ">(&s_vaf[6*" + str(n) + "],6);", \
                                 "printf(\"vaf-f\\n\"); printMat<T,6," + str(n) + ">(&s_vaf[12*" + str(n) + "],6);"])
        self.gen_add_code_line("for (int i = 0; i < " + str(n) + "; i++){printf(\"X[%d]\\n\",i); printMat<T,6,6>(&s_XImats[36*i],6);}")
        self.gen_add_code_line("for (int i = 0; i < " + str(n) + "; i++){printf(\"I[%d]\\n\",i); printMat<T,6,6>(&s_XImats[36*(i+" + str(n) + ")],6);}")
        self.gen_add_code_line("printf(\"-------------------------\\n\");")
        self.gen_add_end_control_flow()
        self.gen_add_sync(use_thread_group)
    
    #
    # Initial temp comps
    #
    self.gen_add_code_line("//")
    self.gen_add_code_line("// Initial Temp Comps")
    self.gen_add_code_line("//")
    # first compute temporary values by type of operation
    # we can use part of FxvI temp mem for Xv and Xa initial comps also compute Iv
    self.gen_add_code_line("// First compute Imat*v and Xmat*v_parent, Xmat*a_parent (store in FxvI for now)")
    self.gen_add_code_line("// Note that if jid_parent == -1 then v_parent = 0 and a_parent = gravity")
    self.gen_add_parallel_loop("ind",str(6*3*n),use_thread_group)
    self.gen_add_code_line("int row = ind % 6; int col = ind / 6; int jid = col % " + str(n) + "; int jid6 = 6*jid;")
    # get the parent (note that in some cases we have more efficient ways of computing this so add some special cases)
    parent_ind_cpp, S_ind_cpp = self.gen_topology_helpers_pointers_for_cpp(NO_GRAD_FLAG = True)
    self.gen_add_code_line("bool parentIsBase = " + parent_ind_cpp + " == -1;")
    # then get the offsets
    self.gen_add_code_lines(["bool comp1 = col < " + str(n) + "; bool comp3 = col >= " + str(2*n) + ";",
                             "int XIOffset  =  comp1 * " + str(36*n) + " + 6*jid6 + row; // rowCol of I (comp1) or X (comp 2 and 3)",
                             "int vaOffset  = comp1 * jid6 + !comp1 * 6*" + parent_ind_cpp + " + comp3 * " + str(6*n) + "; // v_i (comp1) or va_parent (comp 2 and 3)",
                             "int dstOffset = comp1 * " + str(Offset_Iv) + " + !comp1 * " + str(Offset_FxvI) + " + comp3 * " + str(6*n) + " + jid6 + row; // rowCol of dst"])
    self.gen_add_code_lines(["s_temp[dstOffset] = (parentIsBase && !comp1) ? comp3 * s_XImats[XIOffset + 30] * gravity : ",
                             "                                               dot_prod<T,6,6,1>(&s_XImats[XIOffset],&s_vaf[vaOffset]);"])
    self.gen_add_end_control_flow()
    self.gen_add_sync(use_thread_group)

    if self.DEBUG_MODE:
        self.gen_add_sync(use_thread_group)
        self.gen_add_serial_ops(use_thread_group)
        self.gen_add_code_lines(["printf(\"-------------------------\\n\");", \
                                 "printf(\"Temp Comps Part 1\\n\");", \
                                 "printf(\"-------------------------\\n\");", \
                                 "printf(\"Iv\\n\"); printMat<T,6," + str(n) + ">(&s_temp[" + str(Offset_Iv) + "],6);", \
                                 "printf(\"Xv\\n\"); printMat<T,6," + str(n) + ">(&s_temp[" + str(Offset_FxvI) + "],6);", \
                                 "printf(\"Xa\\n\"); printMat<T,6," + str(n) + ">(&s_temp[" + str(Offset_FxvI + 6*n) + "],6);"])
        self.gen_add_code_line("printf(\"-------------------------\\n\");")
        self.gen_add_end_control_flow()
        self.gen_add_sync(use_thread_group)

    # then do the mx comps
    self.gen_add_code_line("// Then compute Mx(Xv), Mx(Xa), Mx(v), Mx(f)")
    self.gen_add_parallel_loop("col",str(4*n),use_thread_group)
    self.gen_add_code_line("int jid = col / 4; int selector = col % 4; int jid6 = 6*jid;")
    select_var_vals = [("int", "dstOffset", [str(Offset_MxXv), str(Offset_MxXa), str(Offset_Mxv), str(Offset_Mxf)]), \
                       ("const T *", "src", ["&s_temp[" + str(Offset_FxvI) + "]", "&s_temp[" + str(Offset_FxvI + 6*n) + "]", \
                                       "&s_vaf[0]", "&s_vaf[" + str(12*n) + "]"])]
    self.gen_add_multi_threaded_select("selector", "==", [str(i) for i in range(4)], select_var_vals)
    updated_var_names = dict(S_ind_name = S_ind_cpp, s_dst_name = "&s_temp[dstOffset + jid6]", s_src_name = "&src[jid6]")
    self.gen_mx_func_call_for_cpp(PEQ_FLAG = False, SCALE_FLAG = False, updated_var_names = updated_var_names)
    self.gen_add_end_control_flow()
    self.gen_add_sync(use_thread_group)

    if self.DEBUG_MODE:
        self.gen_add_sync(use_thread_group)
        self.gen_add_serial_ops(use_thread_group)
        self.gen_add_code_lines(["printf(\"-------------------------\\n\");", \
                                 "printf(\"Temp Comps Part 2\\n\");", \
                                 "printf(\"-------------------------\\n\");", \
                                 "printf(\"Mx(Xv)\\n\"); printMat<T,6," + str(n) + ">(&s_temp[" + str(Offset_MxXv) + "],6);", \
                                 "printf(\"Mx(Xa)\\n\"); printMat<T,6," + str(n) + ">(&s_temp[" + str(Offset_MxXa) + "],6);", \
                                 "printf(\"Mx(v)\\n\"); printMat<T,6," + str(n) + ">(&s_temp[" + str(Offset_Mxv) + "],6);",\
                                 "printf(\"Mx(f)\\n\"); printMat<T,6," + str(n) + ">(&s_temp[" + str(Offset_Mxf) + "],6);"])
        self.gen_add_code_line("printf(\"-------------------------\\n\");")
        self.gen_add_end_control_flow()
        self.gen_add_sync(use_thread_group)

    #
    # FORWARD PASS
    #
    self.gen_add_code_line("//")
    self.gen_add_code_line("// Forward Pass")
    self.gen_add_code_line("//")
    self.gen_add_code_line("// We start with dv/du noting that we only have values")
    self.gen_add_code_line("//    for ancestors and for the current index else 0")
    # then serial dv/du in bfs waves
    for bfs_level in range(n_bfs_levels):
        inds = self.robot.get_ids_by_bfs_level(bfs_level)
        joint_names = [self.robot.get_joint_by_id(ind).get_name() for ind in inds]
        link_names = [self.robot.get_link_by_id(ind).get_name() for ind in inds]
        _, S_ind_cpp, dva_col_offset_for_jid_cpp, _, dva_col_offset_for_parent_cpp, _, _, _ = self.gen_topology_helpers_pointers_for_cpp(inds)
        self.gen_add_code_line("// dv/du where bfs_level is " + str(bfs_level))
        self.gen_add_code_line("//     joints are: " + ", ".join(joint_names))
        self.gen_add_code_line("//     links are: " + ", ".join(link_names))

        # when parent is base dv_dq = 0, dv_dqd = S
        if bfs_level == 0:
            self.gen_add_code_line("// when parent is base dv_dq = 0, dv_dqd = S")
            self.gen_add_parallel_loop("ind",str(6*2*len(inds)),use_thread_group)
            if len(inds) > 1:
                self.gen_add_code_line("int row = ind % 6; int col = ind / 6; int col_du = col % " + str(len(inds)) + "; bool dq_flag = col == col_du;")
                select_var_vals = [("int", "jid", [str(jid) for jid in inds])]
                self.gen_add_multi_threaded_select("col_du", "<", [str((i+1)) for i in range(len(inds))], select_var_vals)
            else:
                self.gen_add_code_line("int row = ind % 6; int dq_flag = (ind / 6) == 0;")
            self.gen_add_code_line("int du_offset = dq_flag ? " + str(Offset_dv_dq) + " : " + str(Offset_dv_dqd) + ";")
            self.gen_add_code_line("s_temp[du_offset + 6*" + dva_col_offset_for_jid_cpp + " + row] = " + \
                                   "(!dq_flag && row == " + S_ind_cpp + ") * static_cast<T>(1);") 
            self.gen_add_end_control_flow()

        # dv/du = X dv_parent/du + {MxXv or S for col ind}
        # there are 2*(bfs_level + 1) columns per du with 2*bfs mults with X and then the addition in the last col
        else:
            self.gen_add_code_line("// dv/du = Xmat*dv_parent/du + {Mx(Xv) or S for col ind}")
            self.gen_add_code_line("// first compute dv/du = Xmat*dv_parent/du")
            self.gen_add_parallel_loop("ind",str(6*2*bfs_level*len(inds)),use_thread_group)
            self.gen_add_code_line("int row = ind % 6; int col = ind / 6; int col_du = col % " + str(bfs_level*len(inds)) + "; " + \
                                                                          "int col_jid = col_du % " + str(bfs_level) + ";")
            if bfs_level > 1 or len(inds) > 1:
                self.gen_add_code_line("int dq_flag = col == col_du;")
            else:
                self.gen_add_code_line("int dq_flag = col < 1;")
            if len(inds) > 1:
                select_var_vals = [("int", "jid", [str(jid) for jid in inds])]
                jid = "jid"
                self.gen_add_multi_threaded_select("col_du", "<", [str((i+1)*bfs_level) for i in range(len(inds))], select_var_vals)
            else:
                jid = str(inds[0])
            self.gen_add_code_line("int du_col_offset = dq_flag * " + str(Offset_dv_dq) + " + !dq_flag * " + str(Offset_dv_dqd) + " + 6 * col_jid;")
            self.gen_add_code_line("s_temp[du_col_offset + 6*" + dva_col_offset_for_jid_cpp + " + row] = ")
            self.gen_add_code_line("    dot_prod<T,6,6,1>(&s_XImats[36*" + jid + " + row]," + \
                                                           "&s_temp[du_col_offset + 6*" + dva_col_offset_for_parent_cpp + "]);")
            self.gen_add_code_line("// then add {Mx(Xv) or S for col ind}")
            # all cols add if bfs_level is 1 so skip the if statement
            if bfs_level > 1:
                self.gen_add_code_line("if (col_jid == " + str(bfs_level-1) + ") {", True)
            # do the non-branching if/else
            self.gen_add_code_line("s_temp[du_col_offset + 6*" + dva_col_offset_for_jid_cpp + " + 6 + row] = ")
            self.gen_add_code_line("    dq_flag * s_temp[" + str(Offset_MxXv) + " + 6*" + jid + " + row] + " + \
                                     "(!dq_flag && row == " + S_ind_cpp + ") * static_cast<T>(1);")
            # all cols add if bfs_level is 1 so skip the if statement
            if bfs_level > 1:
                self.gen_add_end_control_flow()
            self.gen_add_end_control_flow()
        self.gen_add_sync(use_thread_group)

        if self.DEBUG_MODE:
            self.gen_add_sync(use_thread_group)
            self.gen_add_serial_ops(use_thread_group)
            if bfs_level == 0:
                self.gen_add_code_lines(["printf(\"-------------------------\\n\");", \
                                         "printf(\"dv/du in bfs waves\\n\");", \
                                         "printf(\"-------------------------\\n\");"])
            self.gen_add_code_line("printf(\"dv/du in for bfs wave[%d]\\n\"," + str(bfs_level) + ");")
            for ind in inds:
                self.gen_add_code_lines(["printf(\"dv[%d]/dq\\n\"," + str(ind) + ");", \
                                         "printMat<T,6," + str(bfs_level+1) + ">(&s_temp[" + \
                                                str(Offset_dv_dq + 6*running_sum_dva_cols_per_jid[ind]) + "],6);", \
                                         "printf(\"dv[%d]/dqd\\n\"," + str(ind) + ");", \
                                         "printMat<T,6," + str(bfs_level+1) + ">(&s_temp[" + \
                                                str(Offset_dv_dqd + 6*running_sum_dva_cols_per_jid[ind]) + "],6);"])
            self.gen_add_end_control_flow()
            self.gen_add_sync(use_thread_group)

    # Start the da/du comp with da/du = MxS(dv/du)*qd + {MxXa, Mxv}
    self.gen_add_code_line("// start da/du by setting = MxS(dv/du)*qd + {MxXa, Mxv} for all n in parallel")
    self.gen_add_code_line("// start with da/du = MxS(dv/du)*qd")
    _ , S_ind_cpp , _ , _ , _ , _ , dva_col_offset_for_jidp1_cpp, _ = self.gen_topology_helpers_pointers_for_cpp(list(range(n)))
    add_col_for_jid = "(" + dva_col_offset_for_jidp1_cpp + " - 1)"
    self.gen_add_parallel_loop("col",str(2*dva_cols_per_partial),use_thread_group)
    self.gen_add_code_line("int col_du = col % " + str(dva_cols_per_partial) + ";")
    # get the jid
    select_var_vals = [("int", "jid", [str(jid) for jid in range(n)])]
    self.gen_add_multi_threaded_select("col_du", "<", [str(running_sum_dva_cols_per_jid[jid+1]) for jid in range(n)], select_var_vals)
    # call the mx func
    updated_var_names = dict(S_ind_name = S_ind_cpp, s_dst_name = "&s_temp[" + str(Offset_da_dq) + " + 6*col]", \
                             s_src_name = "&s_temp[" + str(Offset_dv_dq) + " + 6*col]", s_scale_name = "s_qd[jid]")
    self.gen_mx_func_call_for_cpp(PEQ_FLAG = False, SCALE_FLAG = True, updated_var_names = updated_var_names)
    # then add to the add col
    self.gen_add_code_lines(["// then add {MxXa, Mxv} to the appropriate column", \
                             "int dq_flag = col == col_du; int src_offset = dq_flag * " + str(Offset_MxXa) + " + !dq_flag * " + str(Offset_Mxv) + " + 6*jid;"])
    self.gen_add_code_line("if(col_du == " + add_col_for_jid + "){", True)
    self.gen_add_code_line("for(int row = 0; row < 6; row++){", True)
    self.gen_add_code_line("s_temp[" + str(Offset_da_dq) + " + 6*col + row] += s_temp[src_offset + row];")
    self.gen_add_end_control_flow()
    self.gen_add_end_control_flow()
    self.gen_add_end_control_flow()
    self.gen_add_sync(use_thread_group)

    if self.DEBUG_MODE:
        self.gen_add_sync(use_thread_group)
        self.gen_add_serial_ops(use_thread_group)
        self.gen_add_code_lines(["printf(\"-------------------------\\n\");", \
                                 "printf(\"da/du part 1 = MxS(dv/du)*qd + {MxXa, Mxf}\\n\");", \
                                 "printf(\"-------------------------\\n\");"])
        for ind in range(n):
            num_cols = self.robot.get_bfs_level_by_id(ind) + 1
            self.gen_add_code_lines(["printf(\"da[%d]/dq\\n\"," + str(ind) + ");", \
                                     "printMat<T,6," + str(num_cols) + ">(&s_temp[" + \
                                            str(Offset_da_dq + 6*running_sum_dva_cols_per_jid[ind]) + "],6);", \
                                     "printf(\"da[%d]/dqd\\n\"," + str(ind) + ");", \
                                     "printMat<T,6," + str(num_cols) + ">(&s_temp[" + \
                                            str(Offset_da_dqd + 6*running_sum_dva_cols_per_jid[ind]) + "],6);"])
        self.gen_add_end_control_flow()
        self.gen_add_sync(use_thread_group)

    # then serial da/du in bfs waves
    self.gen_add_code_line("// Finish da/du with parent updates noting that we only have values")
    self.gen_add_code_line("//    for ancestors and for the current index and nothing for bfs 0")
    for bfs_level in range(1,n_bfs_levels):
        inds = self.robot.get_ids_by_bfs_level(bfs_level)
        joint_names = [self.robot.get_joint_by_id(ind).get_name() for ind in inds]
        link_names = [self.robot.get_link_by_id(ind).get_name() for ind in inds]
        _, _, dva_col_offset_for_jid_cpp, _, dva_col_offset_for_parent_cpp, _, _, _ = self.gen_topology_helpers_pointers_for_cpp(inds)
        parent_inds = [self.robot.get_parent_id(ind) for ind in inds]
        self.gen_add_code_line("// da/du where bfs_level is " + str(bfs_level))
        self.gen_add_code_line("//     joints are: " + ", ".join(joint_names))
        self.gen_add_code_line("//     links are: " + ", ".join(link_names))
        
        # da/du += X da_parent/du
        # there are 2*(bfs_level + 1) columns per du with 2*bfs mults with X and then the addition in the last col
        self.gen_add_code_line("// da/du += Xmat*da_parent/du")
        self.gen_add_parallel_loop("ind",str(6*2*bfs_level*len(inds)),use_thread_group)
        self.gen_add_code_lines(["int row = ind % 6; int col = ind / 6; int col_du = col % " + str(bfs_level*len(inds)) + ";", \
                                 "int dq_flag = col == col_du; int col_jid = col_du % " + str(bfs_level) + ";"])
        if len(inds) > 1:
            select_var_vals = [("int", "jid", [str(jid) for jid in inds])]
            jid = "jid"
            self.gen_add_multi_threaded_select("col_du", "<", [str((i+1)*bfs_level) for i in range(len(inds))], select_var_vals)
        else:
            jid = str(inds[0])
        self.gen_add_code_line("int du_col_offset = dq_flag * " + str(Offset_da_dq) + " + !dq_flag * " + str(Offset_da_dqd) + " + 6 * col_jid;")
        self.gen_add_code_lines(["s_temp[du_col_offset + 6*" + dva_col_offset_for_jid_cpp + " + row] += ", \
                                  "    dot_prod<T,6,6,1>(&s_XImats[36*" + jid + " + row]," + \
                                                        "&s_temp[du_col_offset + 6*" + dva_col_offset_for_parent_cpp + "]);"])
        self.gen_add_end_control_flow()
        self.gen_add_sync(use_thread_group)

        if self.DEBUG_MODE:
            self.gen_add_sync(use_thread_group)
            self.gen_add_serial_ops(use_thread_group)
            if bfs_level == 1:
                self.gen_add_code_lines(["printf(\"-------------------------\\n\");", \
                                         "printf(\"da/du in bfs waves\\n\");", \
                                         "printf(\"-------------------------\\n\");"])
            self.gen_add_code_line("printf(\"da/du for bfs wave[%d]\\n\"," + str(bfs_level) + ");")
            for ind in inds:
                self.gen_add_code_lines(["printf(\"da[%d]/dq\\n\"," + str(ind) + ");", \
                                         "printMat<T,6," + str(bfs_level+1) + ">(&s_temp[" + \
                                                str(Offset_da_dq + 6*running_sum_dva_cols_per_jid[ind]) + "],6);", \
                                         "printf(\"da[%d]/dqd\\n\"," + str(ind) + ");", \
                                         "printMat<T,6," + str(bfs_level+1) + ">(&s_temp[" + \
                                                str(Offset_da_dqd + 6*running_sum_dva_cols_per_jid[ind]) + "],6);"])
            self.gen_add_end_control_flow()
            self.gen_add_sync(use_thread_group)

    # Intiialize df/du to 0 to make sure we don't have issues with remaining values later when we do +=
    self.gen_add_code_line("// Init df/du to 0")
    self.gen_add_parallel_loop("ind",str(6*2*df_cols_per_partial),use_thread_group)
    self.gen_add_code_line("s_temp[" + str(Offset_df_dq) + " + ind] = static_cast<T>(0);")
    self.gen_add_end_control_flow()
    self.gen_add_sync(use_thread_group)

    # Start the df/du by setting = fx(dv/du)*Iv and also compute the temp = Fx(v)*I 
    # aka do all of the Fx comps in parallel
    self.gen_add_code_lines(["// Start the df/du by setting = fx(dv/du)*Iv and also compute the temp = Fx(v)*I ", \
                             "//    aka do all of the Fx comps in parallel", \
                             "// note that while df has more cols than dva the dva cols are the first few df cols"])
    _, _, dva_col_offset_for_jid_cpp, df_col_offset_for_jid_cpp, _, _, _, _ = self.gen_topology_helpers_pointers_for_cpp(list(range(n)))
    self.gen_add_parallel_loop("col",str(2*dva_cols_per_partial + 6*n),use_thread_group)
    self.gen_add_code_line("int col_du = col % " + str(dva_cols_per_partial) + ";")
    select_var_vals = [("int", "jid", [str(jid) for jid in range(n)])]
    self.gen_add_multi_threaded_select("col_du", "<", [str(running_sum_dva_cols_per_jid[jid+1]) for jid in range(n)], select_var_vals)
    self.gen_add_code_lines(["// Compute Offsets and Pointers", \
                             "int dq_flag = col == col_du; int dva_to_df_adjust = " + df_col_offset_for_jid_cpp + " - " + dva_col_offset_for_jid_cpp + ";", \
                             "int Offset_col_du_src = dq_flag * " + str(Offset_dv_dq) + " + !dq_flag * " + str(Offset_dv_dqd) + " + 6*col_du;", \
                             "int Offset_col_du_dst = dq_flag * " + str(Offset_df_dq) + " + !dq_flag * " + str(Offset_df_dqd) + " + 6*(col_du + dva_to_df_adjust);"])
    self.gen_add_code_line("T *dst = &s_temp[Offset_col_du_dst]; " + \
                           "const T *fx_src = &s_temp[Offset_col_du_src]; " + \
                           "const T *mult_src = &s_temp[" + str(Offset_Iv) + " + 6*jid];")
    # do the adjust for the temp comps
    self.gen_add_code_line("// Adjust pointers for temp comps (if applicable)")
    self.gen_add_code_line("if (col >= " + str(2*dva_cols_per_partial) + ") {", True)
    self.gen_add_code_lines(["int comp = col - " + str(2*dva_cols_per_partial) + "; int comp_col = comp % 6; // int jid = comp / 6;", \
                             "int jid6 = comp - comp_col; int jid36_col6 = 6*jid6 + 6*comp_col;"])
    self.gen_add_code_line("dst = &s_temp[" + str(Offset_FxvI) + " + jid36_col6]; " + \
                           "fx_src = &s_vaf[jid6]; " + \
                           "mult_src = &s_XImats[" + str(36*n) + " + jid36_col6];")
    self.gen_add_end_control_flow()
    self.gen_add_code_line("fx_times_v<T>(dst, fx_src, mult_src);")
    self.gen_add_end_control_flow()
    self.gen_add_sync(use_thread_group)

    if self.DEBUG_MODE:
        self.gen_add_sync(use_thread_group)
        self.gen_add_serial_ops(use_thread_group)
        self.gen_add_code_lines(["printf(\"-------------------------\\n\");", \
                                 "printf(\"df/du part 1 = fx(dv/du)*Iv\\n\");", \
                                 "printf(\"     and Temp = Fx(v)*I\\n\");", \
                                 "printf(\"-------------------------\\n\");"])
        for ind in range(n):
            num_cols = df_cols_per_jid[ind]
            self.gen_add_code_lines(["printf(\"df[%d]/dq\\n\"," + str(ind) + ");", \
                                     "printMat<T,6," + str(num_cols) + ">(&s_temp[" + str(Offset_df_dq + 6*running_sum_df_cols_per_jid[ind]) + "],6);", \
                                     "printf(\"df[%d]/dqd\\n\"," + str(ind) + ");", \
                                     "printMat<T,6," + str(num_cols) + ">(&s_temp[" + str(Offset_df_dqd + 6*running_sum_df_cols_per_jid[ind]) + "],6);"])
            self.gen_add_code_lines(["printf(\"Fx(v)*I[%d]\\n\"," + str(ind) + ");", \
                                     "printMat<T,6,6>(&s_temp[" + str(Offset_FxvI) + " + 36*" + str(ind) + "],6);"])
        self.gen_add_end_control_flow()
        self.gen_add_sync(use_thread_group)

    # then in parallel finish df/du += I*da/du + FxvI*dv/du
    self.gen_add_code_line("// Then in parallel finish df/du += I*da/du + (Fx(v)I)*dv/du")
    self.gen_add_parallel_loop("ind",str(6*2*dva_cols_per_partial),use_thread_group)
    self.gen_add_code_line("int row = ind % 6; int col = ind / 6; int col6 = ind - row; int col_du = (col % " + str(dva_cols_per_partial) + ");")
    select_var_vals = [("int", "jid", [str(jid) for jid in range(n)])]
    self.gen_add_multi_threaded_select("col_du", "<", [str(running_sum_dva_cols_per_jid[jid+1]) for jid in range(n)], select_var_vals)
    self.gen_add_code_lines(["// Compute Offsets and Pointers", \
                             "int dva_to_df_adjust = " + df_col_offset_for_jid_cpp + " - " + dva_col_offset_for_jid_cpp + ";", \
                             "if (col >= " + str(dva_cols_per_partial) + "){dva_to_df_adjust += " + str(df_cols_per_partial - dva_cols_per_partial) + ";}", \
                             "T *df_row_col = &s_temp[" + str(Offset_df_dq) + " + 6*dva_to_df_adjust + ind];",
                             "const T *dv_col = &s_temp[" + str(Offset_dv_dq) + " + col6]; " + \
                                    "const T *da_col = &s_temp[" + str(Offset_da_dq) + " + col6];",
                             "int jid36 = 36*jid; const T *I_row = &s_XImats[" + str(36*n) + " + jid36 + row]; " + \
                                               "const T *FxvI_row = &s_temp[" + str(Offset_FxvI) + " + jid36 + row];"])
    self.gen_add_code_lines(["// Compute the values", \
                             "*df_row_col += dot_prod<T,6,6,1>(I_row,da_col) + dot_prod<T,6,6,1>(FxvI_row,dv_col);"])
    self.gen_add_end_control_flow()

    if self.DEBUG_MODE:
        self.gen_add_sync(use_thread_group)
        self.gen_add_serial_ops(use_thread_group)
        self.gen_add_code_lines(["printf(\"-------------------------\\n\");", \
                                 "printf(\"df/du += I*da/du + FxvI*dv/du\\n\");", \
                                 "printf(\"-------------------------\\n\");"])
        for ind in range(n):
            num_cols = df_cols_per_jid[ind]
            self.gen_add_code_lines(["printf(\"df[%d]/dq\\n\"," + str(ind) + ");", \
                                     "printMat<T,6," + str(num_cols) + ">(&s_temp[" + str(Offset_df_dq + 6*running_sum_df_cols_per_jid[ind]) + "],6);", \
                                     "printf(\"df[%d]/dqd\\n\"," + str(ind) + ");", \
                                     "printMat<T,6," + str(num_cols) + ">(&s_temp[" + str(Offset_df_dqd + 6*running_sum_df_cols_per_jid[ind]) + "],6);"])
        self.gen_add_end_control_flow()
        self.gen_add_sync(use_thread_group)

    # and also at the same time compute the temp var -X^T * mxf
    # since all temps are done re-use one in practice
    self.gen_add_code_line("// At the same time compute the last temp var: -X^T * mx(f)")
    self.gen_add_code_line("// use Mx(Xv) temp memory as those values are no longer needed")
    self.gen_add_parallel_loop("ind",str(6*n),use_thread_group)
    self.gen_add_code_line("int XTcol = ind % 6; int jid6 = ind - XTcol;")
    self.gen_add_code_line("s_temp[" + str(Offset_MxXv) + " + ind] = -dot_prod<T,6,1,1>(" + \
                                    "&s_XImats[6*(jid6 + XTcol)], &s_temp[" + str(Offset_Mxf) + " + jid6]);")
    self.gen_add_end_control_flow()
    self.gen_add_sync(use_thread_group)

    if self.DEBUG_MODE:
        self.gen_add_sync(use_thread_group)
        self.gen_add_serial_ops(use_thread_group)
        self.gen_add_code_lines(["printf(\"-------------------------\\n\");", \
                                 "printf(\"Temp = -X^T * mx(f)\\n\");", \
                                 "printf(\"-------------------------\\n\");"])
        for ind in range(n):
            self.gen_add_code_lines(["printf(\"-X^T*mx(f)[%d]\\n\"," + str(ind) + ");", \
                                     "printMat<T,1,6>(&s_temp[" + str(Offset_MxXv) + " + 6*" + str(ind) + "],1);"])
        self.gen_add_end_control_flow()
        self.gen_add_sync(use_thread_group)

    #
    # BACKWARD PASS
    #
    self.gen_add_code_line("//")
    self.gen_add_code_line("// BACKWARD Pass")
    self.gen_add_code_line("//")
    # update df serially (df_lambda/du = X^T * df/du + {Xmx(f), 0})
    for bfs_level in range(max_bfs_levels,0,-1): # STOP AT 1 because updating parent and last is 0 ---- !!!!!
        inds = self.robot.get_ids_by_bfs_level(bfs_level)
        _, _, dva_col_offset_for_jid_cpp, df_col_offset_for_jid_cpp, _, df_col_offset_for_parent_cpp, _, df_col_that_is_jid_cpp = self.gen_topology_helpers_pointers_for_cpp(inds)
        joint_names = [self.robot.get_joint_by_id(ind).get_name() for ind in inds]
        link_names = [self.robot.get_link_by_id(ind).get_name() for ind in inds]
        self.gen_add_code_line("// df/du update where bfs_level is " + str(bfs_level))
        self.gen_add_code_line("//     joints are: " + ", ".join(joint_names))
        self.gen_add_code_line("//     links are: " + ", ".join(link_names))

        if self.DEBUG_MODE:
            self.gen_add_sync(use_thread_group)
            self.gen_add_serial_ops(use_thread_group)
            for ind in self.robot.get_unique_parent_ids(inds):
                self.gen_add_code_lines(["printf(\"df[%d]/dq (parent update) BEFORE UPDATE\\n\"," + str(ind) + ");", \
                                         "printMat<T,6," + str(df_cols_per_jid[ind]) + ">(&s_temp[" + str(Offset_df_dq + \
                                                    6*running_sum_df_cols_per_jid[ind]) + "],6);", \
                                         "printf(\"df[%d]/dqd (parent update) BEFORE UPDATE\\n\"," + str(ind) + ");", \
                                         "printMat<T,6," + str(df_cols_per_jid[ind]) + ">(&s_temp[" + str(Offset_df_dqd + \
                                                    6*running_sum_df_cols_per_jid[ind]) + "],6);"])
            self.gen_add_end_control_flow()
            self.gen_add_sync(use_thread_group)

        # df_lambda/du = X^T * df/du + {Xmx(f), 0}
        # there are 2*(bfs_level + 1) columns per du
        df_cols_per_this_bfs = [df_cols_per_jid[ind] for ind in inds]
        curr_cols_per_du = sum(df_cols_per_this_bfs)
        breakpoints = [sum(df_cols_per_this_bfs[0:i+1]) for i in range(len(inds))]
        col_adjusts = [sum(df_cols_per_this_bfs[0:i]) for i in range(len(inds))]
        sparsity_branch_corrector_vals = [str(jid - self.robot.get_parent_id(jid) - 1) for jid in inds]
        sparsity_branch_corrector_needed = any([int(i) for i in sparsity_branch_corrector_vals])
        if not sparsity_branch_corrector_needed:
            sparsity_branch_corrector = str(0)
        self.gen_add_code_line("// df_lambda/du += X^T * df/du + {Xmx(f), 0}")
        self.gen_add_parallel_loop("ind",str(6*2*curr_cols_per_du),use_thread_group)
        self.gen_add_code_line("int row = ind % 6; int col = ind / 6; int col_du = col % " + str(curr_cols_per_du) + ";")
        if len(inds) > 1:
            select_var_vals = [("int", "jid", [str(jid) for jid in inds])]
            select_var_vals.append(("int", "col_adjust", [str(val) for val in col_adjusts]))
            jid = "jid"
            if sparsity_branch_corrector_needed:
                select_var_vals.append(("int", "sparsity_branch_corrector", sparsity_branch_corrector_vals))
                sparsity_branch_corrector = "sparsity_branch_corrector"
            self.gen_add_multi_threaded_select("col_du", "<", [str(val) for val in breakpoints], select_var_vals, True)
            adjustments = "col_du -= col_adjust; // adjust for variable number of columns"
        else:
            jid = str(inds[0])
            if sparsity_branch_corrector_needed:
                sparsity_branch_corrector = str(self.robot.get_parent_id(inds[0]) - inds[0])
            adjustments = ""
        self.gen_add_code_line("int dq_flag = col == col_du;")
        if adjustments:
            self.gen_add_code_line(adjustments)
        self.gen_add_code_line("int dst_adjust = (col_du >= " + df_col_that_is_jid_cpp + ") * 6 * " + sparsity_branch_corrector + "; // adjust for sparsity compression offsets")
        self.gen_add_code_line("int du_col_offset = dq_flag * " + str(Offset_df_dq) + " + !dq_flag * " + str(Offset_df_dqd) + " + 6*col_du;")
        self.gen_add_code_line("T *dst = &s_temp[du_col_offset + 6*" + df_col_offset_for_parent_cpp + " + dst_adjust + row];")
        self.gen_add_code_lines(["T update_val = dot_prod<T,6,1,1>(&s_XImats[36*" + jid + " + 6*row],&s_temp[du_col_offset + 6*" + df_col_offset_for_jid_cpp + "])",
                                 "              + dq_flag * (col_du == " + df_col_that_is_jid_cpp + ") * s_temp[" + str(Offset_MxXv) + " + 6*" + jid + " + row];"])
        # check for repeated parent and add atomics
        if self.robot.has_repeated_parents(inds):
            self.gen_add_code_line("// Atomics required for shared parent")
            self.gen_add_code_line("atomicAdd(dst,update_val);")
        else:
            self.gen_add_code_line("*dst += update_val;")
        self.gen_add_end_control_flow()
        self.gen_add_sync(use_thread_group)

        if self.DEBUG_MODE:
            self.gen_add_sync(use_thread_group)
            self.gen_add_serial_ops(use_thread_group)
            if bfs_level == 0:
                self.gen_add_code_lines(["printf(\"-------------------------\\n\");", \
                                         "printf(\"df/du in bfs waves\\n\");", \
                                         "printf(\"-------------------------\\n\");"])
            self.gen_add_code_line("printf(\"df/du for bfs wave[%d]\\n\"," + str(bfs_level) + ");")
            for ind in self.robot.get_unique_parent_ids(inds):
                self.gen_add_code_lines(["printf(\"df[%d]/dq (parent update)\\n\"," + str(ind) + ");", \
                                         "printMat<T,6," + str(df_cols_per_jid[ind]) + ">(&s_temp[" + str(Offset_df_dq + 6*running_sum_df_cols_per_jid[ind]) + "],6);", \
                                         "printf(\"df[%d]/dqd (parent update)\\n\"," + str(ind) + ");", \
                                         "printMat<T,6," + str(df_cols_per_jid[ind]) + ">(&s_temp[" + str(Offset_df_dqd + 6*running_sum_df_cols_per_jid[ind]) + "],6);"])
            self.gen_add_end_control_flow()
            self.gen_add_sync(use_thread_group)

    if self.DEBUG_MODE:
        self.gen_add_sync(use_thread_group)
        self.gen_add_serial_ops(use_thread_group)
        self.gen_add_code_lines(["printf(\"-------------------------\\n\");", \
                                 "printf(\"Final dvaf/du\\n\");", \
                                 "printf(\"-------------------------\\n\");"])
        for ind in range(n):
            num_cols = self.robot.get_bfs_level_by_id(ind) + 1
            self.gen_add_code_lines(["printf(\"dv[%d]/dq\\n\"," + str(ind) + ");", \
                                     "printMat<T,6," + str(num_cols) + ">(&s_temp[" + str(Offset_dv_dq + 6*running_sum_dva_cols_per_jid[ind]) + "],6);"])
        for ind in range(n):
            num_cols = self.robot.get_bfs_level_by_id(ind) + 1
            self.gen_add_code_lines(["printf(\"dv[%d]/dqd\\n\"," + str(ind) + ");", \
                                     "printMat<T,6," + str(num_cols) + ">(&s_temp[" + str(Offset_dv_dqd + 6*running_sum_dva_cols_per_jid[ind]) + "],6);"])
        for ind in range(n):
            num_cols = self.robot.get_bfs_level_by_id(ind) + 1
            self.gen_add_code_lines(["printf(\"da[%d]/dq\\n\"," + str(ind) + ");", \
                                     "printMat<T,6," + str(num_cols) + ">(&s_temp[" + str(Offset_da_dq + 6*running_sum_dva_cols_per_jid[ind]) + "],6);"])
        for ind in range(n):
            num_cols = self.robot.get_bfs_level_by_id(ind) + 1
            self.gen_add_code_lines(["printf(\"da[%d]/dqd\\n\"," + str(ind) + ");", \
                                     "printMat<T,6," + str(num_cols) + ">(&s_temp[" + str(Offset_da_dqd + 6*running_sum_dva_cols_per_jid[ind]) + "],6);"])
        for ind in range(n):
            num_cols = self.robot.get_bfs_level_by_id(ind) + 1
            self.gen_add_code_lines(["printf(\"df[%d]/dq\\n\"," + str(ind) + ");", \
                                     "printMat<T,6," + str(num_cols) + ">(&s_temp[" + str(Offset_df_dq + 6*running_sum_df_cols_per_jid[ind]) + "],6);"])
        for ind in range(n):
            num_cols = self.robot.get_bfs_level_by_id(ind) + 1
            self.gen_add_code_lines(["printf(\"df[%d]/dqd\\n\"," + str(ind) + ");", \
                                     "printMat<T,6," + str(num_cols) + ">(&s_temp[" + str(Offset_df_dqd + 6*running_sum_df_cols_per_jid[ind]) + "],6);"])
        self.gen_add_end_control_flow()

    # extract dc/du
    self.gen_add_code_line("// Finally dc[i]/du = S[i]^T*df[i]/du")
    _, S_ind_cpp, _, df_col_offset_for_jid_cpp, _, _, _, _ = self.gen_topology_helpers_pointers_for_cpp(list(range(n)))
    # Note that for a serial chain this is straightforward (all df are size n) but otherwise gets complicated
    if self.robot.is_serial_chain():
        self.gen_add_parallel_loop("ind",str(2*n*n),use_thread_group)
        self.gen_add_code_line("int jid = ind % " + str(n) + "; int jid_dq_qd = ind / " + str(n) + "; " + 
                               "int jid_du = jid_dq_qd % " + str(n) + "; int dq_flag = jid_du == jid_dq_qd;")
        self.gen_add_code_lines(["int Offset_src = dq_flag * " + str(Offset_df_dq) + " + !dq_flag * " + str(Offset_df_dqd) + \
                                    " + 6 * " + str(n) + " * jid + 6 * jid_du + " + S_ind_cpp + ";",
                                 "int Offset_dst = !dq_flag * " + str(n*n) + " + " + str(n) + " * jid_du + jid;"])
        self.gen_add_code_line("s_dc_du[Offset_dst] = s_temp[Offset_src];")
        self.gen_add_end_control_flow()
        self.gen_add_sync(use_thread_group)

    else:
        self.gen_add_parallel_loop("jid_dq_qd",str(2*n),use_thread_group)
        self.gen_add_code_line("int jid = jid_dq_qd % " + str(n) + "; int dq_flag = jid == jid_dq_qd;")
        # now we need to get a local pointer and loop over all n filling in 0 or col data based on the local pointer
        # and the specific topology of the robot
        self.gen_add_code_line("// Note that this gets a tad complicated due to memory compression and variable column length")
        self.gen_add_code_line("//    so we need to fully unroll the loop -- this will not be the most efficient for a serial")
        self.gen_add_code_line("//    chain manipulator but will generalize to branched robots")
        self.gen_add_code_lines(["int Offset_src = dq_flag * " + str(Offset_df_dq) + " + !dq_flag * " + str(Offset_df_dqd) + \
                                                 " + 6*" + df_col_offset_for_jid_cpp + " + " + S_ind_cpp + ";",
                                 "int Offset_dst = !dq_flag * " + str(n*n) + " + jid; bool flag = 0;"])
        for djid in range(n):
            self.gen_add_code_line("// dc[jid]/du[" + str(djid) + "]")
            # extract all the inds we care about for this du
            is_in_subtree_or_ancestor = [self.robot.get_is_in_subtree_of(djid,ind) or self.robot.get_is_ancestor_of(djid,ind) for ind in range(n)]
            non_zero_inds = [i for (i, x) in enumerate(is_in_subtree_or_ancestor) if x == True]
            # then set the if statement (if applicable)
            if len(non_zero_inds) != n:
                if len(non_zero_inds) > n/2:
                    zero_inds = list(set(list(range(n))).difference(set(non_zero_inds)))
                    jid_du_check_for_jid = self.gen_var_not_in_list("jid",[str(i) for i in zero_inds])
                else:
                    jid_du_check_for_jid = self.gen_var_in_list("jid",[str(i) for i in non_zero_inds])
                self.gen_add_code_line("flag = " + jid_du_check_for_jid + ";")
                # compute the val and pointer updates accordingly
                self.gen_add_code_line("s_dc_du[Offset_dst] = flag*s_temp[Offset_src]; Offset_src += flag*6; Offset_dst += " + str(n) + ";")
            else: # else everyone updates and updates their pointer
                self.gen_add_code_line("s_dc_du[Offset_dst] = s_temp[Offset_src]; Offset_src += 6; Offset_dst += " + str(n) + ";")
        self.gen_add_end_control_flow()
        self.gen_add_sync(use_thread_group)

    if self.DEBUG_MODE:
        self.gen_add_sync(use_thread_group)
        self.gen_add_serial_ops(use_thread_group)
        self.gen_add_code_lines(["printf(\"-------------------------\\n\");", \
                                 "printf(\"Final dc/du\\n\");", \
                                 "printf(\"-------------------------\\n\");"])
        self.gen_add_code_lines(["printf(\"dc/dq\\n\");", \
                                 "printMat<T," + str(n) + "," + str(n) + ">(&s_dc_du[0]," + str(n) + ");", \
                                 "printf(\"dc/dqd\\n\");", \
                                 "printMat<T," + str(n) + "," + str(n) + ">(&s_dc_du[" + str(n*n) + "]," + str(n) + ");"])
        self.gen_add_end_control_flow()
        self.gen_add_sync(use_thread_group)

    self.gen_add_end_function()

def gen_inverse_dynamics_gradient_device(self, use_thread_group = False, use_qdd_input = False):
    n = self.robot.get_num_pos()
    # construct the boilerplate and function definition
    func_params = ["s_dc_du is a pointer to memory for the final result of size 2*NUM_JOINTS*NUM_JOINTS = " + str(2*n*n), \
                   "s_q is the vector of joint positions", \
                   "s_qd is the vector of joint velocities", \
                   "d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)", \
                   "gravity is the gravity constant"]
    func_def_start = "void inverse_dynamics_gradient_device(T *s_dc_du, const T *s_q, const T *s_qd, "
    func_def_end = "const robotModel<T> *d_robotModel, const T gravity) {"
    func_notes = []
    if use_thread_group:
        func_def_start += "cgrps::thread_group tgrp, "
        func_params.insert(0,"tgrp is the handle to the thread_group running this function")
    if use_qdd_input:
        func_def_start += "const T *s_qdd, "
        func_params.insert(-2,"s_qdd is the vector of joint accelerations")
    else:
        func_notes.append("optimized for qdd = 0")
    func_def = func_def_start + func_def_end
    self.gen_add_func_doc("Computes the gradient of inverse dynamics",func_notes,func_params,None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)
    # add the shared memory variables
    self.gen_add_code_line("__shared__ T s_vaf[" + str(18*n) + "];")
    shared_mem_size = self.gen_inverse_dynamics_gradient_inner_temp_mem_size() if not self.use_dynamic_shared_mem_flag else None
    self.gen_XImats_helpers_temp_shared_memory_code(shared_mem_size)
    # then load/update XI and run the algo
    self.gen_load_update_XImats_helpers_function_call(use_thread_group)
    self.gen_inverse_dynamics_inner_function_call(use_thread_group,False,use_qdd_input)
    self.gen_inverse_dynamics_gradient_inner_function_call(use_thread_group)
    self.gen_add_end_function()

def gen_inverse_dynamics_gradient_kernel_max_temp_mem_size(self):
    n = self.robot.get_num_pos()
    base_size = 2*n + n*2*n + 18*n + n
    temp_mem_size = self.gen_inverse_dynamics_gradient_inner_temp_mem_size()
    return base_size + temp_mem_size

def gen_inverse_dynamics_gradient_kernel(self, use_thread_group = False, use_qdd_input = False, single_call_timing = False):
    n = self.robot.get_num_pos()
    compute_c = True
    # define function def and params
    func_params = ["d_dc_du is a pointer to memory for the final result of size 2*NUM_JOINTS*NUM_JOINTS = " + str(2*n*n), \
                   "d_q_dq is the vector of joint positions and velocities", \
                   "stride_q_qd is the stide between each q, qd", \
                   "d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)", \
                   "gravity is the gravity constant", \
                   "num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)"]
    func_notes = []
    func_def_start = "void inverse_dynamics_gradient_kernel(T *d_dc_du, const T *d_q_qd, const int stride_q_qd, "
    func_def_end = "const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {"
    if use_qdd_input:
        func_def_start += "const T *d_qdd, "
        func_params.insert(-2,"d_qdd is the vector of joint accelerations")
    else:
        func_notes.append("optimized for qdd = 0")
    func_def = func_def_start + func_def_end
    if single_call_timing:
        func_def = func_def.replace("kernel(", "kernel_single_timing(")
    # then generate the code
    self.gen_add_func_doc("Computes the gradient of inverse dynamics",func_notes,func_params,None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__global__")
    self.gen_add_code_line(func_def, True)
    # add shared memory variables
    shared_mem_vars = ["__shared__ T s_q_qd[2*" + str(n) + "]; T *s_q = s_q_qd; T *s_qd = &s_q_qd[" + str(n) + "];", \
                       "__shared__ T s_dc_du[" + str(n*2*n) + "];",
                       "__shared__ T s_vaf[" + str(18*n) + "];"]
    if use_qdd_input:
        shared_mem_vars.insert(-2,"__shared__ T s_qdd[" + str(n) + "]; ")
    self.gen_add_code_lines(shared_mem_vars)
    shared_mem_size = self.gen_inverse_dynamics_gradient_inner_temp_mem_size() if not self.use_dynamic_shared_mem_flag else None
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
        self.gen_inverse_dynamics_inner_function_call(use_thread_group,False,use_qdd_input)
        self.gen_inverse_dynamics_gradient_inner_function_call(use_thread_group)
        self.gen_add_sync(use_thread_group)
        # save to global
        self.gen_kernel_save_result("dc_du",str(n*2*n),str(n*2*n),use_thread_group)
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
        self.gen_inverse_dynamics_inner_function_call(use_thread_group,False,use_qdd_input)
        self.gen_inverse_dynamics_gradient_inner_function_call(use_thread_group)
        self.gen_add_end_control_flow()
        # save to global
        self.gen_kernel_save_result_single_timing("dc_du",str(n*2*n),use_thread_group)
    self.gen_add_end_function()

def gen_inverse_dynamics_gradient_host(self, mode = 0):
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
    func_def_start = "void inverse_dynamics_gradient(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,"
    func_def_end =   "                               const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {"
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
    func_call_start = "inverse_dynamics_gradient_kernel<T><<<block_dimms,thread_dimms,ID_DU_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_dc_du,hd_data->d_q_qd,stride_q_qd,"
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
                                 "gpuErrchk(cudaMemcpy(hd_data->h_dc_du,hd_data->d_dc_du,NUM_JOINTS*2*NUM_JOINTS*" + \
                                    ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyDeviceToHost));",
                                 "gpuErrchk(cudaDeviceSynchronize());"])
    # finally report out timing if requested
    if single_call_timing:
        self.gen_add_code_line("printf(\"Single Call ID_DU %fus\\n\",time_delta_us_timespec(start,end)/static_cast<double>(num_timesteps));")
    self.gen_add_end_function()

def gen_inverse_dynamics_gradient(self, use_thread_group = False):
    # gen the inner code
    self.gen_inverse_dynamics_gradient_inner(use_thread_group)
    # gen the wrapper code for with and without qdd
    self.gen_inverse_dynamics_gradient_device(use_thread_group,True)
    self.gen_inverse_dynamics_gradient_device(use_thread_group,False)
    # and the kernels
    self.gen_inverse_dynamics_gradient_kernel(use_thread_group,True,True)
    self.gen_inverse_dynamics_gradient_kernel(use_thread_group,True,False)
    self.gen_inverse_dynamics_gradient_kernel(use_thread_group,False,True)
    self.gen_inverse_dynamics_gradient_kernel(use_thread_group,False,False)
    # and host wrapeprs
    self.gen_inverse_dynamics_gradient_host(0)
    self.gen_inverse_dynamics_gradient_host(1)
    self.gen_inverse_dynamics_gradient_host(2)