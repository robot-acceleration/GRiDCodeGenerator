def gen_direct_minv_inner_temp_mem_size(self):
    n = self.robot.get_num_pos()
    max_bfs_width = self.robot.get_max_bfs_width()
    return 6*n*n+36*n+6*n+n + 36*2*max_bfs_width

def gen_direct_minv_inner_function_call(self, use_thread_group = False, updated_var_names = None):
    var_names = dict( \
        s_Minv_name = "s_Minv", \
        s_q_name = "s_q", \
        s_temp_name = "s_temp", \
    )
    if updated_var_names is not None:
        for key,value in updated_var_names.items():
            var_names[key] = value
    minv_code_start = "direct_minv_inner<T>(" + var_names["s_Minv_name"] + ", " + var_names["s_q_name"] + ", "
    minv_code_end = var_names["s_temp_name"] + ");"
    minv_code_middle = self.gen_insert_helpers_function_call()
    if use_thread_group:
        minv_code_start = minv_code_start.replace("(","(tgrp, ")
    minv_code = minv_code_start + minv_code_middle + minv_code_end
    self.gen_add_code_line(minv_code)

def gen_direct_minv_inner(self, use_thread_group = False):
    n = self.robot.get_num_pos()
    max_bfs_levels = self.robot.get_max_bfs_level()
    max_bfs_width = self.robot.get_max_bfs_width()
    # construct the boilerplate and function definition
    func_params = ["s_Minv is a pointer to memory for the final result", \
                   "s_q is the vector of joint positions", \
                   "s_temp is a pointer to helper shared memory of size " + str(self.gen_direct_minv_inner_temp_mem_size())]
    func_notes = ["Assumes the XI matricies have already been updated for the given q", \
                  "Outputs a SYMMETRIC_UPPER triangular matrix for Minv"]
    func_def_start = "void direct_minv_inner(T *s_Minv, const T *s_q, "
    func_def_end = "T *s_temp) {"
    func_def_start, func_params = self.gen_insert_helpers_func_def_params(func_def_start, func_params, -1)
    func_def = func_def_start + func_def_end
    if use_thread_group:
        func_def_start += "cgrps::thread_group tgrp, "
        func_params.insert(0,"tgrp is the handle to the thread_group running this function")
    self.gen_add_func_doc("Compute the inverse of the mass matrix",\
                          func_notes,func_params,None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)

    # add shared memory note
    FOffset = 0
    IAOffset = FOffset + 6*n*n
    UOffset = IAOffset + 36*n
    DinvOffset = UOffset + 6*n
    IaOffset = DinvOffset + n
    IaTempOffset = IaOffset + 36*max_bfs_width
    self.gen_add_code_line("// T *s_F = &s_temp[" + str(FOffset) + "]; T *s_IA = &s_temp[" + str(IAOffset) + "]; T *s_U = &s_temp[" + str(UOffset) + "];" + \
                             " T *s_Dinv = &s_temp[" + str(DinvOffset) + "]; T *s_Ia = &s_temp[" + str(IaOffset) + "]; T *s_IaTemp = &s_temp[" + str(IaTempOffset) + "];")

    # set initial IA to I and zero Minv/F
    self.gen_add_code_line("// Initialize IA = I")
    self.gen_add_parallel_loop("ind",str(36*n),use_thread_group)
    self.gen_add_code_line("s_temp[" + str(IAOffset) + " + ind] = s_XImats[" + str(36*n) + " + ind];")
    self.gen_add_end_control_flow()
    self.gen_add_code_line("// Zero Minv and F")
    self.gen_add_parallel_loop("ind",str(n*n*7),use_thread_group)
    self.gen_add_code_line("if(ind < " + str(6*n*n) + "){s_temp[" + str(FOffset) + " + ind] = static_cast<T>(0);}")
    self.gen_add_code_line("else{s_Minv[ind - " + str(6*n*n) + "] = static_cast<T>(0);}")
    self.gen_add_end_control_flow()
    self.gen_add_sync(use_thread_group)

    if self.DEBUG_MODE:
        self.gen_add_sync(use_thread_group)
        self.gen_add_serial_ops(use_thread_group)
        self.gen_add_code_line("printf(\"q\\n\"); printMat<T,1," + str(n) + ">(s_q,1);")
        self.gen_add_code_line("for (int i = 0; i < " + str(n) + "; i++){printf(\"X[%d]\\n\",i); printMat<T,6,6>(&s_XImats[36*i],6);}")
        self.gen_add_code_line("for (int i = 0; i < " + str(n) + "; i++){printf(\"IA_init = I[%d]\\n\",i); printMat<T,6,6>(&s_temp[" + str(IAOffset) + " + 36*i],6);}")
        self.gen_add_end_control_flow()
        self.gen_add_sync(use_thread_group)

    #
    # First compute the Backward Pass in bfs waves
    #
    self.gen_add_code_line("//")
    self.gen_add_code_line("// Backward Pass")
    self.gen_add_code_line("//")
    for bfs_level in range(max_bfs_levels, -1, -1):
        inds = self.robot.get_ids_by_bfs_level(bfs_level)
        joint_names = [self.robot.get_joint_by_id(ind).get_name() for ind in inds]
        link_names = [self.robot.get_link_by_id(ind).get_name() for ind in inds]
        parent_ind_cpp, S_ind_cpp = self.gen_topology_helpers_pointers_for_cpp(inds, NO_GRAD_FLAG = True)
        ind_subtree_inds = []
        subree_counts = []
        for ind in inds:
            ind_subtree_inds.extend([(ind,subInd) for subInd in self.robot.get_subtree_by_id(ind)])
            subree_counts.append(len(self.robot.get_subtree_by_id(ind)))
        subtree_adjust = [sum(subree_counts[:idx]) for idx in range(len(inds) + 1)]

        self.gen_add_code_line("// backward pass updates where bfs_level is " + str(bfs_level))
        self.gen_add_code_line("//     joints are: " + ", ".join(joint_names))
        self.gen_add_code_line("//     links are: " + ", ".join(link_names))

        # U = Scol of IA then D = Srow of U then note that DInv = 1/D, Minv[i,i] = Dinv
        self.gen_add_code_line("// U = IA*S, D = S^T*U, DInv = 1/D, Minv[i,i] = Dinv")
        if len(inds) > 1:
            self.gen_add_parallel_loop("ind",str(6*len(inds)),use_thread_group)
            self.gen_add_code_line("int row = ind % 6;")
            select_var_vals = [("int", "jid", [str(jid) for jid in inds])]
            self.gen_add_multi_threaded_select("ind", "<", [str(6*(i+1)) for i in range(len(inds))], select_var_vals)
            self.gen_add_code_line("int jid6 = 6*jid;")
            jid = "jid"
            jid6 = "jid6"
        else:
            jid = str(inds[0])
            jid6 = str(6*inds[0])
            self.gen_add_parallel_loop("row",str(6),use_thread_group)
        self.gen_add_code_line("s_temp[" + str(UOffset) + " + " + jid6 + " + row] = s_temp[" + str(IAOffset) + " + 6*" + jid6 + " + 6*" + S_ind_cpp + " + row];")
        self.gen_add_code_line("if(row == " + S_ind_cpp + "){", True)
        self.gen_add_code_line("s_temp[" + str(DinvOffset) + " + " + jid + "] = static_cast<T>(1)/s_temp[" + str(UOffset) + " + " + jid6 + " + " + S_ind_cpp + "];")
        self.gen_add_code_line("s_Minv[" + str(n + 1) + " * " + jid + "] = s_temp[" + str(DinvOffset) + " + " + jid + "];")
        self.gen_add_end_control_flow()
        self.gen_add_end_control_flow()
        self.gen_add_sync(use_thread_group)

        if self.DEBUG_MODE:
            self.gen_add_sync(use_thread_group)
            self.gen_add_serial_ops(use_thread_group)
            for ind in inds:
                self.gen_add_code_line("printf(\"U[" + str(ind) + "]\\n\"); printMat<T,1,6>(&s_temp[" + str(UOffset) + " + 6*" + str(ind) + "],1);")
                self.gen_add_code_line("printf(\"Dinv[" + str(ind) + "] = %f\\n\",s_temp[" + str(DinvOffset) + " + " + str(ind) + "]);")
            self.gen_add_code_line("printf(\"Minv after Dinv setting before subtree\\n\"); printMat<T," + str(n) + "," + str(n) + ">(s_Minv," + str(n) + ");")
            self.gen_add_end_control_flow()
            self.gen_add_sync(use_thread_group)

        # then for the subtrees we know that Minv[i,subTreeInds] -= F[i,Srow,SubTreeInds] scalar -> scalar
        #                                and temp comp F[i,:,subTreeInds] += U*Minv[i,subTreeInds] vector*scalar -> vector (only if parent exists)
        # Note that by not supporting looped URDFs we can ensure that subtrees are independent and safe for parallelism
        self.gen_add_code_line("// Minv[i,subTreeInds] -= Dinv*F[i,Srow,SubTreeInds]")
        if bfs_level != 0:
            self.gen_add_code_line("// Temp Comp: F[i,:,subTreeInds] += U*Minv[i,subTreeInds] - to start Fparent Update")
        self.gen_add_parallel_loop("ind",str(len(ind_subtree_inds)),use_thread_group)
        if len(inds) > 1:
            select_var_vals = [("int", "jid", [str(jid) for jid in inds]),
                               ("int", "subTreeAdj", [str(val) for val in subtree_adjust])]
            self.gen_add_multi_threaded_select("ind", "<", [str(subtree_adjust[i+1]) for i in range(len(inds))], select_var_vals, USE_NON_BRANCH_ALWAYS = True)
            self.gen_add_code_line("int jid_subtree = jid + (ind - subTreeAdj); " + \
                                   "int jid_subtree6 = 6*jid_subtree; int jid_subtreeN = " + str(n) + "*jid_subtree;")
            jid = "jid"
            jid_subtree6 = "jid_subtree6"
            jid_subtreeN = "jid_subtreeN"
        else:
            select_var_vals = []
            jid = str(inds[0])
            if len(ind_subtree_inds) > 1:
                self.gen_add_code_line("int jid_subtree6 = 6*(" + jid + " + ind); int jid_subtreeN = " + str(n) + "*(" + jid + " + ind);")
                jid_subtree6 = "jid_subtree6"
                jid_subtreeN = "jid_subtreeN"
            else:
                subId = ind_subtree_inds[0][1]
                jid_subtree6 = str(subId*6)
                jid_subtreeN = str(subId*n)
        self.gen_add_code_line("s_Minv[" + jid_subtreeN + " + " + jid + "] -= s_temp[" + str(DinvOffset) + " + " + jid + "] * " + \
                                                "s_temp[" + str(FOffset) + " + " + str(n*6) + "*" + jid + " + " + jid_subtree6 + " + " + S_ind_cpp + "];")
        if bfs_level != 0:
            self.gen_add_code_line("for(int row = 0; row < 6; row++) {", True)
            self.gen_add_code_line("s_temp[" + str(FOffset) + " + " + str(n*6) + "*" + jid + " + " + jid_subtree6 + " + row] += " + \
                                        "s_temp[" + str(UOffset) + " + 6*" + jid + " + row] * s_Minv[" + jid_subtreeN + " + " + jid + "];")
            self.gen_add_end_control_flow()
        self.gen_add_end_control_flow()

        if self.DEBUG_MODE:
            self.gen_add_sync(use_thread_group)
            self.gen_add_serial_ops(use_thread_group)
            self.gen_add_code_line("printf(\"Minv after subtree updates\\n\"); printMat<T," + str(n) + "," + str(n) + ">(s_Minv," + str(n) + ");")
            if bfs_level != 0:
                for ind in inds:
                    self.gen_add_code_line("printf(\"F Temp += U*Minv[" + str(ind) + "]\\n\"); printMat<T,6," + str(n) + ">(&s_temp[" + str(FOffset + n*6*ind) + "],6);")
            self.gen_add_end_control_flow()
            self.gen_add_sync(use_thread_group)

        # Then start the IA update (if there is a parent) with Ia = IA[ind] - np.outer(U[ind,:],Dinv[ind]*U[ind,:])
        if bfs_level != 0:
            self.gen_add_code_line("// Ia = IA - U^T Dinv U | to start IAparent Update")
            self.gen_add_parallel_loop("ind",str(36*len(inds)),use_thread_group)
            if len(inds) > 1:
                select_var_vals = [("int", "jid", [str(jid) for jid in inds])]
                self.gen_add_multi_threaded_select("ind", "<", [str(36*(i+1)) for i in range(len(inds))], select_var_vals)
                self.gen_add_code_line("int ind36 = (ind % 36); int row = ind36 % 6; int col = ind36 / 6; int jid6 = 6*jid;")
                self.gen_add_code_line("s_temp[" + str(IaOffset) + " + ind] = s_temp[" + str(IAOffset) + " + 6*jid6 + ind36] - " + \
                  "(s_temp[" + str(UOffset) + " + jid6 + row] * s_temp[" + str(DinvOffset) + " + jid] * s_temp[" + str(UOffset) + " + jid6 + col]);")
            else:
                jid = inds[0]
                self.gen_add_code_line("int row = ind % 6; int col = ind / 6;")
                self.gen_add_code_line("s_temp[" + str(IaOffset) + " + ind] = s_temp[" + str(IAOffset + 36*jid) + " + ind] - " + \
                  "(s_temp[" + str(UOffset + 6*jid) + " + row] * s_temp[" + str(DinvOffset + jid) + "] * s_temp[" + str(UOffset + 6*jid) + " + col]);")
            self.gen_add_end_control_flow()
        self.gen_add_sync(use_thread_group)

        if self.DEBUG_MODE:
            self.gen_add_sync(use_thread_group)
            self.gen_add_serial_ops(use_thread_group)
            for i in range(len(inds)):
                self.gen_add_code_lines(["printf(\"Ia[" + str(inds[i]) + "]\\n\");",
                                         "printMat<T,6,6>(&s_temp[" + str(IaOffset + 36*i) + "],6);"])
            self.gen_add_end_control_flow()
            self.gen_add_sync(use_thread_group)

        if bfs_level != 0:
            # then for the subtrees we can do (in parallel by both subtree and row)
            #                 F[parent_ind,:,subTreeInds] += Xmat^T * F[ind,:,subTreeInds] matrix*vector -> vector
            # At the same time also do next step of IA update: IA_Update_Temp = Xmat^T * Ia
            self.gen_add_code_line("// F[parent_ind,:,subTreeInds] += Xmat^T * F[ind,:,subTreeInds]")
            self.gen_add_code_line("// IA_Update_Temp = Xmat^T * Ia | for IAparent Update")
            self.gen_add_parallel_loop("ind",str(6*len(ind_subtree_inds) + 6*6*len(inds)),use_thread_group)
            self.gen_add_code_line("int row = ind % 6; int col = ind / 6;")
            if len(ind_subtree_inds) > 1:
                if len(inds) > 1:
                    select_var_vals = [("int", "jid", [str(jid) for jid in inds]),
                                       ("int", "subTreeAdj", [str(val) for val in subtree_adjust])]
                    self.gen_add_multi_threaded_select("col", "<", [str(subtree_adjust[i+1]) for i in range(len(inds))], select_var_vals, USE_NON_BRANCH_ALWAYS = True)
                    self.gen_add_code_line("int jid_subtree = jid + (col - subTreeAdj);")

                    jid = "jid"
                    jid_subtree = "jid_subtree"
                else:
                    jid = str(inds[0])
                    jid_subtree = "(" + jid + " + col)"
            else:
                jid = str(inds[0])
                jid_subtree = str(ind_subtree_inds[0][1])
            # do the standard comps first
            self.gen_add_code_lines(["T *src = &s_temp[" + str(FOffset) + " + " + str(6*n) + "*" + jid + " + 6*" + jid_subtree + "]; " + \
                                        "T *dst = &s_temp[" + str(FOffset) + " + " + str(6*n) + "*" + parent_ind_cpp + " + 6*" + jid_subtree + "];"])
            # adjust for temp comps
            self.gen_add_code_line("// adjust for temp comps")
            self.gen_add_code_line("if (col >= " + str(len(ind_subtree_inds)) + ") {",True)
            self.gen_add_code_lines(["col -= " + str(len(ind_subtree_inds)) + "; " + \
                                        "src = &s_temp[" + str(IaOffset) + " + 6*col]; " + \
                                        "dst = &s_temp[" + str(IaTempOffset) + " + 6*col];"])
            if len(inds) > 1:
                self.gen_add_code_line("int jid_selector = col / 6;")
                self.gen_add_multi_threaded_select("jid_selector", "==", [str(i) for i in range(len(inds))], [(None, "jid", [str(ind) for ind in inds])])
            self.gen_add_end_control_flow()
            # then do the computation
            self.gen_add_code_line("dst[row] = dot_prod<T,6,1,1>(&s_XImats[36*" + jid + " + 6*row],src);")
            self.gen_add_end_control_flow()
            self.gen_add_sync(use_thread_group)

            if self.DEBUG_MODE:
                self.gen_add_sync(use_thread_group)
                self.gen_add_serial_ops(use_thread_group)
                for i in range(len(inds)):
                    self.gen_add_code_lines(["printf(\"F[" + str(self.robot.get_parent_id(inds[i])) + "] = X^T F[" + str(inds[i]) + "]\\n\");",
                                             "printMat<T,6," + str(n) + ">(&s_temp[" + str(FOffset + n*6*self.robot.get_parent_id(inds[i])) + "],6);",
                                             "printf(\"Ia*X[" + str(inds[i]) + "]\\n\");",
                                             "printMat<T,6,6>(&s_temp[" + str(IaTempOffset + 36*i) + "],6);"])
                self.gen_add_end_control_flow()
                self.gen_add_sync(use_thread_group)

            # Finally IA[parent_ind] += IA_Update_Temp * Xmat
            self.gen_add_code_line("// IA[parent_ind] += IA_Update_Temp * Xmat")
            self.gen_add_parallel_loop("ind",str(6*6*len(inds)),use_thread_group)
            self.gen_add_code_line("int col = ind / 6; int row = ind % 6;")
            if len(inds) > 1:
                self.gen_add_code_line("int col_max6 = col % 6; int jid_ind = col / 6;")
                select_var_vals = [("int", "jid", [str(jid) for jid in inds])]
                self.gen_add_multi_threaded_select("jid_ind", "==", [str(i) for i in range(len(inds))], select_var_vals)
                self.gen_add_code_line("T * src = &s_temp[" + str(IaTempOffset) + " + 36*jid_ind + row]; " + \
                                        "T * dst = &s_temp[" + str(IAOffset) + " + 36*" + parent_ind_cpp + " + 6*col_max6 + row];")
                dot_prod_code = "dot_prod<T,6,6,1>(src,&s_XImats[36*jid + 6*col_max6])"
                if self.robot.has_repeated_parents(inds):
                    self.gen_add_code_line("// Atomics required for shared parent")
                    self.gen_add_code_line("T val = " + dot_prod_code + ";")
                    self.gen_add_code_line("atomicAdd(dst,val);")
                else:
                    self.gen_add_code_line("*dst += " + dot_prod_code + ";")
            else:
                jid = inds[0]
                self.gen_add_code_line("s_temp[" + str(IAOffset + 36*self.robot.get_parent_id(jid)) + " + 6*col + row] += dot_prod<T,6,6,1>(" + \
                                            "&s_temp[" + str(IaTempOffset) + " + row],&s_XImats[" + str(36*jid) + " + 6*col]);")
            self.gen_add_end_control_flow()
            self.gen_add_sync(use_thread_group)

            if self.DEBUG_MODE:
                self.gen_add_sync(use_thread_group)
                self.gen_add_serial_ops(use_thread_group)
                for ind in inds:
                    self.gen_add_code_lines(["printf(\"IA[" + str(self.robot.get_parent_id(ind)) + "] = X^T*(Ia*X)\\n\");",
                                             "printMat<T,6,6>(&s_temp[" + str(IAOffset + 36*self.robot.get_parent_id(ind)) + "],6);"])
                self.gen_add_end_control_flow()
                self.gen_add_sync(use_thread_group)

    if self.DEBUG_MODE:
        self.gen_add_sync(use_thread_group)
        self.gen_add_serial_ops(use_thread_group)
        self.gen_add_code_lines(["printf(\"-------------------\\n\");", \
                                 "printf(\"After Backward Pass\\n\");", \
                                 "printf(\"-------------------\\n\");", \
                                 "printf(\"U\\n\"); printMat<T,6," + str(n) + ">(&s_temp[" + str(UOffset + 6*i) + "],6);", \
                                 "printf(\"Dinv\\n\"); printMat<T,1," + str(n) + ">(&s_temp[" + str(DinvOffset) + "],1);"])
        for i in range(n):
            self.gen_add_code_line("printf(\"F[%d]\\n\"," + str(i) + "); printMat<T,6," + str(n) + ">(&s_temp[" + str(FOffset + 6*n*i) + "],6);")
        self.gen_add_code_line("printf(\"Minv\\n\"); printMat<T," + str(n) + "," + str(n) + ">(s_Minv," + str(n) + ");")
        self.gen_add_code_line("printf(\"-------------------\\n\");")
        self.gen_add_end_control_flow()
        self.gen_add_sync(use_thread_group)

    #
    # Then compute the Forwad Pass
    #    Note that due to the i: operation we need to go serially over all n
    #
    self.gen_add_code_line("//")
    self.gen_add_code_line("// Forward Pass")
    self.gen_add_code_line("//   Note that due to the i: operation we need to go serially over all n")
    self.gen_add_code_line("//")
    for jid in range(n):
        self.gen_add_code_line("// forward pass for jid: " + str(jid))
        jid_parent = self.robot.get_parent_id(jid)
        jid_cols = list(range(jid,n))
        SInd = str(self.robot.get_S_by_id(jid).tolist().index(1))

        # Minv[i,i:] -= Dinv*U^T*Xmat*F[parent,:,i:] across cols i...N
        # F[i,:,i:] = S^T * Minv[i,i:] + Xmat*F[parent,:,i:] across cols i...N
        if jid_parent != -1:
            self.gen_add_code_line("// Minv[i,i:] -= Dinv*U^T*Xmat*F[parent,:,i:] across cols i...N")
            self.gen_add_code_line("// F[i,:,i:] = S * Minv[i,i:] + Xmat*F[parent,:,i:] across cols i...N")
            # note that we can first compute the same temp part used in both
            self.gen_add_code_line("//   Start this step with F[i,:,i:] = Xmat*F[parent,:,i:] and")
            self.gen_add_parallel_loop("ind",str(6*len(jid_cols)),use_thread_group)
            self.gen_add_code_line("int row = ind % 6; int col_ind = ind - row + " + str(6*jid) + ";")
            self.gen_add_code_line("s_temp[" + str(FOffset + 6*n*jid) + " + col_ind + row] = " + \
                                       "dot_prod<T,6,6,1>(&s_XImats[" + str(36*jid) + " + row], " + \
                                       "&s_temp[" + str(FOffset + 6*n*jid_parent) + " + col_ind]);")
            self.gen_add_end_control_flow()
            self.gen_add_sync(use_thread_group)

            if self.DEBUG_MODE:
                self.gen_add_sync(use_thread_group)
                self.gen_add_serial_ops(use_thread_group)
                self.gen_add_code_lines(["printf(\"F[i,:,i:] = Xmat*F[parent,:,i:] for i[" + str(jid) + "]\\n\");", \
                                         "printMat<T,6," + str(n) + ">(&s_temp[" + str(FOffset) + " + " + str(6*n*jid) + "],6);"])
                self.gen_add_end_control_flow()
                self.gen_add_sync(use_thread_group)

            # Then finish it up by summing across the U^T mult and the -= by that times Dinv for Minv
            # and then taking that balue and adding it to the Srow
            self.gen_add_code_line("//   Finish this step with Minv[i,i:] -= Dinv*U^T*F[i,:,i:]")
            self.gen_add_code_line("//     and then update F[i,:,i:] += S*Minv[i,i:]")
            self.gen_add_parallel_loop("ind",str(len(jid_cols)),use_thread_group)
            self.gen_add_code_line("int col_ind = ind + " + str(jid) + ";")
            self.gen_add_code_line("T *s_Fcol = &s_temp[" + str(FOffset + 6*n*jid) + " + 6*col_ind];");
            self.gen_add_code_line("s_Minv[" + str(n) + " * col_ind + " + str(jid) + "] -= " + \
                                   "s_temp[" + str(DinvOffset + jid) + "] * " + \
                                   "dot_prod<T,6,1,1>(s_Fcol,&s_temp[" + str(UOffset + 6*jid) + "]);")
            if jid < n-1: # skip redundant comp on last loop
                self.gen_add_code_line("s_Fcol[" + SInd + "] += s_Minv[" + str(n) + " * col_ind + " + str(jid) + "];")
            self.gen_add_end_control_flow()
            self.gen_add_sync(use_thread_group)

            if self.DEBUG_MODE:
                self.gen_add_sync(use_thread_group)
                self.gen_add_serial_ops(use_thread_group)
                self.gen_add_code_lines(["printf(\"Minv[i,i:] -= Dinv*U^T*F[i,:,i:] for i = %d\\n\"," + str(jid) + ");", \
                                         "printMat<T," + str(n) + "," + str(n) + ">(s_Minv," + str(n) + ");"])
                if jid < n-1: # redundant comp on last loop
                    self.gen_add_code_lines(["printf(\"F[i,:,i:] += S*Minv[i,i:]\");", \
                                             "printMat<T,6," + str(n) + ">(&s_temp[" + str(FOffset + 6*n*jid) + "],6);"])
                self.gen_add_end_control_flow()
                self.gen_add_sync(use_thread_group)

        elif jid < n-1: # redundant comp on last loop
            self.gen_add_code_line("// F[i,:,i:] = S * Minv[i,i:] as parent is base so rest is skipped")
            self.gen_add_parallel_loop("ind",str(6*len(jid_cols)),use_thread_group)
            self.gen_add_code_line("int row = ind % 6; int col = ind / 6;")
            self.gen_add_code_line("s_temp[" + str(FOffset + 6*n*jid + 6*jid) + " + ind] = (row == " + SInd + ") * " + \
                                        "s_Minv[" + str(n*jid + jid) + " + " + str(n) + " * col];")
            self.gen_add_end_control_flow()
            self.gen_add_sync(use_thread_group)

            if self.DEBUG_MODE:
                self.gen_add_sync(use_thread_group)
                self.gen_add_serial_ops(use_thread_group)
                self.gen_add_code_lines(["printf(\"F[i,:,i:] += S*Minv[i,i:] for i = %d\\n\"," + str(jid) + ");", \
                                         "printMat<T,6," + str(n) + ">(&s_temp[" + str(FOffset + 6*n*jid) + "],6);"])
                self.gen_add_end_control_flow()
                self.gen_add_sync(use_thread_group)
    self.gen_add_end_function()

def gen_direct_minv_device(self, use_thread_group = False):
    # construct the boilerplate and function definition
    func_params = ["s_Minv is a pointer to memory for the final result", \
                   "s_q is the vector of joint positions", \
                   "d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)"]
    func_def = "void direct_minv_device(T *s_Minv, const T *s_q, const robotModel<T> *d_robotModel){"
    func_notes = ["Outputs a SYMMETRIC_UPPER triangular matrix for Minv"]
    if use_thread_group:
        func_def = func_def.replace("(","(cgrps::thread_group tgrp, ")
        func_params.insert(0,"tgrp is the handle to the thread_group running this function")
    # then generate the code
    self.gen_add_func_doc("Compute the inverse of the mass matrix",func_notes,func_params,None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)
    # add the shared memory variables
    shared_mem_size = self.gen_direct_minv_inner_temp_mem_size() if not self.use_dynamic_shared_mem_flag else None
    self.gen_XImats_helpers_temp_shared_memory_code(shared_mem_size)
    # then load/update XI and run the algo
    self.gen_load_update_XImats_helpers_function_call(use_thread_group)
    self.gen_direct_minv_inner_function_call(use_thread_group)
    self.gen_add_end_function()

def gen_direct_minv_kernel(self, use_thread_group = False, single_call_timing = False):
    n = self.robot.get_num_pos()
    # define function def and params
    func_params = ["d_Minv is a pointer to memory for the final result", \
                   "d_q is the vector of joint positions", \
                   "d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)",
                   "num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)"]
    func_def = "void direct_minv_kernel(T *d_Minv, const T *d_q, const int stride_q, const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS){"
    func_notes = ["Outputs a SYMMETRIC_UPPER triangular matrix for Minv"]
    if single_call_timing:
        func_def = func_def.replace("kernel(", "kernel_single_timing(")
    # then generate the code
    self.gen_add_func_doc("Compute the inverse of the mass matrix",func_notes,func_params,None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__global__")
    self.gen_add_code_line(func_def, True)
    # add shared memory variables
    shared_mem_vars = ["__shared__ T s_q[" + str(n) + "];", \
                       "__shared__ T s_Minv[" + str(n*n) + "];"]
    self.gen_add_code_lines(shared_mem_vars)
    shared_mem_size = self.gen_direct_minv_inner_temp_mem_size() if not self.use_dynamic_shared_mem_flag else None
    self.gen_XImats_helpers_temp_shared_memory_code(shared_mem_size)
    if use_thread_group:
        self.gen_add_code_line("cgrps::thread_group tgrp = TBD;")
    if not single_call_timing:
        # load to shared mem and loop over blocks to compute all requested comps
        self.gen_add_parallel_loop("k","NUM_TIMESTEPS",use_thread_group,block_level = True)
        self.gen_kernel_load_inputs("q","stride_q",str(n),use_thread_group)
        # compute
        self.gen_add_code_line("// compute")
        self.gen_load_update_XImats_helpers_function_call(use_thread_group)
        self.gen_direct_minv_inner_function_call(use_thread_group)
        self.gen_add_sync(use_thread_group)
        # save to global
        self.gen_kernel_save_result("Minv",str(n*n),str(n*n),use_thread_group)
        self.gen_add_end_control_flow()
    else:
        #repurpose NUM_TIMESTEPS for number of timing reps
        self.gen_kernel_load_inputs_single_timing("q",str(n),use_thread_group)
        # then compute in loop for timing
        self.gen_add_code_line("// compute with NUM_TIMESTEPS as NUM_REPS for timing")
        self.gen_add_code_line("for (int rep = 0; rep < NUM_TIMESTEPS; rep++){", True)
        self.gen_load_update_XImats_helpers_function_call(use_thread_group)
        self.gen_direct_minv_inner_function_call(use_thread_group)
        self.gen_add_end_control_flow()
        # save to global
        self.gen_kernel_save_result_single_timing("Minv",str(n*n),use_thread_group)
    self.gen_add_end_function()

def gen_direct_minv_host(self, mode = 0):
    # default is to do the full kernel call -- options are for single timing or compute only kernel wrapper
    single_call_timing = True if mode == 1 else False
    compute_only = True if mode == 2 else False

    # define function def and params
    func_params = ["hd_data is the packaged input and output pointers", \
                   "d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)", \
                   "num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)", \
                   "streams are pointers to CUDA streams for async memory transfers (if needed)"]
    func_notes = []
    func_def_start = "void direct_minv(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps,"
    func_def_end =   "                 const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {"
    if single_call_timing:
        func_def_start = func_def_start.replace("(", "_single_timing(")
        func_def_end = "              " + func_def_end
    if compute_only:
        func_def_start = func_def_start.replace("(", "_compute_only(")
        func_def_end = "             " + func_def_end.replace(", cudaStream_t *streams", "")
    # then generate the code
    self.gen_add_func_doc("Compute the inverse of the mass matrix",func_notes,func_params,None)
    self.gen_add_code_line("template <typename T, bool USE_COMPRESSED_MEM = false>")
    self.gen_add_code_line("__host__")
    self.gen_add_code_line(func_def_start)
    self.gen_add_code_line(func_def_end, True)
    func_call_start = "direct_minv_kernel<T><<<block_dimms,thread_dimms,MINV_DYNAMIC_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_Minv,hd_data->d_q,stride_q,"
    func_call_end = "d_robotModel,num_timesteps);"
    if single_call_timing:
        func_call_start = func_call_start.replace("kernel<T>","kernel_single_timing<T>")
    if not compute_only:
        # start code with memory transfer
        self.gen_add_code_lines(["// start code with memory transfer", \
                                 "int stride_q;", \
                                 "if (USE_COMPRESSED_MEM) {stride_q = NUM_JOINTS; " + \
                                    "gpuErrchk(cudaMemcpyAsync(hd_data->d_q,hd_data->h_q,stride_q*" + \
                                    ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyHostToDevice,streams[0]));}", \
                                 "else {stride_q = 3*NUM_JOINTS; " + \
                                    "gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q*" + \
                                    ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyHostToDevice,streams[0]));}", \
                                 "gpuErrchk(cudaDeviceSynchronize());"])
    else:
        self.gen_add_code_line("int stride_q = USE_COMPRESSED_MEM ? NUM_JOINTS: 3*NUM_JOINTS;")
    # then compute
    self.gen_add_code_line("// then call the kernel")
    func_call = "if (USE_COMPRESSED_MEM) {" + func_call_start + func_call_end + "}"
    func_call2 = "else                    {" + func_call_start.replace("hd_data->d_q","hd_data->d_q_qd_u") + func_call_end + "}"
    func_call_code = [func_call, func_call2, "gpuErrchk(cudaDeviceSynchronize());"]
    # wrap function call in timing (if needed)
    if single_call_timing:
        func_call_code.insert(0,"struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);")
        func_call_code.append("clock_gettime(CLOCK_MONOTONIC,&end);")
    self.gen_add_code_lines(func_call_code)
    if not compute_only:
        # then transfer memory back
        self.gen_add_code_lines(["// finally transfer the result back", \
                                 "gpuErrchk(cudaMemcpy(hd_data->h_Minv,hd_data->d_Minv,NUM_JOINTS*NUM_JOINTS*" + \
                                    ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyDeviceToHost));",
                                 "gpuErrchk(cudaDeviceSynchronize());"])
    # finally report out timing if requested
    if single_call_timing:
        self.gen_add_code_line("printf(\"Single Call Minv %fus\\n\",time_delta_us_timespec(start,end)/static_cast<double>(num_timesteps));")
    self.gen_add_end_function()

def gen_direct_minv(self, use_thread_group = False):
    # gen inner
    self.gen_direct_minv_inner(use_thread_group)
    # and device wrapper
    self.gen_direct_minv_device(use_thread_group)
    # and kernel wrappers
    self.gen_direct_minv_kernel(use_thread_group, True)
    self.gen_direct_minv_kernel(use_thread_group, False)
    # and host function call wrappers
    self.gen_direct_minv_host(0)
    self.gen_direct_minv_host(1)
    self.gen_direct_minv_host(2)
