import sympy as sp

def gen_init_XImats(self, include_base_inertia = False):
    # add function description
    if include_base_inertia:
        self.gen_add_func_doc("Initializes the Xmats and Imats in GPU memory", \
            ["Memory order is X[0...N], Ibase, I[0...N]"], \
            [],"A pointer to the XI memory in the GPU")
    else:
        self.gen_add_func_doc("Initializes the Xmats and Imats in GPU memory", \
            ["Memory order is X[0...N], I[0...N]"], \
            [],"A pointer to the XI memory in the GPU")
    # add the function start boilerplate
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__host__")
    self.gen_add_code_line("T* init_XImats() {", True)
    # allocate CPU memory
    n = self.robot.get_num_pos()
    XI_size = 36*2*n + (36 if include_base_inertia else 0)
    self.gen_add_code_line("T *h_XImats = (T *)malloc(" + str(XI_size) + "*sizeof(T));")
    # loop through Xmats and add all constant values from the sp matrix (initialize non-constant to 0)
    Xmats = self.robot.get_Xmats_ordered_by_id()
    for ind in range(len(Xmats)):
        self.gen_add_code_line("// X[" + str(ind) + "]")
        for col in range(6):
            for row in range(6):
                val = Xmats[ind][row,col]
                if not val.is_constant(): # initialize to 0
                    val = 0
                str_val = str(val)
                cpp_ind = self.gen_static_array_ind_3d(ind,col,row)
                self.gen_add_code_line("h_XImats[" + str(cpp_ind) + "] = static_cast<T>(" + str_val + ");")
    # loop through Imats and add all values (inertias are always constant and stored as np arrays)
    Imats = self.robot.get_Imats_ordered_by_id()
    if not include_base_inertia:
        Imats = Imats[1:]
    mem_offset = len(Xmats)
    for ind in range(len(Imats)):
        if include_base_inertia and ind == 0:
            self.gen_add_code_line("// Base Inertia")
        else:
            self.gen_add_code_line("// I[" + str(ind-int(include_base_inertia)) + "]")
        for col in range(6):
            for row in range(6):
                str_val = str(Imats[ind][row,col])
                cpp_ind = str(self.gen_static_array_ind_3d(ind + mem_offset,col,row))
                self.gen_add_code_line("h_XImats[" + cpp_ind + "] = static_cast<T>(" + str_val + ");")
    # allocate and transfer data to the GPU, free CPU memory and return the pointer to the memory
    self.gen_add_code_line("T *d_XImats; gpuErrchk(cudaMalloc((void**)&d_XImats," + str(XI_size) + "*sizeof(T)));")
    self.gen_add_code_line("gpuErrchk(cudaMemcpy(d_XImats,h_XImats," + str(XI_size) + "*sizeof(T),cudaMemcpyHostToDevice));")
    self.gen_add_code_line("free(h_XImats);")
    self.gen_add_code_line("return d_XImats;")
    # add the function end
    self.gen_add_end_function()

def gen_load_update_XImats_helpers_temp_mem_size(self):
    n = self.robot.get_num_pos()
    return 2*n

def gen_load_update_XImats_helpers_function_call(self, use_thread_group = False, updated_var_names = None):
    var_names = dict( \
        s_XImats_name = "s_XImats", \
        d_robotModel_name = "d_robotModel", \
        s_q_name = "s_q", \
        s_temp_name = "s_temp", \
        s_topology_helpers_name = "s_topology_helpers", \
    )
    if updated_var_names is not None:
        for key,value in updated_var_names.items():
            var_names[key] = value
    code_start = "load_update_XImats_helpers<T>(" + var_names["s_XImats_name"] + ", " + var_names["s_q_name"] + ", "
    code_end = var_names["d_robotModel_name"] + ", " + var_names["s_temp_name"] + ");"
    n = self.robot.get_num_pos()
    if not self.robot.is_serial_chain() or not self.robot.are_Ss_identical(list(range(n))):
        code_start += var_names["s_topology_helpers_name"] + ", "
    if use_thread_group:
        code_start = code_start.replace("(","(tgrp, ")
    self.gen_add_code_line(code_start + code_end)

def gen_XImats_helpers_temp_shared_memory_code(self, temp_mem_size = None):
    n = self.robot.get_num_pos()
    if not self.robot.is_serial_chain() or not self.robot.are_Ss_identical(list(range(n))):
        self.gen_add_code_line("__shared__ int s_topology_helpers[" + str(self.gen_topology_helpers_size()) + "];")
    if temp_mem_size is None: # use dynamic shared mem
        self.gen_add_code_line("extern __shared__ T s_XITemp[]; T *s_XImats = s_XITemp; T *s_temp = &s_XITemp[" + str(72*n) + "];")
    else: # use specified static shared mem
        self.gen_add_code_line("__shared__ T s_XImats[" + str(72*n) + "];")
        self.gen_add_code_line("__shared__ T s_temp[" + str(temp_mem_size) + "];")

def gen_load_update_XImats_helpers(self, use_thread_group = False):
    n = self.robot.get_num_pos()
    # add function description
    func_def_start = "void load_update_XImats_helpers("
    func_def_middle = "T *s_XImats, const T *s_q, "
    func_def_end = "const robotModel<T> *d_robotModel, T *s_temp) {"
    func_params = ["s_XImats is the (shared) memory destination location for the XImats",\
        "s_q is the (shared) memory location of the current configuration",\
        "d_robotModel is the pointer to the initialized model specific helpers (XImats, mxfuncs, topology_helpers, etc.)", \
        "s_temp is temporary (shared) memory used to compute sin and cos if needed of size: " + \
                str(self.gen_load_update_XImats_helpers_temp_mem_size())]
    if use_thread_group:
        func_params.insert(0,"tgrp is the handle to the thread_group running this function")
        func_def_start += "cgrps::thread_group tgrp, "
    if not self.robot.is_serial_chain() or not self.robot.are_Ss_identical(list(range(n))):
        func_def_middle += "int *s_topology_helpers, "
        func_params.insert(-2,"s_topology_helpers is the (shared) memory destination location for the topology_helpers")
    func_def = func_def_start + func_def_middle + func_def_end
    # then genearte the code
    self.gen_add_func_doc("Updates the Xmats in (shared) GPU memory acording to the configuration",[],func_params,None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)
    # test to see if we need to compute any trig functions
    Xmats = self.robot.get_Xmats_ordered_by_id()
    use_trig = False
    for mat in Xmats:
        if len(mat.atoms(sp.sin, sp.cos)) > 0:
            use_trig = True
            break
    # if we need trig then compute sin and cos while loading in XI from global to shared (if possible to do async)
    if use_trig and use_thread_group:
        self.gen_add_code_line("cgrps::memcpy_async(tgrp,s_XImats,d_robotModel->d_XImats," + str(72*self.robot.get_num_pos()) + "*sizeof(T));")
        if not self.robot.is_serial_chain() or not self.robot.are_Ss_identical(list(range(n))):
            self.gen_add_code_line("cgrps::memcpy_async(tgrp,s_topology_helpers,d_robotModel->d_topology_helpers," + str(self.gen_topology_helpers_size()) + "*sizeof(int));")
        self.gen_add_parallel_loop("k",str(self.robot.get_num_pos()),use_thread_group)
        # self.gen_add_code_line("sincosf(s_q[k],&s_temp[k],&s_temp[k+" + str(self.robot.get_num_pos()) + "]);")
        self.gen_add_code_line("s_temp[k] = static_cast<T>(sin(s_q[k]));")
        self.gen_add_code_line("s_temp[k+" + str(self.robot.get_num_pos()) + "] = static_cast<T>(cos(s_q[k]));")
        self.gen_add_end_control_flow()
        self.gen_add_code_line("cgrps::wait(tgrp);")
        self.gen_add_sync(use_thread_group)
    # else do them in parallel but sequentially
    elif use_trig:
        self.gen_add_parallel_loop("ind",str(72*self.robot.get_num_pos()),use_thread_group)
        self.gen_add_code_line("s_XImats[ind] = d_robotModel->d_XImats[ind];")
        self.gen_add_end_control_flow()
        if not self.robot.is_serial_chain() or not self.robot.are_Ss_identical(list(range(n))):
            self.gen_add_parallel_loop("ind",str(self.gen_topology_helpers_size()),use_thread_group)
            self.gen_add_code_line("s_topology_helpers[ind] = d_robotModel->d_topology_helpers[ind];")
            self.gen_add_end_control_flow()
        self.gen_add_parallel_loop("k",str(self.robot.get_num_pos()),use_thread_group)
        # self.gen_add_code_line("sincosf(s_q[k],&s_temp[k],&s_temp[k+" + str(self.robot.get_num_pos()) + "]);")
        self.gen_add_code_line("s_temp[k] = static_cast<T>(sin(s_q[k]));")
        self.gen_add_code_line("s_temp[k+" + str(self.robot.get_num_pos()) + "] = static_cast<T>(cos(s_q[k]));")
        self.gen_add_end_control_flow()
        self.gen_add_sync(use_thread_group)
    # else just load in XI from global to shared efficiently
    else:
        self.gen_add_code_line("cgrps::memcpy_async(tgrp,s_XImats,d_robotModel->d_XImats," + str(72*self.robot.get_num_pos()) + ");")
        if not self.robot.is_serial_chain() or not self.robot.are_Ss_identical(list(range(n))):
            self.gen_add_code_line("cgrps::memcpy_async(tgrp,s_topology_helpers,d_robotModel->d_topology_helpers," + str(self.gen_topology_helpers_size()) + "*sizeof(int));")
        self.gen_add_code_line("cgrps::wait(tgrp);")
    # loop through Xmats and update all non-constant values serially
    self.gen_add_serial_ops(use_thread_group)
    for ind in range(n):
        self.gen_add_code_line("// X[" + str(ind) + "]")
        for col in range(3): # TL and BR are identical so only update TL and BL serially
            for row in range(6):
                val = Xmats[ind][row,col]
                if not val.is_constant():
                    # parse the symbolic value into the appropriate array access
                    str_val = str(val)
                    # first check for sin/cos (revolute)
                    str_val = str_val.replace("sin(theta)","s_temp[" + str(ind) + "]")
                    str_val = str_val.replace("cos(theta)","s_temp[" + str(ind + n) + "]")
                    # then just the variable (prismatic)
                    str_val = str_val.replace("theta","s_q[" + str(ind) + "]")
                    # then output the code
                    cpp_ind = str(self.gen_static_array_ind_3d(ind,col,row))
                    self.gen_add_code_line("s_XImats[" + cpp_ind + "] = static_cast<T>(" + str_val + ");")
    # end the serial section
    self.gen_add_end_control_flow()
    self.gen_add_sync(use_thread_group)
    # then copy the TL to BR in parallel across all X
    self.gen_add_parallel_loop("kcr",str(9*self.robot.get_num_pos()),use_thread_group)
    self.gen_add_code_line("int k = kcr / 9; int cr = kcr % 9; int c = cr / 3; int r = cr % 3;")
    self.gen_add_code_line("int srcInd = k*36 + c*6 + r; int dstInd = srcInd + 21; // 3 more rows and cols")
    self.gen_add_code_line("s_XImats[dstInd] = s_XImats[srcInd];")
    self.gen_add_end_control_flow()
    self.gen_add_sync(use_thread_group)
    # add the function end
    self.gen_add_end_function()

def gen_topology_helpers_size(self):
    n = self.robot.get_num_pos()
    size = 0
    if not self.robot.is_serial_chain():
        size += 5*n + 1
    if not self.robot.are_Ss_identical(list(range(n))):
        size += n
    return size

def gen_topology_sparsity_helpers_python(self, INIT_MODE = False):
    n = self.robot.get_num_pos()
    num_ancestors = [len(self.robot.get_ancestors_by_id(jid)) for jid in range(n)]
    num_subtree = [len(self.robot.get_subtree_by_id(jid)) for jid in range(n)]
    running_sum_num_ancestors = [sum(num_ancestors[0:jid]) for jid in range(n+1)] # for the loops that check < jid+1
    running_sum_num_subtree = [sum(num_subtree[0:jid]) for jid in range(n)]

    dva_cols_per_partial = self.robot.get_total_ancestor_count() + n
    df_cols_per_partial = self.robot.get_total_ancestor_count() + self.robot.get_total_subtree_count()

    dva_cols_per_jid = [num_ancestors[jid] + 1 for jid in range(n)]
    df_cols_per_jid = [num_ancestors[jid] + num_subtree[jid] for jid in range(n)]
    df_col_that_is_jid = num_ancestors
    
    running_sum_dva_cols_per_jid = [running_sum_num_ancestors[jid] + jid for jid in range(n+1)] # for the loops that check < jid+1
    running_sum_df_cols_per_jid = [running_sum_num_ancestors[jid] + running_sum_num_subtree[jid] for jid in range(n)]

    if INIT_MODE:
        return [str(val) for val in num_ancestors], [str(val) for val in num_subtree], \
               [str(val) for val in running_sum_num_ancestors], [str(val) for val in running_sum_num_subtree]
    else:
        return dva_cols_per_partial, dva_cols_per_jid, running_sum_dva_cols_per_jid, \
                df_cols_per_partial,  df_cols_per_jid,  running_sum_df_cols_per_jid,  df_col_that_is_jid

def gen_init_topology_helpers(self):
    n = self.robot.get_num_pos()
    if self.robot.is_serial_chain() and self.robot.are_Ss_identical(list(range(n))):
        self.gen_add_code_lines(["//", \
                                 "// Topology Helpers not needed!", \
                                 "//", \
                                 "template <typename T>", \
                                 "__host__", \
                                 "int *init_topology_helpers(){return nullptr;}"])
        return
    # add function description
    self.gen_add_func_doc("Initializes the topology_helpers in GPU memory", \
        [], [],"A pointer to the topology_helpers memory in the GPU")
    # add the function start boilerplate
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__host__")
    self.gen_add_code_line("int *init_topology_helpers() {", True)
    # add the helpers needed
    code = []
    if not self.robot.is_serial_chain():
        parent_inds = [str(self.robot.get_parent_id(jid)) for jid in range(n)]
        # generate sparsity helpers
        num_ancestors, num_subtree, running_sum_num_ancestors, running_sum_num_subtree = self.gen_topology_sparsity_helpers_python(True)
        _, _, running_sum_dva_cols_per_jid, _, _, running_sum_df_cols_per_jid, _ = self.gen_topology_sparsity_helpers_python()
        code.extend(["int h_topology_helpers[] = {" + ",".join(parent_inds) + ", // parent_inds",
                     "                            " + ",".join(num_ancestors) + ", // num_ancestors",
                     "                            " + ",".join(num_subtree) + ", // num_subtree",
                     "                            " + ",".join(running_sum_num_ancestors) + ", // running_sum_num_ancestors",
                     "                            " + ",".join(running_sum_num_subtree) + "}; // running_sum_num_subtree"])
        if not self.robot.are_Ss_identical(list(range(n))):
            S_inds = [str(self.robot.get_S_by_id(jid).tolist().index(1)) for jid in range(n)]
            code.insert(-4,"                            " + ",".join(S_inds) + ", // S_inds")
    elif not self.robot.are_Ss_identical(list(range(n))):
            S_inds = [str(self.robot.get_S_by_id(jid).tolist().index(1)) for jid in range(n)]
            code.append("int h_topology_helpers[] = {" + ",".join(S_inds) + "}; // S_inds")
    self.gen_add_code_lines(code)
    
    # allocate and transfer data to the GPU and return the pointer to the memory
    self.gen_add_code_line("int *d_topology_helpers; gpuErrchk(cudaMalloc((void**)&d_topology_helpers," + str(self.gen_topology_helpers_size()) + "*sizeof(int)));")
    self.gen_add_code_line("gpuErrchk(cudaMemcpy(d_topology_helpers,h_topology_helpers," + str(self.gen_topology_helpers_size()) + "*sizeof(int),cudaMemcpyHostToDevice));")
    self.gen_add_code_line("return d_topology_helpers;")
    self.gen_add_end_function()

def gen_topology_helpers_pointers_for_cpp(self, inds = None, updated_var_names = None, NO_GRAD_FLAG = False):
    var_names = dict(jid_name = "jid", s_topology_helpers_name = "s_topology_helpers")
    if updated_var_names is not None:
        for key,value in updated_var_names.items():
            var_names[key] = value
    n = self.robot.get_num_pos()
    if inds == None:
        inds = list(range(n))
    IDENTICAL_S_FLAG_INDS = self.robot.are_Ss_identical(inds)
    IDENTICAL_S_FLAG_GLOBAL = self.robot.are_Ss_identical(list(range(n)))

    # check for one ind
    if len(inds) == 1:
        parent_ind = str(self.robot.get_parent_id(inds[0]))
        dva_cols_per_partial, _, running_sum_dva_cols_per_jid, _, _, running_sum_df_cols_per_jid, df_col_that_is_jid = self.gen_topology_sparsity_helpers_python()
        dva_col_offset_for_jid = str(running_sum_dva_cols_per_jid[inds[0]])
        df_col_offset_for_jid = str(running_sum_df_cols_per_jid[inds[0]])
        dva_col_offset_for_parent = str(running_sum_dva_cols_per_jid[self.robot.get_parent_id(inds[0])])
        df_col_offset_for_parent = str(running_sum_df_cols_per_jid[self.robot.get_parent_id(inds[0])])
        dva_col_offset_for_jid_p1 = str(running_sum_dva_cols_per_jid[inds[0] + 1])
        df_col_that_is_jid = str(df_col_that_is_jid[inds[0]])
    
    # else branch based on type of robot
    else:
    
        # special case for serial chain
        if self.robot.is_serial_chain():
            parent_ind = "(" + var_names["jid_name"] + "-1" + ")"
            dva_col_offset_for_jid = var_names["jid_name"] + "*(" + var_names["jid_name"] + "+1)/2"
            df_col_offset_for_jid = str(n) + "*" + var_names["jid_name"]
            dva_col_offset_for_parent = var_names["jid_name"] + "*(" + var_names["jid_name"] + "-1)/2"
            df_col_offset_for_parent = str(n) + "*(" + var_names["jid_name"] + "-1)"
            dva_col_offset_for_jid_p1 = "(" + var_names["jid_name"] + "+1)*(" + var_names["jid_name"] + "+2)/2"
            df_col_that_is_jid = var_names["jid_name"]
            if not IDENTICAL_S_FLAG_INDS:
                S_ind = "s_topology_helpers[jid]"
    
        # generic robot
        else:
            parent_ind = var_names["s_topology_helpers_name"] + "[" + var_names["jid_name"] + "]"
            if not IDENTICAL_S_FLAG_INDS: # this set of inds can be optimized if all S are the same
                S_ind = var_names["s_topology_helpers_name"] + "[" + str(n) + " + " + var_names["jid_name"] +  "]"
            if not IDENTICAL_S_FLAG_GLOBAL: # ofset is based on any S different at all
                ancestor_offset = 2*n
            else:
                ancestor_offset = n
                
            subtree_offset = ancestor_offset + n
            running_sum_ancestor_offset = subtree_offset + n
            running_sum_subtree_offset = running_sum_ancestor_offset + n + 1

            dva_col_offset_for_jid = "(" + var_names["s_topology_helpers_name"] + "[" + str(running_sum_ancestor_offset) + " + " + var_names["jid_name"] + "]" + \
                                     " + " + var_names["jid_name"] + ")"
            df_col_offset_for_jid = "(" + var_names["s_topology_helpers_name"] + "[" + str(running_sum_ancestor_offset) + " + " + var_names["jid_name"] + "]" + \
                                    " + " + var_names["s_topology_helpers_name"] + "[" + str(running_sum_subtree_offset) + " + " + var_names["jid_name"] + "])"

            dva_col_offset_for_parent = "(" + var_names["s_topology_helpers_name"] + "[" + str(running_sum_ancestor_offset) + " + " + parent_ind + "]" + \
                                        " + " + parent_ind + ")"
            df_col_offset_for_parent = "(" + var_names["s_topology_helpers_name"] + "[" + str(running_sum_ancestor_offset) + " + " + parent_ind + "]" + \
                                       " + " + var_names["s_topology_helpers_name"] + "[" + str(running_sum_subtree_offset) + " + " + parent_ind + "])"

            dva_col_offset_for_jid_p1 = "(" + var_names["s_topology_helpers_name"] + "[" + str(running_sum_ancestor_offset) + " + " + var_names["jid_name"] + " + 1]" + \
                                        " + " + var_names["jid_name"] + " + 1)"

            df_col_that_is_jid = var_names["s_topology_helpers_name"] + "[" + str(ancestor_offset) + " + " + var_names["jid_name"] + "]"

    if IDENTICAL_S_FLAG_INDS: # always true for one ind
        S_ind = str(self.robot.get_S_by_id(inds[0]).tolist().index(1))

    if NO_GRAD_FLAG:
        return parent_ind, S_ind
    else:
        return parent_ind, S_ind, dva_col_offset_for_jid, df_col_offset_for_jid, dva_col_offset_for_parent, df_col_offset_for_parent, dva_col_offset_for_jid_p1, df_col_that_is_jid

def gen_insert_helpers_function_call(self, updated_var_names = None):
    n = self.robot.get_num_pos()
    var_names = dict( \
        s_XImats_name = "s_XImats", \
        s_topology_helpers_name = "s_topology_helpers", \
    )
    if updated_var_names is not None:
        for key,value in updated_var_names.items():
            var_names[key] = value
    func_call = var_names["s_XImats_name"] + ", "
    if not self.robot.is_serial_chain() or not self.robot.are_Ss_identical(list(range(n))):
        func_call += var_names["s_topology_helpers_name"] + ", "
    return func_call

def gen_insert_helpers_func_def_params(self, func_def, func_params, param_insert_position = -1, updated_var_names = None):
    n = self.robot.get_num_pos()
    var_names = dict( \
        s_XImats_name = "s_XImats", \
        s_topology_helpers_name = "s_topology_helpers", \
    )
    if updated_var_names is not None:
        for key,value in updated_var_names.items():
            var_names[key] = value
    n = self.robot.get_num_pos()
    func_def += "T *" + var_names["s_XImats_name"] + ", "
    func_params.insert(param_insert_position,"s_XImats is the (shared) memory holding the updated XI matricies for the given s_q")
    if not self.robot.is_serial_chain() or not self.robot.are_Ss_identical(list(range(n))):
        func_def += "int *" + var_names["s_topology_helpers_name"] + ", "
        func_params.insert(param_insert_position,"s_topology_helpers is the (shared) memory destination location for the topology_helpers")
    return func_def, func_params

def gen_init_robotModel(self):
    self.gen_add_func_doc("Initializes the robotModel helpers in GPU memory", \
                           [], [],"A pointer to the robotModel struct")
    # add the function start boilerplate
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__host__")
    self.gen_add_code_line("robotModel<T>* init_robotModel() {", True)
    # then construct the host side struct
    self.gen_add_code_lines(["robotModel<T> h_robotModel;", \
                             "h_robotModel.d_XImats = init_XImats<T>();", \
                             "h_robotModel.d_topology_helpers = init_topology_helpers<T>();"])
    # then allocate memeory and copy to device
    self.gen_add_code_lines(["robotModel<T> *d_robotModel; gpuErrchk(cudaMalloc((void**)&d_robotModel,sizeof(robotModel<T>)));",
                             "gpuErrchk(cudaMemcpy(d_robotModel,&h_robotModel,sizeof(robotModel<T>),cudaMemcpyHostToDevice));"])
    self.gen_add_code_line("return d_robotModel;")
    self.gen_add_end_function()