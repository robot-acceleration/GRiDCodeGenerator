import numpy as np
import copy
#np.set_printoptions(precision=4, suppress=True, linewidth = 100)

def gen_crba_inner_temp_mem_size(self):
    n = self.robot.get_num_pos()
    return 140*n

def gen_crba_inner_function_call(self, use_thread_group = False, updated_var_names = None):
    var_names = dict( \
        s_M_name = "s_M", \
        s_q_name = "s_q", \
        s_qd_name = "s_qd", \
        s_temp_name = "s_temp", \
        #s_XI = "s_XImats", \
        gravity_name = "gravity"
    )
    if updated_var_names is not None:
        for key,value in updated_var_names.items():
            var_names[key] = value
    crba_code_start = "crba_inner<T>(" + var_names["s_M_name"] + ", " +  var_names["s_q_name"] + ", " + var_names["s_qd_name"] + ", " 
    crba_code_end = var_names["s_temp_name"] + ", " + var_names["gravity_name"] + ");"
    if use_thread_group:
        id_code_start = id_code_start.replace("(","(tgrp, ")
    crba_code_middle = self.gen_insert_helpers_function_call()
    crba_code = crba_code_start + crba_code_middle + crba_code_end
    self.gen_add_code_line(crba_code)


def gen_crba_inner(self, use_thread_group = False):
    
    n = self.robot.get_num_pos()
    n_bfs_levels = self.robot.get_max_bfs_level() + 1

    #construct the boilerplate and function definition
    func_params = [ "s_q is the vector of joint positions", \
                    "s_qd is the vector of joint velocities", \
                    "s_M is a pointer to the matrix of inertia" \
                    "s_XI is the pointer to the transformation and inertia matricies ", \
                    "s_temp is a pointer to helper shared memory of size 6*NUM_JOINTS = " + \
                            str(self.gen_crba_inner_temp_mem_size()), \
                    "gravity is the gravity constant"]
    func_notes = [] #insert notes abt function 
    func_def_start = "void crba_inner("
    func_def_middle = "T *s_M, const T *s_q, const T *s_qd, "
    func_def_end = "T *s_temp, const T gravity) {"
    if use_thread_group:
        func_def_start = func_def_start.replace("(", "(cgrps::thread_group tgrp, ")
        func_params.insert(0,"tgrp is the handle to the thread_group running this function")

    func_def_middle, func_params = self.gen_insert_helpers_func_def_params(func_def_middle, func_params, -2)
    func_def = func_def_start + func_def_middle + func_def_end
    self.gen_add_func_doc("Compute the Composite Rigid Body Algorithm", func_notes, func_params, None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)

    
    #deal with like memory for variables --> memory is taken care of in device and host 
    alpha_offset = 0
    beta_offset = alpha_offset + 36*n
    #t_offset = beta_offset + 36*n
    fh_offset = beta_offset + 36*n
    #j_offset = fh_offset + 36*n #bc fh is matrix 
    parent_offset = fh_offset + 6*n #bc j is 1 int 
    jid_offset = parent_offset + n
    self.gen_add_code_line("T *alpha = &s_temp[" + str(alpha_offset) + "];")
    self.gen_add_code_line("T *beta = &s_temp[" + str(beta_offset) + "];")
    #self.gen_add_code_line("T *transpose = &s_temp[" + str(t_offset) + "];")
    self.gen_add_code_line("T *s_fh = &s_temp[" + str(fh_offset) + "];")
    #self.gen_add_code_line("T *s_j = &s_temp[" + str(j_offset) + "];")
    self.gen_add_code_line("T *s_parent_inds = &s_temp[" + str(parent_offset) + "];")
    self.gen_add_code_line("T *s_jid_list = &s_temp[" + str(jid_offset) + "];")
    #self.gen_add_code_line("T *temparr = &s_t_emp[" + str(temparr_offset) + "];")
    #self.gen_add_code_line("T *s_X = &s_temp[" + str(x_offset) + "];")
    #self.gen_add_code_line("T *s_IC = &s_temp[" + str(ic_offset) + "];")
    #self.gen_add_code_line("T *ind = &s_temp[" + str(ind_offset) + "];")
    #self.gen_add_code_line("T *parent_ind_cpp = &s_temp[" + str(parent_offset) + "];")
    #self.gen_add_code_line("T *S_ind_cpp = &s_temp[" + str(sval_offset) + "];")

    #x_offset = 0
    #ic_offset = x_offset + 6*6*6*4

    #self.gen_add_code_line("T *s_X = &s_XImats[" + str(x_offset) + "];")
    #self.gen_add_code_line("T *s_IC = &s_XImats[" + str(ic_offset) + "];")

    """for i in range(7):
        self.gen_add_code_line("s_X[" + str(i) + "] = s_XImats[" + str(36*i) + "];")
    
    for i in range(7):
        self.gen_add_code_line("s_IC[" + str(i) + "] = s_XImats[" + str(36*(i+7)) + "];")"""

    
    self.gen_add_code_line("//")
    self.gen_add_code_line("// first loop (split into 2 parallel loops in bfs loop)")
    self.gen_add_code_line("// each bfs level runs in parallel")
    self.gen_add_code_line("//")

    """self.gen_add_parallel_loop("ind",str(n),use_thread_group)
    self.gen_add_code_line("for(int i=0; i<" + str(n) + "; i++) {")
    self.gen_add_code_line("    for(int j=0; j<" + str(n) + "; j++) {")
    self.gen_add_code_line("        transpose[36*ind + i][36*ind + j] = &s_XImats[36*ind + j][36*ind + i];")
    self.gen_add_code_line("    }")
    self.gen_add_code_line("}")"""

    #self.gen_add_code_line("transpose[ind] = dot_prod<T,6,6,1>(&s_XImats[36*(ind+7)],&s_XImats[36*ind]);")
    """self.gen_add_code_line("alpha[ind] = dot_prod<T,6,6,1>(&s_XImats[36*(ind+7)],&transpose[36*ind]);")
    self.gen_add_end_control_flow()
    self.gen_add_sync(use_thread_group) 

    self.gen_add_parallel_loop("ind",str(n),use_thread_group)
    self.gen_add_code_line("int row = ind % 6;")
    self.gen_add_code_line("beta[ind] = dot_prod<T,6,6,1>(&s_XImats[36*ind + row],&alpha[ind]);")
    self.gen_add_end_control_flow()
    self.gen_add_sync(use_thread_group) 


    self.gen_add_code_line("//")
    self.gen_add_code_line("// each bfs level runs in parallel")
    self.gen_add_code_line("//")"""
 
    for bfs_level in range(n_bfs_levels-1,0,-1):
        inds = self.robot.get_ids_by_bfs_level(bfs_level)
        #ind = str(inds)
        #ind = ind[1]
        #self.gen_add_code_line("ind = " + ind)
        joint_names = [self.robot.get_joint_by_id(indj).get_name() for indj in inds]
        link_names = [self.robot.get_link_by_id(indl).get_name() for indl in inds]

        self.gen_add_code_line("// pass updates where bfs_level is " + str(bfs_level))
        self.gen_add_code_line("//     joints are: " + ", ".join(joint_names))
        self.gen_add_code_line("//     links are: " + ", ".join(link_names))

        parent_ind_cpp, S_ind_cpp = self.gen_topology_helpers_pointers_for_cpp(inds, NO_GRAD_FLAG = True)
        #self.gen_add_code_line("// S_ind_cpp = " + S_ind_cpp)
        
        #self.gen_add_parallel_loop("ind",str(36*len(inds)),use_thread_group)
        self.gen_add_parallel_loop("ind",str(36),use_thread_group)
        #row = ind % 6   
        #self.gen_add_code_line("rowwww = " + str(row))
        #self.gen_add_code_line("int row = ind % 6;")
        if len(inds) > 1:
            select_var_vals = [("int", "jid", [str(jid) for jid in inds])]
            self.gen_add_multi_threaded_select("ind", "< ", [str(6*(i+1)) for i in range(len(inds))], select_var_vals)
            jid = "jid"
            self.gen_add_code_line("s_jid_list[ind] = jid;")
            #self.gen_add_code_line("for(int i=0; i<"+ str(len(inds)) +"; i++){int count = 0; if(jid != s_jid_list[i]){s_jid_list[count] = jid; count += 1;}}")
            self.gen_add_end_control_flow()
            self.gen_add_sync(use_thread_group) 

            for i in range(len(inds)):
                self.gen_add_parallel_loop("ind",str(36),use_thread_group)
                self.gen_add_code_line("int jid = s_jid_list[" + str(len(inds)+ i*6) + "];")
                self.gen_add_code_line("int row = ind % 6; int col = (ind / 6) % 6; int jid6 = jid * 6;")
                #self.gen_add_code_line("alpha[ind] = dot_prod<T,6,6,1>(&s_XImats[36*(ind+7)],&s_XImats[36*ind + row]);")
                self.gen_add_code_line("alpha[6*jid6 + row + (6*col)] = dot_prod<T,6,1,1>(&s_XImats[6*jid6 + row*6],&s_XImats[36*(jid+" + str(n-7) + "+7) + (col*6)]);")
                self.gen_add_end_control_flow()
                self.gen_add_sync(use_thread_group) 


                self.gen_add_parallel_loop("ind",str(36),use_thread_group)
                self.gen_add_code_line("int jid = s_jid_list[" + str(len(inds)+ i*6) + "];")
                self.gen_add_code_line("int parent_ind = " + str(parent_ind_cpp) + ";")
                self.gen_add_code_line("int row = ind % 6; int col = (ind / 6) % 6; int jid6 = jid * 6;")
                #self.gen_add_code_line("alpha[ind] = dot_prod<T,6,6,1>(&s_XImats[36*(ind+7)],&s_XImats[36*ind + row]);")
                self.gen_add_code_line("beta[6*jid6 + col + (6*row)] = dot_prod<T,6,6,1>(&alpha[6*jid6 + row],&s_XImats[6*jid6 + (col*6)]);")
                self.gen_add_code_line("s_XImats[36*(parent_ind +" + str(n-7) + "+7) + col + (6*row)] += beta[6*jid6 + col + (6*row)];")           
                self.gen_add_end_control_flow()
                self.gen_add_sync(use_thread_group) 

        else:
            jid = str(inds[0])
            self.gen_add_code_line("int jid = " + str(jid) + " ;")
        
            self.gen_add_code_line("int row = ind % 6; int col = (ind / 6) % 6; int jid6 = jid * 6;")
            #self.gen_add_code_line("alpha[ind] = dot_prod<T,6,6,1>(&s_XImats[36*(ind+7)],&s_XImats[36*ind + row]);")
            self.gen_add_code_line("alpha[6*jid6 + row + (6*col)] = dot_prod<T,6,1,1>(&s_XImats[6*jid6 + row*6],&s_XImats[36*(jid+7) + (col*6)]);")

            self.gen_add_end_control_flow()
            self.gen_add_sync(use_thread_group) 

            self.gen_add_parallel_loop("ind",str(36),use_thread_group)
            jid = str(inds[0])
            self.gen_add_code_line("int jid = " + str(jid) + " ;")
            self.gen_add_code_line("int parent_ind = " + str(parent_ind_cpp) + ";")

            self.gen_add_code_line("int row = ind % 6; int col = (ind / 6) % 6; int jid6 = jid * 6;")
            #self.gen_add_code_line("alpha[ind] = dot_prod<T,6,6,1>(&s_XImats[36*(ind+7)],&s_XImats[36*ind + row]);")
            self.gen_add_code_line("beta[6*jid6 + col + (6*row)] = dot_prod<T,6,6,1>(&alpha[6*jid6 + row],&s_XImats[6*jid6 + (col*6)]);")
            self.gen_add_code_line("s_XImats[36*(parent_ind +" + str(n-7) + "+7) + col + (6*row)] += beta[6*jid6 + col + (6*row)];")
        
            self.gen_add_end_control_flow()
            self.gen_add_sync(use_thread_group)

        """if len(inds) > 1:
            select_var_vals = [("int", "jid", [str(jid) for jid in inds])]
            self.gen_add_multi_threaded_select("ind", "< ", [str(6*(i+1)) for i in range(len(inds))], select_var_vals)
            jid = "jid" """
        
        
        

        """self.gen_add_code_line("for(int ind = 0; ind <" + str(n_bfs_levels-1) + "; ind++) {")
        self.gen_add_code_line("    int parent_ind = " + str(parent_ind_cpp) + ";")
        self.gen_add_code_line("    s_XImats[36* (" + str(parent_ind_cpp) + " + 7)] = s_XImats[36*(" + str(parent_ind_cpp) + " + 7)] + dot_prod<T,6,6,6>(&alpha[36*ind],&s_XImats[36*ind]);")
        self.gen_add_code_line("}")"""

                 
        #self.gen_add_code_line("int ind = " + ind + ";")
        #self.gen_add_code_line("int row = ind % 6;")

        #parent_ind = self.robot.get_parent_id(ind)
        #comment = "// parent_ind = self.robot.get_parent_id(ind)"
        #self.gen_add_code_line(comment)
        #self.gen_add_code_line("int parent_ind = " + parent_ind_cpp + ";")
        
        #Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind])
        #comment = "// Xmat = self.robot.get_Xmat_Func_by_id(ind)(q[ind]) --> as param so don't need to init it now" 
        #self.gen_add_code_line(comment)
    
        #IC[parent_ind] = IC[parent_ind] + np.matmul(Xmat.T@IC[ind],Xmat)
        #comment = "// IC[parent_ind] = IC[parent_ind] + (Xmat.T)@IC[ind]@Xmat" 
        #self.gen_add_code_line(comment)
        #self.gen_add_code_line("temparr[jid] = dot_prod<T,6,6,1>(&s_IC[jid],&s_X[jid]);")
        #self.gen_add_code_line("temparr[ind] = dot_prod<T,6,6,1>(&s_XImats[36*(ind+7)],&s_XImats[36*ind]);")
        #self.gen_add_code_line("s_XImats[36*(" + parent_ind_cpp + "+7)] = s_XImats[36*(" + parent_ind_cpp + "+7)] + dot_prod<T,6,6,1>(&s_XImats[36*ind + row], &temparr[ind]);") 
        #self.gen_add_code_line("&s_IC[" + parent_ind_cpp + "] = &s_IC[" + parent_ind_cpp + "] + dot_prod<T,6,6,1>(s_X[6*jid6 + row], dot_prod<T,6,6,1>(&s_IC[jid],s_X));") 
        #self.gen_add_end_control_flow()

    #self.gen_add_sync(use_thread_group) 
    
    self.gen_add_code_line("//")
    self.gen_add_code_line("// Calculation of fh  ")
    self.gen_add_code_line("//")

    for ind in range(n-1, -1, -1): # in parallel
        # Calculation of fh
        _, S_ind_cpp = self.gen_topology_helpers_pointers_for_cpp(NO_GRAD_FLAG = True)
        
        #self.gen_add_parallel_loop("ind",str(n),use_thread_group)
        self.gen_add_parallel_loop("ind",str(6),use_thread_group)

        self.gen_add_code_line("int jid = " + str(ind) + " ; int jid6 = jid * 6;")
        self.gen_add_code_line("int row = ind % 6; ")

        #fh_code = "if (row == " + S_ind_cpp + "){s_fh[ind] = s_XImats[36*(jid+7) + row + col*6];}"
        fh_code = "s_fh[jid6 + row] = s_XImats[36*(jid+7 + " + str(n-7) + ") + " +  S_ind_cpp + "*6 + ind];"
        self.gen_add_code_line(fh_code)

        self.gen_add_end_control_flow()
    self.gen_add_sync(use_thread_group)
    
    self.gen_add_code_line("//")
    self.gen_add_code_line("// Calculation of M[ind, ind] ")
    self.gen_add_code_line("//")
    
    self.gen_add_parallel_loop("jid",str(n),use_thread_group)
    self.gen_add_code_line("int jidn = jid * " + str(n) + "; int jid6 = jid * 6;")

    h_code = "s_M[jid + jidn] = s_fh[jid6 + " + S_ind_cpp + "];"
    self.gen_add_code_line(h_code)

    #parent_code = "s_parent[jid] =  "+ parent_ind_cpp + "];"
    #self.gen_add_code_line(parent_code)
    self.gen_add_end_control_flow()
    self.gen_add_sync(use_thread_group)

    self.gen_add_code_line("//")
    self.gen_add_code_line("// Calculation of parent_ind, fh, M[ind, parent] and M[parent, ind] ")
    self.gen_add_code_line("//")

    for bfs_level in range(n_bfs_levels-1,0,-1):
        inds = self.robot.get_ids_by_bfs_level(bfs_level)
        parent_ind_cpp, S_ind_cpp = self.gen_topology_helpers_pointers_for_cpp(inds, NO_GRAD_FLAG = True)

        
        joint_names = [self.robot.get_joint_by_id(indj).get_name() for indj in inds]
        link_names = [self.robot.get_link_by_id(indl).get_name() for indl in inds]

        self.gen_add_code_line("// pass updates where bfs_level is " + str(bfs_level))
        self.gen_add_code_line("//     joints are: " + ", ".join(joint_names))
        self.gen_add_code_line("//     links are: " + ", ".join(link_names))

        """if len(inds) > 1:
            select_var_vals = [("int", "jid", [str(jid) for jid in inds])]
            self.gen_add_multi_threaded_select("ind", "< ", [str(6*(i+1)) for i in range(len(inds))], select_var_vals)
            jid = "jid"
        else:
            jid = str(inds[0])"""

        if len(inds) > 1:
           #     select_var_vals = [("int", "jid", [str(jid) for jid in inds])]
           #     self.gen_add_multi_threaded_select("parallel_ind", "< ", [str(6*(i+1)) for i in range(len(inds))], select_var_vals)
           #     jid = "jid"
           # else:
            self.gen_add_parallel_loop("parallel_ind",str(len(inds)*6),use_thread_group)
            select_var_vals = [("int", "jid", [str(jid) for jid in inds])]
            self.gen_add_multi_threaded_select("parallel_ind", "< ", [str(6*(i+1)) for i in range(len(inds))], select_var_vals)
            self.gen_add_code_line("s_parent_inds[jid] = " + str(parent_ind_cpp) + ";")
            #self.gen_add_code_line("s_inds[jid] = " + str(jid) + ";")
            self.gen_add_end_control_flow()
            self.gen_add_sync(use_thread_group)
          
           # self.gen_add_parallel_loop("parallel_ind",str(len(inds)*6),use_thread_group)


        if len(inds) > 1:
        #     select_var_vals = [("int", "jid", [str(jid) for jid in inds])]
        #     self.gen_add_multi_threaded_select("parallel_ind", "< ", [str(6*(i+1)) for i in range(len(inds))], select_var_vals)
        #     jid = "jid"
        # else:
            self.gen_add_parallel_loop("parallel_ind",str(len(inds)*6),use_thread_group)
            select_var_vals = [("int", "jid", [str(jid) for jid in inds])]
            self.gen_add_multi_threaded_select("parallel_ind", "< ", [str(6*(i+1)) for i in range(len(inds))], select_var_vals)         
            self.gen_add_code_line("int jid6 = jid * 6; int row = parallel_ind % 6;")
            self.gen_add_code_line("int curr_parent = s_parent_inds[jid];")
            self.gen_add_code_line("s_fh[jid6 + row] = dot_prod<T,6,1,1>(&s_XImats[36*(curr_parent+1) + 6*row], &s_fh[jid6]);")
            self.gen_add_end_control_flow()
            self.gen_add_sync(use_thread_group)



        for parent_level in range(bfs_level, 0, -1):
            par_inds = self.robot.get_ids_by_bfs_level(parent_level)
        
            parent_ind_cpp_par, S_ind_cpp_par = self.gen_topology_helpers_pointers_for_cpp(par_inds, NO_GRAD_FLAG = True)
            
            self.gen_add_code_line("// pass updates where parent_level is " + str(parent_ind_cpp_par))
            self.gen_add_code_line("//     joints are: " + ", ".join(joint_names))
            self.gen_add_code_line("//     links are: " + ", ".join(link_names))

            if len(inds) <= 1:
                self.gen_add_parallel_loop("parallel_ind",str(len(inds)*6),use_thread_group)
            #     select_var_vals = [("int", "jid", [str(jid) for jid in inds])]
            #     self.gen_add_multi_threaded_select("parallel_ind", "< ", [str(6*(i+1)) for i in range(len(inds))], select_var_vals)
            #     jid = "jid"
            # else:
                jid = str(inds[0])
                self.gen_add_code_line("int jid = " + str(jid) + ";")
                self.gen_add_code_line("s_parent_inds[jid] = " + str(parent_ind_cpp_par) + ";")
                #self.gen_add_code_line("s_inds[jid] = " + str(jid) + ";")
                self.gen_add_end_control_flow()
                self.gen_add_sync(use_thread_group)
            
            # self.gen_add_parallel_loop("parallel_ind",str(len(inds)*6),use_thread_group)

            if len(inds) <= 1:
                self.gen_add_parallel_loop("parallel_ind",str(len(inds)*6),use_thread_group)
                # select_var_vals = [("int", "jid", [str(jid) for jid in inds])]
                # self.gen_add_multi_threaded_select("parallel_ind", "< ", [str(6*(i+1)) for i in range(len(inds))], select_var_vals)
                # jid = "jid"
            # else:
                jid = str(inds[0])
                self.gen_add_code_line("int jid = " + str(jid) + ";")
            
                self.gen_add_code_line("int jid6 = jid * 6; int row = parallel_ind % 6;")
                self.gen_add_code_line("int curr_parent = s_parent_inds[jid];")
                self.gen_add_code_line("s_fh[jid6 + row] = dot_prod<T,6,1,1>(&s_XImats[36*(curr_parent+1) + 6*row], &s_fh[jid6]);")
                self.gen_add_end_control_flow()
                self.gen_add_sync(use_thread_group)

            self.gen_add_parallel_loop("parallel_ind",str(len(inds)*6),use_thread_group)
            if len(inds) > 1:
                select_var_vals = [("int", "jid", [str(jid) for jid in inds])]
                self.gen_add_multi_threaded_select("parallel_ind", "< ", [str(6*(i+1)) for i in range(len(inds))], select_var_vals)
                jid = "jid"
                self.gen_add_code_line("int jidn = jid * " + str(n) + "; int jid6 = jid * 6;" )
                self.gen_add_code_line("if((jid-1) != -1){")
                self.gen_add_code_line("    s_M[jid-1 + jidn] = s_fh[jid6 + " + S_ind_cpp_par + "];") 
                self.gen_add_code_line("    s_M[jid + (jid-1)*" + str(n) + "] = s_M[jid-1 + jidn];")
            else:
                jid = str(inds[0])
                self.gen_add_code_line("int jid = " + str(jid) + ";")        
                self.gen_add_code_line("int jidn = jid * " + str(n) + "; int jid6 = jid * 6;" )
                self.gen_add_code_line("int curr_parent = s_parent_inds[jid];")
                self.gen_add_code_line("if(s_parent_inds[jid] != -1){")
                self.gen_add_code_line("    s_M[jid + curr_parent*" + str(n) + "] = s_fh[jid6 + " + S_ind_cpp_par + "];")
                self.gen_add_code_line("    s_M[curr_parent + jidn] = s_M[jid + curr_parent*" + str(n) + "];")

            self.gen_add_code_line("}")
            self.gen_add_end_control_flow()
            self.gen_add_sync(use_thread_group)       
  
    self.gen_add_end_function()

    # for bfs_level in range(n_bfs_levels-1,0,-1):
    #     inds = self.robot.get_ids_by_bfs_level(bfs_level)
    #     parent_ind_cpp, S_ind_cpp = self.gen_topology_helpers_pointers_for_cpp(inds, NO_GRAD_FLAG = True)

    #     #self.gen_add_code_line("s_inds = ", inds)
    #     #self.gen_add_code_line("s_parent_inds = ", inds)
        
    #     joint_names = [self.robot.get_joint_by_id(indj).get_name() for indj in inds]
    #     link_names = [self.robot.get_link_by_id(indl).get_name() for indl in inds]

    #     self.gen_add_code_line("// pass updates where bfs_level is " + str(bfs_level))
    #     self.gen_add_code_line("//     joints are: " + ", ".join(joint_names))
    #     self.gen_add_code_line("//     links are: " + ", ".join(link_names))

    #     """if len(inds) > 1:
    #         select_var_vals = [("int", "jid", [str(jid) for jid in inds])]
    #         self.gen_add_multi_threaded_select("ind", "< ", [str(6*(i+1)) for i in range(len(inds))], select_var_vals)
    #         jid = "jid"
    #     else:
    #         jid = str(inds[0])"""

    #     self.gen_add_parallel_loop("parallel_ind",str(len(inds)*6),use_thread_group)

    #     if len(inds) > 1:
    #         select_var_vals = [("int", "jid", [str(jid) for jid in inds])]
    #         self.gen_add_multi_threaded_select("parallel_ind", "< ", [str(6*(i+1)) for i in range(len(inds))], select_var_vals)
    #         jid = "jid"
    #     else:
    #         jid = str(inds[0])
    #         self.gen_add_code_line("int jid = " + str(jid) + ";")

    #     self.gen_add_code_line("s_parent_inds[jid] = " + str(parent_ind_cpp) + ";")
    #     #self.gen_add_code_line("s_inds[jid] = " + str(jid) + ";")
    #     self.gen_add_end_control_flow()
    #     self.gen_add_sync(use_thread_group)
        
    #     self.gen_add_parallel_loop("parallel_ind",str(len(inds)*6),use_thread_group)

    #     if len(inds) > 1:
    #         select_var_vals = [("int", "jid", [str(jid) for jid in inds])]
    #         self.gen_add_multi_threaded_select("parallel_ind", "< ", [str(6*(i+1)) for i in range(len(inds))], select_var_vals)
    #         jid = "jid"
    #     else:
    #         jid = str(inds[0])
    #         self.gen_add_code_line("int jid = " + str(jid) + ";")
        
    #     self.gen_add_code_line("int jid6 = jid * 6; int row = parallel_ind % 6;")
    #     #self.gen_add_code_line("int row = parallel_ind % 6; int col = (parallel_ind / 7) % 7;")
    #     #self.gen_add_code_line("int curr_joint = s_inds[jid];")
    #     self.gen_add_code_line("int curr_parent = s_parent_inds[jid];")
        
    #     #self.gen_add_code_line("if(s_parent_inds[jid + row*7] != -1){s_fh[curr_joint] = dot_prod<T,6,1,1>(&s_XImats[6*jid6 + curr_joint*6], &s_fh[jid + row*7]);}")
    #     #self.gen_add_code_line("if(s_parent_inds[parallel_ind] != -1){s_fh[row*7 + curr_joint] = dot_prod<T,6,1,7>(&s_XImats[6*jid6 + 6*row], &s_fh[row*7 + curr_joint]);}")
    #     self.gen_add_code_line("s_fh[jid6 + row] = dot_prod<T,6,1,1>(&s_XImats[36*(curr_parent+1) + 6*row], &s_fh[jid6]);")

    #     self.gen_add_end_control_flow()
    #     self.gen_add_sync(use_thread_group)



    #     for parent_level in range(bfs_level, 0, -1):
    #         par_inds = self.robot.get_ids_by_bfs_level(parent_level)
        
    #         parent_ind_cpp_par, S_ind_cpp_par = self.gen_topology_helpers_pointers_for_cpp(par_inds, NO_GRAD_FLAG = True)
            
    #         self.gen_add_code_line("// pass updates where parent_level is " + str(parent_ind_cpp_par))
    #         self.gen_add_code_line("//     joints are: " + ", ".join(joint_names))
    #         self.gen_add_code_line("//     links are: " + ", ".join(link_names))


           
    #         # self.gen_add_parallel_loop("parallel_ind",str(len(inds)*6),use_thread_group)

    #         # if len(inds) > 1:
    #         #     select_var_vals = [("int", "jid", [str(jid) for jid in inds])]
    #         #     self.gen_add_multi_threaded_select("parallel_ind", "< ", [str(6*(i+1)) for i in range(len(inds))], select_var_vals)
    #         #     jid = "jid"
    #         # else:
    #         #     jid = str(inds[0])
    #         #     self.gen_add_code_line("int jid = " + str(jid) + ";")

    #         # self.gen_add_code_line("s_parent_inds[jid] = " + str(parent_ind_cpp_par) + ";")
    #         # #self.gen_add_code_line("s_inds[jid] = " + str(jid) + ";")
    #         # self.gen_add_end_control_flow()
    #         # self.gen_add_sync(use_thread_group)
            
    #         # self.gen_add_parallel_loop("parallel_ind",str(len(inds)*6),use_thread_group)

    #         # if len(inds) > 1:
    #         #     select_var_vals = [("int", "jid", [str(jid) for jid in inds])]
    #         #     self.gen_add_multi_threaded_select("parallel_ind", "< ", [str(6*(i+1)) for i in range(len(inds))], select_var_vals)
    #         #     jid = "jid"
    #         # else:
    #         #     jid = str(inds[0])
    #         #     self.gen_add_code_line("int jid = " + str(jid) + ";")
            
    #         # self.gen_add_code_line("int jid6 = jid * 6; int row = parallel_ind % 6;")
    #         # #self.gen_add_code_line("int row = parallel_ind % 6; int col = (parallel_ind / 7) % 7;")
    #         # #self.gen_add_code_line("int curr_joint = s_inds[jid];")
    #         # self.gen_add_code_line("int curr_parent = s_parent_inds[jid];")
            
    #         # #self.gen_add_code_line("if(s_parent_inds[jid + row*7] != -1){s_fh[curr_joint] = dot_prod<T,6,1,1>(&s_XImats[6*jid6 + curr_joint*6], &s_fh[jid + row*7]);}")
    #         # #self.gen_add_code_line("if(s_parent_inds[parallel_ind] != -1){s_fh[row*7 + curr_joint] = dot_prod<T,6,1,7>(&s_XImats[6*jid6 + 6*row], &s_fh[row*7 + curr_joint]);}")
    #         # self.gen_add_code_line("s_fh[jid6 + row] = dot_prod<T,6,1,1>(&s_XImats[36*(curr_parent+1) + 6*row], &s_fh[jid6]);")

    #         # self.gen_add_end_control_flow()
    #         # self.gen_add_sync(use_thread_group)

    #         self.gen_add_parallel_loop("parallel_ind",str(len(inds)*6),use_thread_group)
    #         if len(inds) > 1:
    #             select_var_vals = [("int", "jid", [str(jid) for jid in inds])]
    #             self.gen_add_multi_threaded_select("parallel_ind", "< ", [str(6*(i+1)) for i in range(len(inds))], select_var_vals)
    #             jid = "jid"
    #             self.gen_add_code_line("int jidn = jid * " + str(n) + "; int jid6 = jid * 6;" )
    #             #self.gen_add_code_line("int row = parallel_ind % 7; int col = (parallel_ind / 7) % 7;")
    #             #self.gen_add_code_line("int curr_joint = s_inds[parallel_ind];")
    #             #self.gen_add_code_line("int curr_parent = s_parent_inds[parallel_ind];")

    #             self.gen_add_code_line("if((jid-1) != -1){")
    #             self.gen_add_code_line("    s_M[jid-1 + jidn] = s_fh[jid6 + " + S_ind_cpp_par + "];")  
    #             self.gen_add_code_line("    s_M[jid + (jid-1)*" + str(n) + "] = s_M[jid-1 + jidn];")
    #         else:
    #             jid = str(inds[0])
    #             self.gen_add_code_line("int jid = " + str(jid) + ";")
            
    #             self.gen_add_code_line("int jidn = jid * " + str(n) + "; int jid6 = jid * 6;" )
    #             #self.gen_add_code_line("int row = parallel_ind % 7; int col = (parallel_ind / 7) % 7;")
    #             #self.gen_add_code_line("int curr_joint = s_inds[parallel_ind];")
    #             self.gen_add_code_line("int curr_parent = s_parent_inds[parallel_ind];")

    #             self.gen_add_code_line("if(s_parent_inds[jid] != -1){")
    #             self.gen_add_code_line("    s_M[jid + curr_parent*" + str(n) + "] = s_fh[jid6 + " + S_ind_cpp_par + "];") 
    #             self.gen_add_code_line("    s_M[curr_parent + jidn] = s_M[jid + curr_parent*" + str(n) + "];")

    #             #self.gen_add_code_line("    if(s_j[jid] == " + str(S_ind_cpp) +"){s_M[jid + j_jid*7] = s_fh[jid];} else {s_M[jid + j_jid*7] = 0;}")
    #             #self.gen_add_code_line("    s_M[j_jid + jid*7] = s_M[jid + j_jid*7];")
    #         self.gen_add_code_line("}")

    #         self.gen_add_end_control_flow()
    #         self.gen_add_sync(use_thread_group)       
  
    # self.gen_add_end_function()

def gen_crba_device_temp_mem_size(self):
    n = self.robot.get_num_pos()
    wrapper_size = self.gen_topology_helpers_size() + 72*n # for XImats
    return self.gen_crba_inner_temp_mem_size() + wrapper_size

def gen_crba_device(self, use_thread_group = False):
    n = self.robot.get_num_pos()

    # construct the boilerplate and function definition
    func_params = ["s_M is a pointer to the matrix of inertia", \
                   "s_q is the vector of joint positions", \
                   "s_qd is the vector of joint velocities", \
                   "d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)", \
                   "gravity is the gravity constant"]
    func_notes = []
    func_def_start = "void crba_device("
    func_def_middle = "T *s_M, const T *s_q, const T *s_qd,"
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
    shared_mem_size = self.gen_crba_device_temp_mem_size() if not self.use_dynamic_shared_mem_flag else None 
    self.gen_XImats_helpers_temp_shared_memory_code(shared_mem_size)

    # then load/update XI and run the algo
    self.gen_load_update_XImats_helpers_function_call(use_thread_group)
    self.gen_crba_inner_function_call(use_thread_group)
    self.gen_add_end_function()

def gen_crba_kernel(self, use_thread_group = False, single_call_timing = False):
    n = self.robot.get_num_pos()
    # define function def and params
    func_params = ["d_M is the pointer to the matrix of inertia", \
                    "d_q_qd is the vector of joint positions and velocities", \
                    "stride_q_qd is the stride between each q, qd", \
                    "d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)", \
                    "gravity is the gravity constant", \
                    "num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)"]
    func_notes = []
    func_def_start = "void crba_kernel(T *d_M, const T *d_q_qd, const int stride_q_qd, "
    func_def_end = "const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {"
    func_def = func_def_start + func_def_end
    if single_call_timing:
        func_def = func_def.replace("kernel(", "kernel_single_timing(")
    
    # then generate the code
    self.gen_add_func_doc("Compute the CRBA (Composite Rigid Body Algorithm)", \
                            func_notes, func_params, None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__global__")
    self.gen_add_code_line(func_def, True)

    # add shared memory variables
    shared_mem_vars = ["__shared__ T s_M[" + str(n*n) + "];", \
                        "__shared__ T s_q_qd[3*" + str(n) + "]; T *s_q = s_q_qd; T *s_qd = &s_q_qd[" + str(n) + "];", ]
    self.gen_add_code_lines(shared_mem_vars)
    shared_mem_size = self.gen_crba_inner_temp_mem_size() if not self.use_dynamic_shared_mem_flag else None
    self.gen_XImats_helpers_temp_shared_memory_code(shared_mem_size)
    if use_thread_group:
        self.gen_add_code_line("cgrps::thread_group tgrp = TBD;")
    if not single_call_timing:
        # load to shared mem and loop over blocks to compute all requested comps
        self.gen_add_parallel_loop("k","NUM_TIMESTEPS",use_thread_group,block_level = True)
        self.gen_kernel_load_inputs("q_qd","stride_q_qd",str(3*n),use_thread_group)
        # compute
        self.gen_add_code_line("// compute")
        self.gen_load_update_XImats_helpers_function_call(use_thread_group)
        self.gen_crba_inner_function_call(use_thread_group)
        self.gen_add_sync(use_thread_group)
        # save to global
        self.gen_kernel_save_result("M","1",str(n*n),use_thread_group)
        self.gen_add_end_control_flow()
    else:
        # repurpose NUM_TIMESTEPS for number of timing reps
        self.gen_kernel_load_inputs_single_timing("q_qd",str(3*n),use_thread_group)
        # then compute in loop for timing
        self.gen_add_code_line("// compute with NUM_TIMESTEPS as NUM_REPS for timing")
        self.gen_add_code_line("for (int rep = 0; rep < NUM_TIMESTEPS; rep++){", True)
        self.gen_load_update_XImats_helpers_function_call(use_thread_group)
        self.gen_crba_inner_function_call(use_thread_group)
        self.gen_add_end_control_flow()
        # save to global
        self.gen_kernel_save_result_single_timing("M",str(n*n),use_thread_group)
    self.gen_add_end_function()

def gen_crba_host(self, mode = 0):


    #old version that works for iiwa but not for hyq
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
    func_call_start = "crba_kernel<T><<<block_dimms,thread_dimms,CRBA_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_M,hd_data->d_q_qd,stride_q_qd,"
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
                                    ("num_timesteps*" if not single_call_timing else "") + "sizeof(T),cudaMemcpyHostToDevice,streams[0]));}"])
    else:
        self.gen_add_code_line("int stride_q_qd = USE_COMPRESSED_MEM ? 2*NUM_JOINTS: 3*NUM_JOINTS;")
    self.gen_add_code_line("// then call the kernel")
    func_call = func_call_start + func_call_end
    # add in compressed mem adjusts
    func_call_mem_adjust = "if (USE_COMPRESSED_MEM) {" + func_call + "}"
    func_call_mem_adjust2 = "else                    {" + func_call.replace("hd_data->d_q_qd","hd_data->d_q_qd_u") + "}"
    # compule into a set of code
    func_call_code = [func_call_mem_adjust, func_call_mem_adjust2, "gpuErrchk(cudaDeviceSynchronize());"]
    # wrap function call in timing (if needed)
    if single_call_timing:
        func_call_code.insert(0,"struct timespec start, end; clock_gettime(CLOCK_MONOTONIC,&start);")
        func_call_code.append("clock_gettime(CLOCK_MONOTONIC,&end);")
    self.gen_add_code_lines(func_call_code)
    if not compute_only:
        # then transfer memory back
        self.gen_add_code_lines(["// finally transfer the result back", \
                                 "gpuErrchk(cudaMemcpy(hd_data->h_M,hd_data->d_M,NUM_JOINTS*NUM_JOINTS*" + \
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