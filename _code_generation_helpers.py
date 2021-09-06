def gen_add_code_line(self, new_code_line, add_indent_after = False):
    self.code_str += self.indent_level * "    " + new_code_line + "\n"
    if add_indent_after:
        self.indent_level += 1

def gen_add_code_lines(self, new_code_lines, add_indent_after = False):
    for new_code_line in new_code_lines:
        self.gen_add_code_line(new_code_line)
    if add_indent_after:
        self.indent_level += 1

def gen_add_end_control_flow(self):
    self.indent_level -= 1
    self.gen_add_code_line("}")

def gen_add_end_function(self):
    self.indent_level -= 1
    self.gen_add_code_line("}\n")

def gen_add_func_doc(self, func_desc, notes = [], params = [], return_val = None):
    self.gen_add_code_line("/**")
    self.gen_add_code_line(" * " + func_desc)
    self.gen_add_code_line(" *")
    if len(notes) > 0:
        self.gen_add_code_line(" * Notes:")
        for note in notes:
            self.gen_add_code_line(" *   " + note)
        self.gen_add_code_line(" *")
    for param in params:
        self.gen_add_code_line(" * @param " + param)
    if return_val is not None:
        self.gen_add_code_line(" * @return " + return_val)
    self.gen_add_code_line(" */")

def gen_add_serial_ops(self, use_thread_group = False):
    if use_thread_group:
        self.gen_add_code_line("if(tgrp.thread_rank() == 0){", True)
    else:
        self.gen_add_code_line("if(threadIdx.x == 0 && threadIdx.y == 0){", True)

def gen_add_parallel_loop(self, var_name, max_val, use_thread_group = False, block_level = False):
    if block_level:
        if use_thread_group:
            print("![ERROR]: BLOCK LEVEL THREAD GROUP LOOP NOT IMPLEMENTED YET")
        else:
            code = "for(int " + var_name + " = blockIdx.x + blockIdx.y*gridDim.x; " + \
                        var_name + " < " + max_val + "; " + var_name + " += gridDim.x*gridDim.y){"
    else:
        if use_thread_group:
            code = "for(int " + var_name + " = tgrp.thread_rank(); " + \
                        var_name + " < " + max_val + "; " + var_name + " += tgrp.size()){"
        else:
            code = "for(int " + var_name + " = threadIdx.x + threadIdx.y*blockDim.x; " + \
                        var_name + " < " + max_val + "; " + var_name + " += blockDim.x*blockDim.y){"
    self.gen_add_code_line(code, True)

def gen_static_array_ind_2d(self, col, row, col_stride = 6):
    return col_stride*col + row

def gen_static_array_ind_3d(self, ind, col, row, ind_stride = 36, col_stride = 6):
    return ind_stride*ind + col_stride*col + row

def gen_add_sync(self, use_thread_group = False):
    if use_thread_group:
        self.gen_add_code_line("tgrp.sync();")
    else:
        self.gen_add_code_line("__syncthreads();")

def gen_var_in_list(self, var_name, option_list):
    if len(option_list) == 1:
        return "(" + var_name + " == " + option_list[0] + ")"
    else:
        return "(" + " || ".join(["(" + var_name + " == " + option + ")" for option in option_list]) + ")"

def gen_var_not_in_list(self, var_name, option_list):
    if len(option_list) == 1:
        return "(" + var_name + " != " + option_list[0] + ")"
    else:
        return "(" + " && ".join(["(" + var_name + " != " + option + ")" for option in option_list]) + ")"

def gen_add_multi_threaded_select(self, loop_counter, comparator, counts, select_tuples, USE_NON_BRANCH_ALWAYS = False):
    # first find the resulting type and variable name
    dst_code = []
    for (dst_type, dst_var, select_list) in select_tuples:
        if dst_type is None:
            dst_code.append(dst_var)
        elif "|" in dst_type:
            dst_type_parts = dst_type.split("|")
            dst_code.append(dst_type_parts[0] + dst_var + ")" + dst_type_parts[1])
        else:
            dst_code.append(dst_type + " " + dst_var)
    # then if many things to select gen it and branch
    if len(select_tuples) > 1 and not USE_NON_BRANCH_ALWAYS:
        self.gen_add_code_line("// branch to get pointer locations")
        # init pointers outside fo select
        self.gen_add_code_line("; ".join(dst_code)  + ";")
        # if / else if / else to select pointers
        n = len(counts)
        code_end = "}"
        for ind in range(n):
            if ind == 0:
                code_start = "     if (" + loop_counter + " " + comparator + " " + counts[ind] + "){ "
            elif ind < n-1:
                code_start = "else if (" + loop_counter + " " + comparator + " " + counts[ind] + "){ "
            else:
                code_start = "else              { "
            code_middle = ""
            for (dst_type, dst_var, select_list) in select_tuples:
                code_middle += dst_var + " = " + select_list[ind] + "; "
            self.gen_add_code_line(code_start + code_middle + code_end)
    # else use a non-branching selector
    else:
        self.gen_add_code_line("// non-branching pointer selector")
        # get the inverse comparator
        n = len(counts)
        inverse_comparator = comparator.replace("<",">") if "<" in comparator else comparator.replace(">","<")
        inverse_comparator = inverse_comparator + "=" if len(inverse_comparator) == 1 else (inverse_comparator[0] if inverse_comparator != "==" else inverse_comparator)
        for tuple_i in range(len(select_tuples)):
            branch_code = ""
            for ind in range(n):
                if ind == 0 or comparator == "==":
                    if comparator == "==" and ind > 0:
                        branch_code += " + "
                    branch_code += "(" + loop_counter + " " + comparator + " " + counts[ind] + ")" 
                elif ind < n-1:
                    branch_code += " + (" + loop_counter + " " + comparator + " " + counts[ind] + " && " + loop_counter + " " + inverse_comparator + " " + counts[ind-1] + ")"
                else:
                    branch_code += " + (" + loop_counter + " " + inverse_comparator + " " + counts[ind-1] + ")"
                branch_code += " * " + select_tuples[tuple_i][2][ind]
            self.gen_add_code_line(dst_code[tuple_i] + " = " + branch_code + ";")

def gen_kernel_load_inputs(self, name, stride, amount, use_thread_group = False, \
                                 name2 = None, stride2 = 1, amount2 = 1, name3 = None, stride3 = 1, amount3 = 1):
    self.gen_add_code_line("// load to shared mem")
    self.gen_add_code_line("const T *d_" + name + "_k = &d_" + name + "[k*" + stride + "];")
    self.gen_add_parallel_loop("ind",amount,use_thread_group)
    self.gen_add_code_line("s_" + name + "[ind] = d_" + name + "_k[ind];")
    self.gen_add_end_control_flow()
    if name2 is not None:
        self.gen_add_code_line("const T *d_" + name2 + "_k = &d_" + name2 + "[k*" + stride2 + "];")
        self.gen_add_parallel_loop("ind",amount2,use_thread_group)
        self.gen_add_code_line("s_" + name2 + "[ind] = d_" + name2 + "_k[ind];")
        self.gen_add_end_control_flow()
    if name3 is not None:
        self.gen_add_code_line("const T *d_" + name3 + "_k = &d_" + name3 + "[k*" + stride3 + "];")
        self.gen_add_parallel_loop("ind",amount3,use_thread_group)
        self.gen_add_code_line("s_" + name3 + "[ind] = d_" + name3 + "_k[ind];")
        self.gen_add_end_control_flow()
    self.gen_add_sync(use_thread_group)

def gen_kernel_save_result(self, store_to_name, stride, amount, use_thread_group = False, load_from_name = None):
    if load_from_name is None:
        load_from_name = "s_" + store_to_name
    self.gen_add_code_line("// save down to global")
    self.gen_add_code_line("T *d_" + store_to_name + "_k = &d_" + store_to_name + "[k*" + stride + "];")
    self.gen_add_parallel_loop("ind",amount,use_thread_group)
    self.gen_add_code_line("d_" + store_to_name + "_k[ind] = " + load_from_name + "[ind];")
    self.gen_add_end_control_flow()
    self.gen_add_sync(use_thread_group)

def gen_kernel_load_inputs_single_timing(self, name, amount, use_thread_group = False, \
                                               name2 = None, amount2 = 1, name3 = None, amount3 = 1):
    self.gen_add_code_line("// load to shared mem")
    self.gen_add_parallel_loop("ind",amount,use_thread_group)
    self.gen_add_code_line("s_" + name + "[ind] = d_" + name + "[ind];")
    self.gen_add_end_control_flow()
    if name2 is not None:
        self.gen_add_parallel_loop("ind",amount2,use_thread_group)
        self.gen_add_code_line("s_" + name2 + "[ind] = d_" + name2 + "[ind];")
        self.gen_add_end_control_flow()
    if name3 is not None:
        self.gen_add_parallel_loop("ind",amount3,use_thread_group)
        self.gen_add_code_line("s_" + name3 + "[ind] = d_" + name3 + "[ind];")
        self.gen_add_end_control_flow()
    self.gen_add_sync(use_thread_group)

def gen_kernel_save_result_single_timing(self, store_to_name, amount, use_thread_group = False, load_from_name = None):
    if load_from_name is None:
        load_from_name = "s_" + store_to_name
    self.gen_add_code_line("// save down to global")
    self.gen_add_parallel_loop("ind",amount,use_thread_group)
    self.gen_add_code_line("d_" + store_to_name + "[ind] = " + load_from_name + "[ind];")
    self.gen_add_end_control_flow()
    self.gen_add_sync(use_thread_group)