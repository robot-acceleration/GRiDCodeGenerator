def gen_mx_func_call_for_cpp(self, inds = None, PEQ_FLAG = False, SCALE_FLAG = False, updated_var_names = None):
    var_names = dict(S_ind_name = "S_ind", s_dst_name = "s_dst", s_src_name = "s_src", s_scale_name = "s_scale")
    if updated_var_names is not None:
        for key,value in updated_var_names.items():
            var_names[key] = value
            n = self.robot.get_num_pos()
    if inds == None:
        inds = list(range(n))
    IDENTICAL_S_FLAG_INDS = self.robot.are_Ss_identical(inds)

    # check for all the same mxFunc
    if IDENTICAL_S_FLAG_INDS:
        S_ind = str(self.robot.get_S_by_id(inds[0]).tolist().index(1))
    else:
        S_ind = "X"
    # find which function type
    if not SCALE_FLAG and not PEQ_FLAG:
        func_name = "mx" + S_ind + "<T>"
    elif not SCALE_FLAG:
        func_name = "mx" + S_ind + "_peq<T>"
    elif not PEQ_FLAG:
        func_name = "mx" + S_ind + "_scaled<T>"
    else:
        func_name = "mx" + S_ind + "_peq_scaled<T>"
    # then make the code line
    code_start = func_name + "(" + var_names["s_dst_name"] + ", " + var_names["s_src_name"]
    code_middle = ""
    if SCALE_FLAG:
        code_middle += ", " + var_names["s_scale_name"]
    if not IDENTICAL_S_FLAG_INDS:
        code_middle += ", " + var_names["S_ind_name"]
    code_end = ");"
    self.gen_add_code_line(code_start + code_middle + code_end)

def gen_spatial_algebra_helpers(self):
    # First function -- add dot product code with and without const
    for i in range(4):
        self.gen_add_func_doc("Compute the dot product between two vectors", ["Assumes computed by a single thread"], \
                              ["vec1 is the first vector of length N with stride S1", "vec2 is the second vector of length N with stride S2"], \
                              "the resulting final value")
        self.gen_add_code_line("template <typename T, int N, int S1, int S2>")
        self.gen_add_code_line("__device__")
        if i == 0:
            self.gen_add_code_line("T dot_prod(const T *vec1, const T *vec2) {", True)
        elif i == 1:
            self.gen_add_code_line("T dot_prod(T *vec1, const T *vec2) {", True)
        elif i == 2:
            self.gen_add_code_line("T dot_prod(const T *vec1, T *vec2) {", True)
        else:
            self.gen_add_code_line("T dot_prod(T *vec1, T *vec2) {", True)
        self.gen_add_code_line("T result = 0;")
        self.gen_add_code_line("for(int i = 0; i < N; i++) {", True)
        self.gen_add_code_line("result += vec1[i*S1] * vec2[i*S2];")
        self.gen_add_end_control_flow()
        self.gen_add_code_line("return result;")
        self.gen_add_end_function()

    # Then the motion vector matrix cross product operations
    # We need to compute each column seperately as they are often multipled by S to pull out a column
    # We need versions with and without an alpha multiplier to take advantage of fused multiple add opps
    # and with and without a PEQ -- finally we need a generic one with a switch for dynamic column selection
    MxCode = []
    # Mx0
    MxCode.append(["s_vecX[0] = static_cast<T>(0);", \
                  "s_vecX[1] = s_vec[2];", \
                  "s_vecX[2] = -s_vec[1];", \
                  "s_vecX[3] = static_cast<T>(0);", \
                  "s_vecX[4] = s_vec[5];", \
                  "s_vecX[5] = -s_vec[4];"])
    # Mx1
    MxCode.append(["s_vecX[0] = -s_vec[2];", \
                  "s_vecX[1] = static_cast<T>(0);", \
                  "s_vecX[2] = s_vec[0];", \
                  "s_vecX[3] = -s_vec[5];", \
                  "s_vecX[4] = static_cast<T>(0);", \
                  "s_vecX[5] = s_vec[3];"])
    # Mx2
    MxCode.append(["s_vecX[0] = s_vec[1];", \
                  "s_vecX[1] = -s_vec[0];", \
                  "s_vecX[2] = static_cast<T>(0);", \
                  "s_vecX[3] = s_vec[4];", \
                  "s_vecX[4] = -s_vec[3];", \
                  "s_vecX[5] = static_cast<T>(0);"])
    # Mx3
    MxCode.append(["s_vecX[0] = static_cast<T>(0);", \
                  "s_vecX[1] = static_cast<T>(0);", \
                  "s_vecX[2] = static_cast<T>(0);", \
                  "s_vecX[3] = static_cast<T>(0);", \
                  "s_vecX[4] = s_vec[2];", \
                  "s_vecX[5] = -s_vec[1];"])
    # Mx3
    MxCode.append(["s_vecX[0] = static_cast<T>(0);", \
                  "s_vecX[1] = static_cast<T>(0);", \
                  "s_vecX[2] = static_cast<T>(0);", \
                  "s_vecX[3] = -s_vec[2];", \
                  "s_vecX[4] = static_cast<T>(0);", \
                  "s_vecX[5] = s_vec[0];"])
    # Mx3
    MxCode.append(["s_vecX[0] = static_cast<T>(0);", \
                  "s_vecX[1] = static_cast<T>(0);",  \
                  "s_vecX[2] = static_cast<T>(0);",  \
                  "s_vecX[3] = s_vec[1];", \
                  "s_vecX[4] = -s_vec[0];", \
                  "s_vecX[5] = static_cast<T>(0);"])
    for ind in range(6):
        # without alpha
        self.gen_add_func_doc("Generates the motion vector cross product matrix column " + str(ind),\
                             ["Assumes only one thread is running each function call"],\
                             ["s_vecX is the destination vector","s_vec is the source vector"],None)
        self.gen_add_code_line("template <typename T>")
        self.gen_add_code_line("__device__")
        self.gen_add_code_line("void mx" + str(ind) + "(T *s_vecX, const T *s_vec) {", True)
        self.gen_add_code_lines(MxCode[ind])
        self.gen_add_end_function()
        # without alpha PEQ
        self.gen_add_func_doc("Adds the motion vector cross product matrix column " + str(ind),\
                             ["Assumes only one thread is running each function call"],\
                             ["s_vecX is the destination vector","s_vec is the source vector"],None)
        self.gen_add_code_line("template <typename T>")
        self.gen_add_code_line("__device__")
        self.gen_add_code_line("void mx" + str(ind) + "_peq(T *s_vecX, const T *s_vec) {", True)
        MxCodePEQ = [code_line.replace("=","+=") for code_line in MxCode[ind]]
        MxCodePEQ = [code_line for code_line in MxCodePEQ if "static_cast<T>(0)" not in code_line]
        self.gen_add_code_lines(MxCodePEQ)
        self.gen_add_end_function()
        # with alpha
        self.gen_add_func_doc("Generates the motion vector cross product matrix column " + str(ind),\
                             ["Assumes only one thread is running each function call"],\
                             ["s_vecX is the destination vector","s_vec is the source vector","alpha is the scaling factor"],None)
        self.gen_add_code_line("template <typename T>")
        self.gen_add_code_line("__device__")
        self.gen_add_code_line("void mx" + str(ind) + "_scaled(T *s_vecX, const T *s_vec, const T alpha) {", True)
        MxCodeWithAlpha = [code_line.replace("];","]*alpha;") for code_line in MxCode[ind]]
        self.gen_add_code_lines(MxCodeWithAlpha)
        self.gen_add_end_function()
        # with alpha PEQ
        self.gen_add_func_doc("Adds the motion vector cross product matrix column " + str(ind),\
                             ["Assumes only one thread is running each function call"],\
                             ["s_vecX is the destination vector","s_vec is the source vector","alpha is the scaling factor"],None)
        self.gen_add_code_line("template <typename T>")
        self.gen_add_code_line("__device__")
        self.gen_add_code_line("void mx" + str(ind) + "_peq_scaled(T *s_vecX, const T *s_vec, const T alpha) {", True)
        MxCodePEQ = [code_line.replace("=","+=") for code_line in MxCode[ind]]
        MxCodePEQWithAlpha = [code_line.replace("];","]*alpha;") for code_line in MxCodePEQ]
        MxCodePEQWithAlpha = [code_line for code_line in MxCodePEQWithAlpha if "static_cast<T>(0)" not in code_line]
        self.gen_add_code_lines(MxCodePEQWithAlpha)
        self.gen_add_end_function()
    # then the generics with a switch statement
    # without alpha
    func_params = ["s_vecX is the destination vector","s_vec is the source vector"]
    func_call_params = ["T *s_vecX", "const T *s_vec", "const int S_ind"]
    subFunc_call_params = ["s_vecX", "s_vec"]
    for i in range(4):
        func_name_suffix = ("_peq" if i % 2 else "") + ("_scaled" if i > 1 else "")
        if i == 2: # add the alpha in once we start doing the scaled ones
            func_params.append("alpha is the scaling factor")
            func_call_params.insert(-1,"const T alpha")
            subFunc_call_params.append("alpha")
        self.gen_add_func_doc("Generates the motion vector cross product matrix for a runtime selected column",\
                             ["Assumes only one thread is running each function call"],func_params,None)
        self.gen_add_code_line("template <typename T>")
        self.gen_add_code_line("__device__")
        self.gen_add_code_line("void mxX" + func_name_suffix + "(" + ", ".join(func_call_params) + ") {", True)
        self.gen_add_code_line("switch(S_ind){", True)
        for ind in range(6):
            self.gen_add_code_line("case " + str(ind) + ": mx" + str(ind) + func_name_suffix + "<T>(" + ", ".join(subFunc_call_params) + "); break;")
        self.gen_add_end_control_flow()
        self.gen_add_end_function()

    # The force cross product matrix transpose columns are simply 
    # the negative of the motion vector matrix columns (as fx = -mx^T)
    # so we can skip their codegen and just use -mx later
    # We also need to compute the full fx matrix and full fx*vector
    # Note: fx vec is:
    #   0  -v(2)  v(1)    0  -v(5)  v(4)
    # v(2)    0  -v(0)  v(5)    0  -v(3)
    #-v(1)  v(0)    0  -v(4)  v(3)    0
    #   0     0     0     0  -v(2)  v(1)
    #   0     0     0   v(2)    0  -v(0)
    #   0     0     0  -v(1)  v(0)    0
    self.gen_add_func_doc("Generates the motion vector cross product matrix",\
                         ["Assumes only one thread is running each function call"],\
                         ["s_matX is the destination matrix","s_vecX is the source vector"],None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void fx(T *s_matX, const T *s_vecX) {", True)
    FxCode = ["s_matX[6*0 + 0] = static_cast<T>(0);",
              "s_matX[6*0 + 1] = s_vecX[2];", \
              "s_matX[6*0 + 2] = -s_vecX[1];", \
              "s_matX[6*0 + 3] = static_cast<T>(0);", \
              "s_matX[6*0 + 4] = static_cast<T>(0);", \
              "s_matX[6*0 + 5] = static_cast<T>(0);", \
              "s_matX[6*1 + 0] = -s_vecX[2];", \
              "s_matX[6*1 + 1] = static_cast<T>(0);", \
              "s_matX[6*1 + 2] = s_vecX[0];", \
              "s_matX[6*1 + 3] = static_cast<T>(0);", \
              "s_matX[6*1 + 4] = static_cast<T>(0);", \
              "s_matX[6*1 + 5] = static_cast<T>(0);", \
              "s_matX[6*2 + 0] = s_vecX[1];", \
              "s_matX[6*2 + 1] = -s_vecX[0];", \
              "s_matX[6*2 + 2] = static_cast<T>(0);", \
              "s_matX[6*2 + 3] = static_cast<T>(0);", \
              "s_matX[6*2 + 4] = static_cast<T>(0);", \
              "s_matX[6*2 + 5] = static_cast<T>(0);", \
              "s_matX[6*3 + 0] = static_cast<T>(0);", \
              "s_matX[6*3 + 1] = s_vecX[5];", \
              "s_matX[6*3 + 2] = -s_vecX[4];", \
              "s_matX[6*3 + 3] = static_cast<T>(0);", \
              "s_matX[6*3 + 4] = s_vecX[2];", \
              "s_matX[6*3 + 5] = -s_vecX[1];", \
              "s_matX[6*4 + 0] = -s_vecX[5];", \
              "s_matX[6*4 + 1] = static_cast<T>(0);", \
              "s_matX[6*4 + 2] = s_vecX[3];", \
              "s_matX[6*4 + 3] = -s_vecX[2];", \
              "s_matX[6*4 + 4] = static_cast<T>(0);", \
              "s_matX[6*4 + 5] = s_vecX[0];", \
              "s_matX[6*5 + 0] = s_vecX[4];", \
              "s_matX[6*5 + 1] = -s_vecX[3];", \
              "s_matX[6*5 + 2] = static_cast<T>(0);", \
              "s_matX[6*5 + 3] = s_vecX[1];", \
              "s_matX[6*5 + 4] = -s_vecX[0];", \
              "s_matX[6*5 + 5] = static_cast<T>(0);"]
    self.gen_add_code_lines(FxCode)
    self.gen_add_end_function()
    self.gen_add_func_doc("Generates the motion vector cross product matrix for a pre-zeroed destination",\
                         ["Assumes only one thread is running each function call", "Assumes destination is zeroed"],\
                         ["s_matX is the destination matrix","s_vecX is the source vector"],None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void fx_zeroed(T *s_matX, const T *s_vecX) {", True)
    FxCodeZeroed = [code_line for code_line in FxCode if "static_cast<T>(0)" not in code_line]
    self.gen_add_code_lines(FxCodeZeroed)
    self.gen_add_end_function()
    self.gen_add_func_doc("Generates the motion vector cross product matrix and multiples by the input vector",\
                         ["Assumes only one thread is running each function call"],\
                         ["s_result is the result vector","s_fxVec is the fx vector","s_timesVec is the multipled vector"],None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void fx_times_v(T *s_result, const T *s_fxVec, const T *s_timesVec) {", True)
    FxCodeTimesVec = []
    FxCodeTimesVec.append("s_result[0] = -s_fxVec[2] * s_timesVec[1] + s_fxVec[1] * s_timesVec[2] - s_fxVec[5] * s_timesVec[4] + s_fxVec[4] * s_timesVec[5];")
    FxCodeTimesVec.append("s_result[1] =  s_fxVec[2] * s_timesVec[0] - s_fxVec[0] * s_timesVec[2] + s_fxVec[5] * s_timesVec[3] - s_fxVec[3] * s_timesVec[5];")
    FxCodeTimesVec.append("s_result[2] = -s_fxVec[1] * s_timesVec[0] + s_fxVec[0] * s_timesVec[1] - s_fxVec[4] * s_timesVec[3] + s_fxVec[3] * s_timesVec[4];")
    FxCodeTimesVec.append("s_result[3] =                                                          - s_fxVec[2] * s_timesVec[4] + s_fxVec[1] * s_timesVec[5];")
    FxCodeTimesVec.append("s_result[4] =                                                            s_fxVec[2] * s_timesVec[3] - s_fxVec[0] * s_timesVec[5];")
    FxCodeTimesVec.append("s_result[5] =                                                          - s_fxVec[1] * s_timesVec[3] + s_fxVec[0] * s_timesVec[4];")
    self.gen_add_code_lines(FxCodeTimesVec)
    self.gen_add_end_function()
    self.gen_add_func_doc("Adds the motion vector cross product matrix multiplied by the input vector",\
                         ["Assumes only one thread is running each function call"],\
                         ["s_result is the result vector","s_fxVec is the fx vector","s_timesVec is the multipled vector"],None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line("void fx_times_v_peq(T *s_result, const T *s_fxVec, const T *s_timesVec) {", True)
    FxCodeTimesVecPEQ = [code_line.replace("=","+=") for code_line in FxCodeTimesVec]
    self.gen_add_code_lines(FxCodeTimesVecPEQ)
    self.gen_add_end_function()