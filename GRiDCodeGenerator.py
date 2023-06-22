import numpy as np

class GRiDCodeGenerator:
    # first import helpers to write code generation, spatial algebra, and opology helpers (parent, child, Sind, XImats) and the robotModel object wrapepr
    from .helpers import gen_add_code_line, gen_add_code_lines, gen_add_end_control_flow, gen_add_end_function, \
                         gen_add_func_doc, gen_add_serial_ops, gen_add_parallel_loop, gen_add_sync, gen_var_in_list, \
                         gen_var_not_in_list, gen_add_multi_threaded_select, gen_kernel_load_inputs, gen_kernel_save_result, \
                         gen_kernel_load_inputs_single_timing, gen_kernel_save_result_single_timing, \
                         gen_static_array_ind_2d, gen_static_array_ind_3d, \
                         gen_mx_func_call_for_cpp, gen_spatial_algebra_helpers, \
                         gen_init_XImats, gen_load_update_XImats_helpers_temp_mem_size, gen_load_update_XImats_helpers_function_call, \
                         gen_XImats_helpers_temp_shared_memory_code, gen_load_update_XImats_helpers, gen_topology_helpers_size, \
                         gen_topology_sparsity_helpers_python, gen_init_topology_helpers, gen_topology_helpers_pointers_for_cpp, \
                         gen_insert_helpers_function_call, gen_insert_helpers_func_def_params, gen_init_robotModel

    # then import all of the algorithms
    from .algorithms import gen_inverse_dynamics_inner_temp_mem_size, gen_inverse_dynamics_inner_function_call, \
                            gen_inverse_dynamics_device_temp_mem_size, gen_inverse_dynamics_inner, gen_inverse_dynamics_device, \
                            gen_inverse_dynamics_kernel, gen_inverse_dynamics_host, gen_inverse_dynamics, \
                            gen_direct_minv_inner_temp_mem_size, gen_direct_minv_inner_function_call, gen_direct_minv_inner, \
                            gen_direct_minv_device, gen_direct_minv_kernel, gen_direct_minv_host, gen_direct_minv, \
                            gen_forward_dynamics_inner_temp_mem_size, gen_forward_dynamics_finish_function_call, gen_forward_dynamics_finish, \
                            gen_forward_dynamics_inner_function_call, gen_forward_dynamics_inner, gen_forward_dynamics_device, \
                            gen_forward_dynamics_kernel, gen_forward_dynamics_host, gen_forward_dynamics, \
                            gen_inverse_dynamics_gradient_inner_temp_mem_size, gen_inverse_dynamics_gradient_kernel_max_temp_mem_size, \
                            gen_inverse_dynamics_gradient_inner_function_call, gen_inverse_dynamics_gradient_inner, gen_inverse_dynamics_gradient_device, \
                            gen_inverse_dynamics_gradient_kernel, gen_inverse_dynamics_gradient_host, gen_inverse_dynamics_gradient, \
                            gen_forward_dynamics_gradient_inner_temp_mem_size, gen_forward_dynamics_gradient_kernel_max_temp_mem_size, \
                            gen_forward_dynamics_gradient_inner_python, gen_forward_dynamics_gradient_device, gen_forward_dynamics_gradient_kernel, \
                            gen_forward_dynamics_gradient_host, gen_forward_dynamics_gradient, gen_crba_inner, gen_aba, gen_aba_inner, gen_aba_host, \
                            gen_aba_inner_function_call, gen_aba_kernel, gen_aba_device, gen_aba_inner_temp_mem_size

    # finally import the test code
    from ._test import test_rnea_fpass, test_rnea_bpass, test_rnea, test_minv_bpass, test_minv_fpass, test_densify_Minv, test_minv, test_rnea_grad_inner, \
                      test_rnea_grad, test_fd_grad, mx0, mx1, mx2, mx3, mx4, mx5, mx, mxS, mxv, fx, fxS, fxv

    # initialize the object
    def __init__(self, robotObj, DEBUG_MODE = False, NEED_PRINT_MAT = False, USE_DYNAMIC_SHARED_MEM = True, FILE_NAMESPACE = "grid"):
        self.robot = robotObj
        self.code_str = ""
        self.indent_level = 0
        self.DEBUG_MODE = DEBUG_MODE
        self.gen_print_mat = DEBUG_MODE or NEED_PRINT_MAT
        # even if dynamic shared mem is not requested for large robots we need to use it
        self.use_dynamic_shared_mem_flag = USE_DYNAMIC_SHARED_MEM or (self.robot.get_num_pos() > 12)
        # check for the file/namespace name
        self.file_namespace = FILE_NAMESPACE
    
    # add generic code needs and helpers (includes, memory initialization, constants, kernel settings etc.)
    def gen_add_includes(self, use_thread_group = False):
        # first all of the includes
        self.gen_add_code_line("")
        self.gen_add_code_line("#include <assert.h>")
        self.gen_add_code_line("#include <stdio.h>")
        self.gen_add_code_line("#include <stdlib.h>")
        self.gen_add_code_line("#include <time.h>")
        self.gen_add_code_line("#include <cuda_runtime.h>")
        if use_thread_group:
            self.gen_add_code_line("#include <cooperative_groups.h>")
            self.gen_add_code_line("#include <cooperative_groups/memcpy_async.h>")
        # then any namespaces
        if use_thread_group:
            self.gen_add_code_line("namespace cgrps = cooperative_groups;")
        # then any #defines
        self.gen_add_code_lines(["// single kernel timing helper code", \
            "#define time_delta_us_timespec(start,end) (1e6*static_cast<double>(end.tv_sec - start.tv_sec)+1e-3*static_cast<double>(end.tv_nsec - start.tv_nsec))"])
        self.gen_add_code_line("")

    def gen_add_constants_helpers(self):
        # first add constants
        n = self.robot.get_num_pos()
        XI_size = 72*n
        dva_cols_per_partial = self.robot.get_total_ancestor_count() + n
        max_threads_in_comp_loop = 6*2*dva_cols_per_partial
        suggested_threads = 32 * int(np.ceil(max_threads_in_comp_loop/32.0))
        self.gen_add_code_lines(["const int NUM_JOINTS = " + str(self.robot.get_num_pos()) + ";", \
                                 "const int ID_DYNAMIC_SHARED_MEM_COUNT = " + str(self.gen_inverse_dynamics_inner_temp_mem_size() + XI_size) + ";", \
                                 "const int MINV_DYNAMIC_SHARED_MEM_COUNT = " + str(self.gen_direct_minv_inner_temp_mem_size() + XI_size) + ";", \
                                 "const int FD_DYNAMIC_SHARED_MEM_COUNT = " + str(self.gen_forward_dynamics_inner_temp_mem_size() + XI_size) + ";", \
                                 "const int ID_DU_DYNAMIC_SHARED_MEM_COUNT = " + str(self.gen_inverse_dynamics_gradient_inner_temp_mem_size() + XI_size) + ";", \
                                 "const int FD_DU_DYNAMIC_SHARED_MEM_COUNT = " + str(self.gen_forward_dynamics_gradient_inner_temp_mem_size() + XI_size) + ";", \
                                 "const int ID_DU_MAX_SHARED_MEM_COUNT = " + str(int(self.gen_inverse_dynamics_gradient_kernel_max_temp_mem_size()) + XI_size) + ";", \
                                 "const int FD_DU_MAX_SHARED_MEM_COUNT = " + str(int(self.gen_forward_dynamics_gradient_kernel_max_temp_mem_size()) + XI_size) + ";", \
                                 "const int SUGGESTED_THREADS = " + str(min(suggested_threads, 512)) + ";"]) # max of 512 to avoid exceeding available registers
        # then the structs
        # first add the struct
        self.gen_add_code_line("// Define custom structs")
        self.gen_add_code_lines(["template <typename T>", \
                                 "struct robotModel {", \
                                 "    T *d_XImats;", \
                                 "    int *d_topology_helpers;", \
                                 "};"])
        self.gen_add_code_lines(["template <typename T>", \
                                 "struct gridData {", \
                                 "    // GPU INPUTS", \
                                 "    T *d_q_qd_u;", \
                                 "    T *d_q_qd;", \
                                 "    T *d_q;", \
                                 "    // CPU INPUTS", \
                                 "    T *h_q_qd_u;", \
                                 "    T *h_q_qd;", \
                                 "    T *h_q;", \
                                 "    // GPU OUTPUTS", \
                                 "    T *d_c;", \
                                 "    T *d_Minv;", \
                                 "    T *d_qdd;", \
                                 "    T *d_dc_du;", \
                                 "    T *d_df_du;", \
                                 "    // CPU OUTPUTS", \
                                 "    T *h_c;", \
                                 "    T *h_Minv;", \
                                 "    T *h_qdd;", \
                                 "    T *h_dc_du;", \
                                 "    T *h_df_du;", \
                                 "};"])

    def gen_init_gridData(self):
        code_lines = ["gridData<T> *hd_data = (gridData<T> *)malloc(sizeof(gridData<T>));"
                      "// first the input variables on the GPU", \
                      "gpuErrchk(cudaMalloc((void**)&hd_data->d_q_qd_u, 3*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));", \
                      "gpuErrchk(cudaMalloc((void**)&hd_data->d_q_qd, 2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));", \
                      "gpuErrchk(cudaMalloc((void**)&hd_data->d_q, NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));", \
                      "// and the CPU", \
                      "hd_data->h_q_qd_u = (T *)malloc(3*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));", \
                      "hd_data->h_q_qd = (T *)malloc(2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));", \
                      "hd_data->h_q = (T *)malloc(NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));", \
                      "// then the GPU outputs", \
                      "gpuErrchk(cudaMalloc((void**)&hd_data->d_c, NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));", \
                      "gpuErrchk(cudaMalloc((void**)&hd_data->d_Minv, NUM_JOINTS*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));", \
                      "gpuErrchk(cudaMalloc((void**)&hd_data->d_qdd, NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));", \
                      "gpuErrchk(cudaMalloc((void**)&hd_data->d_dc_du, NUM_JOINTS*2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));", \
                      "gpuErrchk(cudaMalloc((void**)&hd_data->d_df_du, NUM_JOINTS*2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T)));", \
                      "// and the CPU", \
                      "hd_data->h_c = (T *)malloc(NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));", \
                      "hd_data->h_Minv = (T *)malloc(NUM_JOINTS*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));", \
                      "hd_data->h_qdd = (T *)malloc(NUM_JOINTS*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));", \
                      "hd_data->h_dc_du = (T *)malloc(NUM_JOINTS*2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));", \
                      "hd_data->h_df_du = (T *)malloc(NUM_JOINTS*2*NUM_JOINTS*NUM_TIMESTEPS*sizeof(T));", \
                      "return hd_data;"]
        # generate as templated or not function
        self.gen_add_func_doc("Allocated device and host memory for all computations",
                              [], [], "A pointer to the gridData struct of pointers")
        self.gen_add_code_line("template <typename T, int NUM_TIMESTEPS>")
        self.gen_add_code_line("__host__")
        self.gen_add_code_line("gridData<T> *init_gridData(){", True)
        self.gen_add_code_lines(code_lines)
        self.gen_add_end_function()
        self.gen_add_func_doc("Allocated device and host memory for all computations",
                              [], ["Max number of timesteps in the trajectory"], "A pointer to the gridData struct of pointers")
        self.gen_add_code_line("template <typename T>")
        self.gen_add_code_line("__host__")
        self.gen_add_code_line("gridData<T> *init_gridData(int NUM_TIMESTEPS){", True)
        self.gen_add_code_lines(code_lines)
        self.gen_add_end_function()

    def gen_init_close_grid(self):
        # set the max shared mem to account for large robots and allocate streams
        MAX_STREAMS = 3 # max needed in any of our functions
        self.gen_add_func_doc("Sets shared mem needed for gradient kernels and initializes streams for host functions", \
                              [], [], "A pointer to the array of streams")
        self.gen_add_code_line("template <typename T>")
        self.gen_add_code_line("__host__")
        self.gen_add_code_line("cudaStream_t *init_grid(){", True)
        self.gen_add_code_lines(["// set the max temp memory for the gradient kernels to account for large robots", \
                                 "auto id_kern1 = static_cast<void (*)(T *, const T *, const int, const T *, const robotModel<T> *, const T, const int)>(&inverse_dynamics_gradient_kernel<T>);", \
                                 "auto id_kern2 = static_cast<void (*)(T *, const T *, const int, const robotModel<T> *, const T, const int)>(&inverse_dynamics_gradient_kernel<T>);", \
                                 "auto id_kern_timing1 = static_cast<void (*)(T *, const T *, const int, const T *, const robotModel<T> *, const T, const int)>(&inverse_dynamics_gradient_kernel_single_timing<T>);", \
                                 "auto id_kern_timing2 = static_cast<void (*)(T *, const T *, const int, const robotModel<T> *, const T, const int)>(&inverse_dynamics_gradient_kernel_single_timing<T>);", \
                                 "auto fd_kern1 = static_cast<void (*)(T *, const T *, const int, const T *, const T *, const robotModel<T> *, const T, const int)>(&forward_dynamics_gradient_kernel<T>);", \
                                 "auto fd_kern2 = static_cast<void (*)(T *, const T *, const int, const robotModel<T> *, const T, const int)>(&forward_dynamics_gradient_kernel<T>);", \
                                 "auto fd_kern_timing1 = static_cast<void (*)(T *, const T *, const int, const T *, const T *, const robotModel<T> *, const T, const int)>(&forward_dynamics_gradient_kernel_single_timing<T>);", \
                                 "auto fd_kern_timing2 = static_cast<void (*)(T *, const T *, const int, const robotModel<T> *, const T, const int)>(&forward_dynamics_gradient_kernel_single_timing<T>);", \
                                 "cudaFuncSetAttribute(id_kern1,cudaFuncAttributeMaxDynamicSharedMemorySize, ID_DU_MAX_SHARED_MEM_COUNT*sizeof(T));", \
                                 "cudaFuncSetAttribute(id_kern2,cudaFuncAttributeMaxDynamicSharedMemorySize, ID_DU_MAX_SHARED_MEM_COUNT*sizeof(T));", \
                                 "cudaFuncSetAttribute(id_kern_timing1,cudaFuncAttributeMaxDynamicSharedMemorySize, ID_DU_MAX_SHARED_MEM_COUNT*sizeof(T));", \
                                 "cudaFuncSetAttribute(id_kern_timing2,cudaFuncAttributeMaxDynamicSharedMemorySize, ID_DU_MAX_SHARED_MEM_COUNT*sizeof(T));", \
                                 "cudaFuncSetAttribute(fd_kern1,cudaFuncAttributeMaxDynamicSharedMemorySize, FD_DU_MAX_SHARED_MEM_COUNT*sizeof(T));", \
                                 "cudaFuncSetAttribute(fd_kern2,cudaFuncAttributeMaxDynamicSharedMemorySize, FD_DU_MAX_SHARED_MEM_COUNT*sizeof(T));", \
                                 "cudaFuncSetAttribute(fd_kern_timing1,cudaFuncAttributeMaxDynamicSharedMemorySize, FD_DU_MAX_SHARED_MEM_COUNT*sizeof(T));", \
                                 "cudaFuncSetAttribute(fd_kern_timing2,cudaFuncAttributeMaxDynamicSharedMemorySize, FD_DU_MAX_SHARED_MEM_COUNT*sizeof(T));", \
                                 "gpuErrchk(cudaDeviceSynchronize());", \
                                 "// allocate streams", \
                                 "cudaStream_t *streams = (cudaStream_t *)malloc(" + str(MAX_STREAMS) + "*sizeof(cudaStream_t));", \
                                 "int priority, minPriority, maxPriority;", \
                                 "gpuErrchk(cudaDeviceGetStreamPriorityRange(&minPriority, &maxPriority));", \
                                 "for(int i=0; i<" + str(MAX_STREAMS) + "; i++){", \
                                 "    int adjusted_max = maxPriority - i; priority = adjusted_max > minPriority ? adjusted_max : minPriority;", \
                                 "    gpuErrchk(cudaStreamCreateWithPriority(&(streams[i]),cudaStreamNonBlocking,priority));", \
                                 "}", "return streams;"])
        self.gen_add_end_function()
        # free the streams and all allocated data
        self.gen_add_func_doc("Frees the memory used by grid", [], ["streams allocated by init_grid", "robotModel allocated by init_robotModel", "data allocated by init_gridData"], None)
        self.gen_add_code_line("template <typename T>")
        self.gen_add_code_line("__host__")
        self.gen_add_code_line("void close_grid(cudaStream_t *streams, robotModel<T> *d_robotModel, gridData<T> *hd_data){", True)
        self.gen_add_code_lines(["gpuErrchk(cudaFree(d_robotModel));", \
                                 "gpuErrchk(cudaFree(hd_data->d_q_qd_u)); gpuErrchk(cudaFree(hd_data->d_q_qd)); gpuErrchk(cudaFree(hd_data->d_q));", \
                                 "gpuErrchk(cudaFree(hd_data->d_c)); gpuErrchk(cudaFree(hd_data->d_Minv)); gpuErrchk(cudaFree(hd_data->d_qdd));", \
                                 "gpuErrchk(cudaFree(hd_data->d_dc_du)); gpuErrchk(cudaFree(hd_data->d_df_du));", \
                                 "free(hd_data->h_q_qd_u); free(hd_data->h_q_qd); free(hd_data->h_q);", \
                                 "free(hd_data->h_c); free(hd_data->h_Minv); free(hd_data->h_qdd);", \
                                 "free(hd_data->h_dc_du); free(hd_data->h_df_du);", \
                                 "for(int i=0; i<" + str(MAX_STREAMS) + "; i++){gpuErrchk(cudaStreamDestroy(streams[i]));} free(streams);"])
        self.gen_add_end_function()
        
    def gen_add_gpu_err(self):
        # add the GPU error check code
        self.gen_add_func_doc("Check for runtime errors using the CUDA API", \
                ["Adapted from https://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api"], \
                [],None)
        self.gen_add_code_line("__host__")
        self.gen_add_code_line("void gpuAssert(cudaError_t code, const char *file, const int line, bool abort=true){", True)
        self.gen_add_code_line("if (code != cudaSuccess){", True)
        # note that below we need to escape the \n and "" to get it to print to a string or file correctly
        self.gen_add_code_line("fprintf(stderr,\"GPUassert: %s %s %d\\n\", cudaGetErrorString(code), file, line);")
        self.gen_add_code_line("if (abort){cudaDeviceReset(); exit(code);}")
        self.gen_add_end_control_flow()
        self.gen_add_end_control_flow() # end of function but don't want spacing
        self.gen_add_code_line("#define gpuErrchk(err) {gpuAssert(err, __FILE__, __LINE__);}")
        self.gen_add_code_line("")

        # also add printMat for debug if requested
        if self.gen_print_mat:
            self.gen_add_code_line("template <typename T, int M, int N>")
            self.gen_add_code_line("__host__ __device__")
            self.gen_add_code_line("void printMat(T *A, int lda){", True)
            self.gen_add_code_line("for(int i=0; i<M; i++){", True)
            self.gen_add_code_line("for(int j=0; j<N; j++){printf(\"%.4f \",A[i + lda*j]);}")
            self.gen_add_code_line("printf(\"\\n\");")
            self.gen_add_end_control_flow()
            self.gen_add_end_function()
            self.gen_add_code_line("template <typename T, int M, int N>")
            self.gen_add_code_line("__host__ __device__")
            self.gen_add_code_line("void printMat(const T *A, int lda){", True)
            self.gen_add_code_line("for(int i=0; i<M; i++){", True)
            self.gen_add_code_line("for(int j=0; j<N; j++){printf(\"%.4f \",A[i + lda*j]);}")
            self.gen_add_code_line("printf(\"\\n\");")
            self.gen_add_end_control_flow()
            self.gen_add_end_function()

    # finally generate all of the code
    def gen_all_code(self, use_thread_group = False, include_base_inertia = False):
        # first generate the file info
        file_notes = [ "Interface is:", \
            "    __host__   robotModel<T> *d_robotModel = init_robotModel<T>()", \
            "    __host__   cudaStream_t streams = init_grid<T>()", \
            "    __host__   gridData<T> *hd_ata = init_gridData<T,NUM_TIMESTEPS>();"
            "    __host__   close_grid<T>(cudaStream_t *streams, robotModel<T> *d_robotModel, gridData<T> *hd_data)", \
            "",\
            "    __device__ inverse_dynamics_device<T>(T *s_c, const T *s_q, const T *s_qd, const robotModel<T> *d_robotModel, const T gravity)", \
            "    __device__ inverse_dynamics_device<T>(T *s_c, const T *s_q, const T *s_qd, const T *s_qdd, const robotModel<T> *d_robotModel, const T gravity)", \
            "    __global__ inverse_dynamics_kernel<T>(T *d_c, const T *d_q_qd, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS)", \
            "    __global__ inverse_dynamics_kernel<T>(T *d_c, const T *d_q_qd, const T *d_qdd, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS)", \
            "    __host__   inverse_dynamics<T,USE_QDD_FLAG=false,USE_COMPRESSED_MEM=false>(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps, const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams)", \
            "",\
            "    __device__ inverse_dynamics_vaf_device<T>(T *s_vaf, const T *s_q, const T *s_qd, const robotModel<T> *d_robotModel, const T gravity)", \
            "    __device__ inverse_dynamics_vaf_device<T>(T *s_vaf, const T *s_q, const T *s_qd, const T *s_qdd, const robotModel<T> *d_robotModel, const T gravity)", \
            "",\
            "    __device__ direct_minv_device<T>(T *s_Minv, const T *s_q, const robotModel<T> *d_robotModel)", \
            "    __global__ direct_minv_Kernel<T>(T *d_Minv, const T *d_q, const robotModel<T> *d_robotModel, const int NUM_TIMESTEPS)", \
            "    __host__   direct_minv<T,USE_COMPRESSED_MEM=false>(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const int num_timesteps, const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams)", \
            "",\
            "    __device__ forward_dynamics_device<T>(T *s_qdd, const T *s_q, const T *s_qd, const T *s_u, const robotModel<T> *d_robotModel, const T gravity)", \
            "    __global__ forward_dynamics_kernel<T>(T *d_qdd, const T *d_q_qd_u, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS)", \
            "    __host__   forward_dynamics<T>(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps, const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams)", \
            "",\
            "    __device__ inverse_dynamics_gradient_device<T>(T *s_dc_du, const T *s_q, const T *s_qd, const T *robotModel<T> *d_robotModel, const T gravity)", \
            "    __device__ inverse_dynamics_gradient_device<T>(T *s_dc_du, const T *s_q, const T *s_qd, const T *s_qdd, const robotModel<T> *d_robotModel, const T gravity)", \
            "    __global__ inverse_dynamics_gradient_kernel<T>(T *d_dc_du, const T *d_q_qd, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS)", \
            "    __global__ inverse_dynamics_gradient_kernel<T>(T *d_dc_du, const T *d_q_qd, const T *d_qdd, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS)", \
            "    __host__   inverse_dynamics_gradient<T,USE_QDD_FLAG=false,USE_COMPRESSED_MEM=false>(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps, const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams)", \
            "",\
            "    __device__ forward_dynamics_gradient_device<T>(T *s_df_du, const T *s_q, const T *s_qd, const T *s_u, const robotModel<T> *d_robotModel, const T gravity)",\
            "    __device__ forward_dynamics_gradient_device<T>(T *s_df_du, const T *s_q, const T *s_qd, const T *s_qdd, const T *s_Minv, const robotModel<T> *d_robotModel, const T gravity)", \
            "    __global__ forward_dynamics_gradient_kernel<T>(T *d_df_du, const T *d_q_qd_u, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS)", \
            "    __global__ forward_dynamics_gradient_kernel<T>(T *d_df_du, const T *d_q_qd, const T *d_qdd, const T *d_Minv, const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS)", \
            "    __host__   forward_dynamics_gradient<T,USE_QDD_MINV_FLAG=false>(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps, const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams)", \
            "","Suggested Type T is float",\
            "","Additional helper functions and ALGORITHM_inner functions which take in __shared__ memory temp variables exist -- see function descriptions in the file",\
            "","By default device and kernels need to be launched with dynamic shared mem of size <FUNC_CODE>_DYNAMIC_SHARED_MEM_COUNT where <FUNC_CODE> = [ID, MINV, FD, ID_DU, FD_DU]"]
        self.gen_add_func_doc("This instance of grid.cuh is optimized for the urdf: " + self.robot.name,file_notes)
        # then all of the includes (and namespaces and defines)
        self.gen_add_includes(use_thread_group)
        # then add the gpu error macro
        self.gen_add_gpu_err()
        # then open our namespace
        self.gen_add_func_doc("All functions are kept in this namespace")
        self.gen_add_code_line("namespace " + self.file_namespace + " {", True)
        # then generate any constants and other helpers
        self.gen_add_constants_helpers()
        # then the spatial algebra related helpers
        self.gen_spatial_algebra_helpers()
        # then generate the robot specific transformation and inertia matricies
        self.gen_init_topology_helpers()
        self.gen_init_XImats(include_base_inertia)
        self.gen_init_robotModel()
        self.gen_init_gridData()
        self.gen_load_update_XImats_helpers(use_thread_group)
        # then generate the robot optimized algorithms
        self.gen_inverse_dynamics(use_thread_group)
        self.gen_direct_minv(use_thread_group)
        self.gen_forward_dynamics(use_thread_group)
        self.gen_inverse_dynamics_gradient(use_thread_group)
        self.gen_forward_dynamics_gradient(use_thread_group)
        self.gen_crba_inner(use_thread_group)
        self.gen_aba(use_thread_group)
        # then finally the master init and close the namespace
        self.gen_init_close_grid()
        self.gen_add_end_control_flow()
        # then output to a file
        file = open(self.file_namespace + ".cuh","w")
        file.write(self.code_str)
        file.close()