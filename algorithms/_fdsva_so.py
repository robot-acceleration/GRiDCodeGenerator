import numpy as np
import copy 


def gen_fdsva_so_inner(self, use_thread_group = False): 
	# construct the boilerplate and function definition
    func_params = ["s_df2 are forward dynamics WRT q,qd,tau", \
                "s_q is the vector of joint positions", \
                "s_qd is the vector of joint velocities", \
                "s_qdd is the vector of joint accelerations", \
                "s_tau is the vector of joint torques", \
                "s_temp is the pointer to the shared memory needed of size: " + \
                            str(self.gen_fdsva_so_inner_temp_mem_size()), \
                "gravity is the gravity constant"]
    func_def_start = "void fdsva_so_inner("
    func_def_middle = "T *s_df2, T *s_q, T *s_qd, const T *s_qdd, const T *s_tau, "
    func_def_end = "T *s_temp, const T gravity) {"
    func_notes = ["Assumes works with IDSVA"]
    if use_thread_group:
        func_def_start = func_def_start.replace("(", "(cgrps::thread_group tgrp, ")
        func_params.insert(0,"tgrp is the handle to the thread_group running this function")
    func_def_middle, func_params = self.gen_insert_helpers_func_def_params(func_def_middle, func_params, -2)
    func_def = func_def_start + func_def_middle + func_def_end
    self.gen_add_func_doc("Second Order of Forward Dynamics with Spatial Vector Algebra", func_notes, func_params, None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)

    n = self.robot.get_num_pos()

    self.gen_add_code_line("__shared__ T s_Minv[" + str(n*n) + "];")
    self.gen_add_code_line("__shared__ T s_dc_du[" + str(2*n*n) + "];")
    self.gen_add_code_line("__shared__ T s_df_du[" + str(n*n*3) + "];")
    self.gen_add_code_line("__shared__ T s_di_du[" + str(n*n*n*4) + "];")
    self.gen_add_code_line("__shared__ T s_vaf[" + str(12*n) + "];")
    #declare s_d2tau_dq, s_d2tau_dqd, s_d2tau_cross, s_dm_dq from idsvaso
    self.gen_add_code_line("T *s_d2tau_dq = &s_di_du[" + str(n*n*n*0) + "];" )
    self.gen_add_code_line("T *s_d2tau_dqd = &s_di_du[" + str(n*n*n*1) + "];" )
    self.gen_add_code_line("T *s_d2tau_cross = &s_di_du[" + str(n*n*n*2) + "];" )
    self.gen_add_code_line("T *s_dm_dq = &s_di_du[" + str(n*n*n*3) + "];" )
    #declare df_dq, df_dqd, fd_dtau from df_du 
    self.gen_add_code_line("T *s_df_dq = s_df_du; T *s_df_dqd = &s_df_du[" + str(n*n) + "]; T *s_df_tau = &s_df_du[" + str(n*n*2) + "];")
    self.gen_add_code_line("__shared__ T s_a[12*6*7]; __shared__ T s_b[3*2*6*6*7]; __shared__ T s_c[2024]; __shared__ T s_d[7300];")
  
    
    self.gen_add_code_line("grid::direct_minv_inner<T>(s_Minv, s_q, s_XImats, s_temp);")
    
    self.gen_add_code_line("grid::fdsva_inner<T>(s_df_du, s_q, s_qd, s_qdd, s_tau, s_XImats, s_temp, gravity);")

    self.gen_add_code_line(" // grid::idsva_inner<T>(d2tau_dq, d2tau_dqd, d2tau_cross, s_dm_dq, s_a, s_q, s_qd, s_XImats, s_b, s_c, gravity, s_d);")


    #load in vals for idsvaso (hard code these vals until idsva_inner is ready)

    d2tau_dq  = [[[  0,       0,       0,       0,     0,       0,       0   ],
  [  0,      -3.7776, -33.6437,  -6.4784  , 1.1112,  -0.2367 , -0    ],
  [  0,     -33.6437 ,  0.5122 ,  1.1417  ,-0.2045 ,  0.3734 ,  0    ],
  [  0,      -6.4784  , 1.1417 ,  0.5624   ,0.0869  ,-0.1264 ,  0    ],
  [  0,       1.1112  ,-0.2045 ,  0.0869  , 0.8871 , -0.9337 , -0    ],
  [  0,      -0.2367   ,0.3734  ,-0.1264  ,-0.9337 ,  0.8705 , -0    ],
  [  0,      -0,      0,       0,      -0,     -0,    -0    ]],

 [[  0,       0,       0,      0,       0,      0,      0  ],
  [  0,     -15.2062,  -0.7944 , 22.6216 , -0.8058 , -1.1408 ,  0    ],
  [  0,      -0.7944 ,  5.1351  , 7.564   ,-1.541  ,  0.5702  , 0   ],
  [  0,      22.6216 ,  7.564  ,-15.6772  , 0.8337 ,  0.9211 , -0    ],
  [  0,      -0.8058 , -1.541  ,  0.8337 , -2.961  ,  1.0748  , 0    ],
  [  0,      -1.1408 ,  0.5702 ,  0.9211  , 1.0748  , 0.0666  ,-0    ],
  [  0,       0,      0,    -0,      0,      -0,      0    ]],

 [[   0,       0,      0,    -0,      0,      -0,      0       ],
  [  0,       0.0943, -22.9957 ,  0.545  , -0.4841 ,  0.3314 , -0   ],
  [  0,     -22.9957 ,  0.3321 , -4.2154 ,  0.0987 ,  0.4098 ,  0    ],
  [  0,       0.545  , -4.2154 , -0.4266 ,  0.384  , -0.1734 ,  0   ],
  [  0,      -0.4841 ,  0.0987 ,  0.384  ,  0.9858 , -1.1466 , -0    ],
  [  0,       0.3314 ,  0.4098 , -0.1734 , -1.1466 ,  1.0194 , -0   ],
  [   0,       0,      0,    -0,      0,      -0,      0      ]],

 [[  0,       0,      0,    -0,      0,      -0,      0      ],
  [  0 ,    22.5909 ,  1.6118 ,-22.1175,   0.7809 ,  1.1213 , -0   ],
  [  0,       1.6118 ,  6.2803 ,  0.5029,   0.1506 , -0.1397 , -0    ],
  [  0,    -22.1175  , 0.5029 , 24.0137  ,-0.7944  ,-1.0712  , 0    ],
  [  0,       0.7809 ,  0.1506,  -0.7944 ,  1.9665 , -0.4747 , -0    ],
  [  0,       1.1213 , -0.1397 , -1.0712 , -0.4747 , -0.1148 ,  0    ],
  [   0,       0,      0,    -0,      0,      -0,      0       ]],

 [[   0,       0,      0,    -0,      0,      -0,      0       ],
  [  0,      -0.77  ,  -0.51  ,   0.742  , -0.3332,  -0.1197,  -0    ],
  [  0 ,     -0.51   , -0.1555 , -0.2179  ,-0.1157 , -0.1466 ,  0    ],
  [  0,       0.742   ,-0.2179  ,-0.9033   ,0.0204 , -0.0684  , 0    ],
  [  0,     -0.3332  ,-0.1157   ,0.0204  ,-0.2871  , 0.357   ,-0    ],
  [  0,      -0.1197  ,-0.1466  ,-0.0684  , 0.357   ,-0.1104  ,-0    ],
  [   0,       0,      0,    -0,      0,      -0,      0      ]],

 [[   0,       0,      0,    -0,      0,      -0,      0      ],
  [  0,      -1.1581 ,  0.2502,   1.1424 , -0.1308 , -1.1767  , 0    ],
  [  0,       0.2502  ,-0.336  ,  0.0604 , -0.2431 ,  0.02    , 0    ],
  [  0,       1.1424  , 0.0604 , -1.0326 , -0.1065 ,  1.0678  ,-0    ],
  [  0,      -0.1308 , -0.2431 , -0.1065 ,  0.8071 ,  0.0014  , 0    ],
  [  0,      -1.1767 ,  0.02   ,  1.0678 ,  0.0014 ,  0.1306  ,-0    ],
  [   0,       0,      0,    -0,      0,      -0,      0      ]],

 [[   0,       0,      0,    -0,      0,      -0,      0      ],
  [  0,      -0.0006 , -0.0002 ,  0.0006 , -0.0002  ,-0.0005 , -0    ],
  [  0,      -0.0002  ,-0.001  ,  0.0094 , -0.0017  ,-0.009  , -0    ],
  [  0,       0.0006  , 0.0094 , -0.002  ,  0.0008  , 0.0018  , 0    ],
  [  0,      -0.0002  ,-0.0017 ,  0.0008 , -0.013   ,-0.1214 , -0    ],
  [  0,      -0.0005  ,-0.009  ,  0.0018 , -0.1214  , 0.0545   ,0    ],
  [   0,       0,      0,    -0,      0,      -0,      0       ]]]

    for i in range(n):
        d2tau_dq[i] = np.array(d2tau_dq[i]).T
    flat_d2tau_dq = (np.array(d2tau_dq).flatten())
    for k in range(n):
        for i in range(n):
            for j in range(n):
                self.gen_add_code_line("s_di_du["+ str(n*n*k + n*i + j) + "] = "+ str(flat_d2tau_dq[n*n*k + i + n*j]) + ";")

    for i in range(len(flat_d2tau_dq)):
        self.gen_add_code_line("s_di_du["+ str(i) + "] = "+ str(flat_d2tau_dq[i]) + ";")


    d2tau_dqd = [[[ 0 ,     0   ,   0  ,    0     , 0 ,    0  ,    0    ],
  [ 0,     -0.1945 ,-1.279  , 0.1688, -0.0154  ,0.0067,  0.0049],
  [ 0,     -1.279  ,-0.0339, -0.1428 , 0.0035  ,0.006 ,  0    ],
  [ 0,      0.1688 ,-0.1428, -0.0073 ,-0.0086 ,-0.0039, -0.0049],
  [ 0,     -0.0154 , 0.0035 ,-0.0086 ,-0.0294 ,-0.0514,  0.0006],
  [ 0,      0.0067 , 0.006 , -0.0039 ,-0.0514 ,-0.0123 , 0.0048],
  [ 0,      0.0049 , 0,     -0.0049  ,0.0006 , 0.0048 ,-0    ]],

 [[ 1.7669, -0.0973 , 1.279,  -0.1688 , 0.0154, -0.0067 ,-0.0049],
  [-0.0973, -0,     -0,     -0,      0,      0,     -0   ],
  [ 1.279 , -0,      2.3478, -0.1695 , 0.0112, -0.005  ,-0.0048],
  [-0.1688, -0,     -0.1695 , 1.9774 ,-0.0288 ,-0.0493 ,-0.0003],
  [ 0.0154,  0,      0.0112, -0.0288 , 0.0788 ,-0.0061 ,-0.0026],
  [-0.0067,  0,     -0.005 , -0.0493 ,-0.0061 , 0.1483 , 0.0012],
  [-0.0049, -0,     -0.0048, -0.0003 ,-0.0026 , 0.0012, -0    ]],

 [[ 0.0375 ,-1.279  ,-0.0169,  0.1428 ,-0.0035 ,-0.006 , -0   ],
  [-1.279 , -0.2303 , 1.1739,  0.1695 ,-0.0112 , 0.005 ,  0.0048],
  [-0.0169 , 1.1739 ,-0,     -0,      0,      0,     0    ],
  [ 0.1428 , 0.1695 ,-0,      0.0384 ,-0.0136, -0.0033, -0.0047],
  [-0.0035 ,-0.0112 , 0,     -0.0136 ,-0.0387, -0.0595, 0.0006],
  [-0.006  , 0.005 ,  0,     -0.0033 ,-0.0595, -0.011  , 0.0046],
  [-0     , 0.0048,  0,    -0.0047,  0.0006 , 0.0046 ,-0    ]],

 [[-0.4489 , 0.1688, -0.1428 ,-0.0036 , 0.0086,  0.0039 , 0.0049],
  [ 0.1688  ,2.0164, -0.1695  ,0.9887,  0.0288,  0.0493,  0.0003],
  [-0.1428 ,-0.1695, -0.3501  ,0.0192,  0.0136,  0.0033,  0.0047],
  [-0.0036  ,0.9887,  0.0192 , 0,    -0,    -0,     -0    ],
  [ 0.0086 , 0.0288,  0.0136 ,-0,     -0.0564 , 0.0137,  0.0026],
  [ 0.0039 , 0.0493,  0.0033 ,-0,      0.0137 ,-0.0507 ,-0.0009],
    [ 0.0049,  0.0003,  0.0047, -0,      0.0026 ,-0.0009,  0    ]],

 [[ 0.0122, -0.0154 , 0.0035, -0.0086, -0.0147 , 0.0514, -0.0006],
  [-0.0154 ,-0.0797  ,0.0112, -0.0288,  0.0394,  0.0061,  0.0026],
  [ 0.0035 , 0.0112  ,0.0101, -0.0136, -0.0193,  0.0595, -0.0006],
  [-0.0086 ,-0.0288, -0.0136, -0.0009, -0.0282, -0.0137, -0.0026],
  [-0.0147 , 0.0394, -0.0193, -0.0282, -0,     -0,      0    ],
  [ 0.0514 , 0.0061,  0.0595, -0.0137, -0,     -0,      0.0027],
  [-0.0006 , 0.0026, -0.0006, -0.0026,  0,      0.0027, -0    ]],

 [[-0.0204,  0.0067,  0.006 , -0.0039, -0.0514, -0.0061, -0.0048],
  [ 0.0067 ,-0.1434, -0.005 , -0.0493, -0.0061,  0.0741, -0.0012],
  [ 0.006 , -0.005  ,-0.0375 ,-0.0033, -0.0595, -0.0055, -0.0046],
  [-0.0039, -0.0493, -0.0033 ,-0.0525,  0.0137 ,-0.0254 , 0.0009],
  [-0.0514, -0.0061, -0.0595,  0.0137 , 0.0116 ,-0,     -0.0027],
  [-0.0061,  0.0741, -0.0055, -0.0254 ,-0,      0,      0    ],
  [-0.0048, -0.0012, -0.0046,  0.0009, -0.0027,  0,     -0    ]],

 [[ 0 ,     0.0049 , 0,     -0.0049 , 0.0006 , 0.0048,  0    ],
  [ 0.0049, -0 ,   -0.0048, -0.0003 ,-0.0026 , 0.0012,  0    ],
  [ 0  ,   -0.0048 , 0  ,   -0.0047 , 0.0006 , 0.0046 , 0    ],
  [-0.0049, -0.0003 ,-0.0047 ,-0   ,   0.0026 ,-0.0009 , 0    ],
  [ 0.0006 ,-0.0026 , 0.0006 , 0.0026 , 0     , 0.0027 , 0    ],
  [ 0.0048 , 0.0012 , 0.0046 ,-0.0009 , 0.0027 ,-0     , 0    ],
  [ 0,   0,      0,      0,      0,      0,      0    ]]]

    for i in range(n):
        d2tau_dqd[i] = np.array(d2tau_dqd[i]).T
    flat_d2tau_dqd = (np.array(d2tau_dqd).flatten())
    for k in range(n):
        for i in range(n):
            for j in range(n):
                self.gen_add_code_line("s_di_du["+ str(n*n*n +n*n*k + n*i + j) + "] = "+ str(flat_d2tau_dqd[n*n*k + i + n*j]) + ";")

    for i in range(len(flat_d2tau_dqd)):    
        self.gen_add_code_line("s_di_du["+ str(7*7*7 +i) + "] = "+ str(flat_d2tau_dqd[i]) + ";")

    d2tau_cross = [[[ -0 ,     0   ,   0  ,    0     , 0 ,    0  ,    0    ],
  [ 0,     0 ,0  , 0, 0  ,0,  0],
  [ 0,     0  ,0, 0 , 0  ,0 ,  0    ],
  [ 0,      0 ,0, 0 ,0 ,0, 0],
  [ 0,     0 , 0 ,0 ,0 ,0,  0],
  [ 0,      0 , 0 , 0 ,0 ,0, 0],
  [ 0,      0 , 0,     0  ,0 , 0 ,0    ]],

 [[ -0 ,     -0   ,   0  ,    0     , 0 ,    0  ,    0    ],
  [ 0,     -0 ,0  , 0, 0  ,0,  0],
  [ 0,     0  ,0, 0 , 0  ,0 ,  0    ],
  [ 0,      0 ,0, 0 ,0 ,0, 0],
  [ 0,     0 , 0 ,0 ,0 ,0,  0],
  [ 0,      0 , 0 , 0 ,0 ,0, 0],
  [ 0,      0 , 0,     0  ,0 , 0 ,0    ]],

 [[ -0 ,     -0   ,   -0  ,    0     , 0 ,    0  ,    0    ],
  [ 0,     -0 ,-0  , 0, 0  ,0,  0],
  [ 0,     0  ,-0, 0 , 0  ,0 ,  0    ],
  [ 0,      0 ,0, 0 ,0 ,0, 0],
  [ 0,     0 , 0 ,0 ,0 ,0,  0],
  [ 0,      0 , 0 , 0 ,0 ,0, 0],
  [ 0,      0 , 0,     0  ,0 , 0 ,0    ]],

 [[ -0 ,     -0   ,   -0  ,    -0     , 0 ,    0  ,    0    ],
  [ 0,     -0 ,-0  , -0, 0  ,0,  0],
  [ 0,     0  ,-0, -0 , 0  ,0 ,  0    ],
  [ 0,      0 ,0, -0 ,0 ,0, 0],
  [ 0,     0 , 0 ,0 ,0 ,0,  0],
  [ 0,      0 , 0 , 0 ,0 ,0, 0],
  [ 0,      0 , 0,     0  ,0 , 0 ,0    ]],

 [[ -0 ,     -0   ,  -0  ,    -0     , -0 ,    0  ,    0    ],
  [ 0,     -0 ,-0  , -0, -0  ,0,  0],
  [ 0,     0  ,0, -0 , -0  ,0 ,  0    ],
  [ 0,      0 ,0, 0 ,-0 ,0, 0],
  [ 0,     0 , 0 ,0 ,0 ,0,  0],
  [ 0,      0 , 0 , 0 ,0 ,0, 0],
  [ 0,      0 , 0,     0  ,0 , 0 ,0    ]],

[[ -0 ,     -0   ,   -0  ,    -0     , -0 ,    -0  ,    0    ],
  [ 0,     -0 ,-0  , -0, -0  ,-0,  0],
  [ 0,     0  ,-0, -0 , -0  ,-0 ,  0    ],
  [ 0,      0 ,0, -0 ,-0 ,-0, 0],
  [ 0,     0 , 0 ,0 ,-0 ,-0,  0],
  [ 0,      0 , 0 , 0 ,0 ,-0, 0],
  [ 0,      0 , 0,     0  ,0 , 0 ,0    ]],

 [[ -0 ,     -0   ,   -0  ,    -0     , -0 ,    -0  ,    -0    ],
  [ 0,     -0 ,-0  , -0, -0  ,-0,  -0],
  [ 0,     0  ,-0, -0 , -0  ,-0 ,  -0    ],
  [ 0,      0 ,0, -0 ,-0 ,-0, -0],
  [ 0,     0 , 0 ,0 ,-0 ,-0,  -0],
  [ 0,      0 , 0 , 0 ,0 ,-0, -0],
  [ 0,      0 , 0,     0  ,0 , 0 ,-0    ]]]
    
    for i in range(n):
        d2tau_cross[i] = np.array(d2tau_cross[i]).T
    flat_d2tau_cross = (np.array(d2tau_cross).flatten())
    for k in range(n):
        for i in range(n):
            for j in range(n):
                self.gen_add_code_line("s_di_du["+ str(n*n*n*2 +n*n*k + n*i + j) + "] = "+ str(flat_d2tau_cross[n*n*k + i + n*j]) + ";")
    # for i in range(7*7*7):
    #     self.gen_add_code_line("s_di_du["+ str(2*7*7*7 +i) + "] = "+ str(0) + ";")

    dM_dq = [[[ 0,     -1.7669, -0.0375,  0.4489, -0.0122 , 0.0204, -0   ],
  [ 0,    -0.0973  ,1.2634 , 0.0692  ,0.03   ,-0.0239 ,-0   ],
  [ 0,     -1.279  ,-0.0169,  0.4199 ,-0.0117,  0.0279, -0    ],
  [ 0,      0.1688 ,-0.1428, -0.0036 ,-0.0111 , 0.0036,  0    ],
  [ 0,     -0.0154 , 0.0035, -0.0086 ,-0.0147, -0.0413, -0    ],
  [ 0,      0.0067 , 0.006,  -0.0039 ,-0.0514, -0.0061, -0    ],
  [ 0,      0.0049 , 0,     -0.0049  ,0.0006 , 0.0048 , 0    ]],

 [[ 0,    -0.0973 , 1.2634 , 0.0692 , 0.03 ,  -0.0239 ,-0   ],
  [ 0,      0,      0.2303 ,-2.0164 , 0.0797,  0.1434 , 0    ],
  [ 0,      0,      1.1739 , 0.0768 , 0.0267 ,-0.022  ,-0    ],
  [ 0,      0,     -0.1695 , 0.9887 ,-0.0397 ,-0.097  ,-0    ],
  [ 0,      0,      0.0112 ,-0.0288 , 0.0394 ,-0.0052 ,-0    ],
  [ 0,      0,     -0.005  ,-0.0493 ,-0.0061 , 0.0741 , 0    ],
  [ 0,      0,     -0.0048 ,-0.0003 ,-0.0026 , 0.0012  ,0    ]],

[[ 0,    -1.279 , -0.0169 , 0.4199, -0.0117 , 0.0279, -0    ],
  [ 0,      0,      1.1739,  0.0768,  0.0267, -0.022,  -0    ],
  [ 0,      0,      0,      0.3501 ,-0.0101 , 0.0375 ,-0    ],
  [ 0,      0,      0,      0.0192 ,-0.0086 , 0,      0    ],
  [ 0,      0,      0,     -0.0136 ,-0.0193 ,-0.0474, -0    ],
  [ 0,      0,      0,     -0.0033 ,-0.0595 ,-0.0055 ,-0    ],
  [ 0,      0,      0,     -0.0047 , 0.0006 , 0.0046 , 0    ]],

 [[ 0,      0.1688 ,-0.1428, -0.0036, -0.0111,  0.0036,  0    ],
  [ 0,     0,     -0.1695  ,0.9887 ,-0.0397, -0.097 , -0    ],
  [ 0,      0,     0,      0.0192 ,-0.0086 , 0,      0    ],
  [ 0,      0,      0,      0,      0.0009 , 0.0525,  0    ],
  [ 0,      0,      0,      0,     -0.0282 , 0.0103, 0    ],
  [ 0,      0,      0,      0,      0.0137 ,-0.0254,-0   ],
  [ 0,      0,      0,      0,      0.0026 ,-0.0009 , 0    ]],

 [[ 0,     -0.0154 , 0.0035, -0.0086, -0.0147, -0.0413,-0    ],
  [ 0,      0,      0.0112, -0.0288,  0.0394, -0.0052, -0    ],
  [ 0,      0,      0,     -0.0136, -0.0193, -0.0474, -0    ],
  [ 0,      0,      0,      0,    -0.0282,  0.0103,  0    ],
  [ 0,      0,      0,      0,      0,     -0.0116 ,-0    ],
  [ 0,      0,      0,      0,      0,     -0,     -0    ],
  [ 0,      0,      0,      0,      0,      0.0027,  0    ]],

 [[ 0,      0.0067,  0.006,  -0.0039, -0.0514, -0.0061 ,-0    ],
  [ 0,      0,     -0.005,  -0.0493, -0.0061 , 0.0741,  0    ],
  [ 0,      0,      0,     -0.0033 ,-0.0595 ,-0.0055 ,-0    ],
  [ 0 ,     0,      0  ,    0,      0.0137 ,-0.0254, -0    ],
  [ 0,      0,      0,     0,     0,     -0,     -0    ],
  [ 0,      0,      0,      0,      0,      0,      0    ],
  [ 0,      0,      0,      0,      0,      0,      0    ]],

 [[ 0,      0.0049 , 0,     -0.0049 , 0.0006,  0.0048,  0    ],
  [ 0,      0,    -0.0048 ,-0.0003, -0.0026,  0.0012 , 0    ],
  [ 0,      0,      0,     -0.0047,  0.0006 , 0.0046,  0    ],
  [ 0,      0,      0,      0,      0.0026 ,-0.0009,  0    ],
  [ 0,      0,      0,      0,      0,      0.0027 , 0    ],
  [ 0,      0,      0,      0,      0,      0,      0    ],
  [ 0,      0,      0,      0,      0,     0,      0    ]]]

    for i in range(n):
        dM_dq[i] = np.array(dM_dq[i]).T
    flat_dM_dq = (np.array(dM_dq).flatten())
    for k in range(n):
        for i in range(n):
            for j in range(n):
                self.gen_add_code_line("s_di_du["+ str(3*n*n*n +n*n*k + n*i + j) + "] = "+ str(flat_dM_dq[n*n*k + n*i + j]) + ";")
    

    self.gen_add_sync(use_thread_group)
    
    self.gen_add_code_line("T *dM_dqxfd_dq = s_temp;")

    self.gen_add_parallel_loop("ind",str(n*n*n),use_thread_group)
    #fill in mat mult of dM_dq + df_dq
    self.gen_add_code_line("int page = ind / " + str(n*n) + ";")
    self.gen_add_code_line("int row = ind % " + str(n) + ";")
    self.gen_add_code_line("int col = ind % " + str(n*n) + " / " + str(n) + ";")
    # self.gen_add_code_line("dM_dqxfd_dq[" + str(i*n*n) + "+ ind] = dot_prod<T,7,7,1>(&s_dm_dq" + str(i) + "[row], &s_df_dq[" + str(n) + "*row + col]);")
    self.gen_add_code_line("dM_dqxfd_dq[ind] = dot_prod<T," + str(n) + "," + str(n) + ",1>(&s_dm_dq[" + str(n*n) + "*page + row], &s_df_dq[" + str(n) + "*col]);")
    self.gen_add_end_control_flow()
    self.gen_add_sync(use_thread_group)

    #rotation of dM_dq + df_dq
    self.gen_add_code_line("T *rot_dM_dqxfd_dqd = &s_temp[0];")
    for i in range(n):
        self.gen_add_parallel_loop("ind",str(n*n),use_thread_group)
        self.gen_add_code_line("int page = "+ str(i) + ";")
        self.gen_add_code_line("int row = ind % " + str(n) + ";")
        self.gen_add_code_line("int col = ind / " + str(n) + ";")
        self.gen_add_code_line("rot_dM_dqxfd_dqd["+ str(n*n) + "*col + row + "+ str(n) + "*page] = dM_dqxfd_dq["+ str(n*n) + "*page + ind];")
        # self.gen_add_code_line("rot_dM_dqxfd_dqd = rotateMatrix(dM_dqxfd_dq, 7,7,7);")
        self.gen_add_end_control_flow()
        self.gen_add_sync(use_thread_group)

    self.gen_add_parallel_loop("ind",str(n*n*n),use_thread_group)
    # #big addition step for df2_dq
    self.gen_add_code_line("s_df2[ind] = s_di_du[ind] + dM_dqxfd_dq[ind] + rot_dM_dqxfd_dqd[ind];")
    self.gen_add_end_control_flow()
    self.gen_add_sync(use_thread_group)

    self.gen_add_code_line("T *dM_dqxfd_dqd = s_temp;")

    self.gen_add_parallel_loop("ind",str(n*n*n),use_thread_group)
    #fill in mat mult of dM_dq + df_dqd
    self.gen_add_code_line("int page = ind / " + str(n*n) + ";")
    self.gen_add_code_line("int row = ind % " + str(n) + ";")
    self.gen_add_code_line("int col = ind % " + str(n*n) + " / " + str(n) + ";")
    # self.gen_add_code_line("dM_dqxfd_dq[" + str(i*n*n) + "+ ind] = dot_prod<T,7,7,1>(&s_dm_dq" + str(i) + "[row], &s_df_dq[" + str(n) + "*row + col]);")
    self.gen_add_code_line("dM_dqxfd_dqd[ind] = dot_prod<T," + str(n) + "," + str(n) + ",1>(&s_dm_dq[" + str(n*n) + "*page + row], &s_df_dqd[" + str(n) + "*col]);")
    self.gen_add_end_control_flow()
    self.gen_add_sync(use_thread_group)


    self.gen_add_parallel_loop("ind",str(n*n*n),use_thread_group)
    #big addition step for df2_dq_qd
    self.gen_add_code_line("s_df2[" + str(n*n*n) + "+ ind] = s_d2tau_cross[ind] + dM_dqxfd_dqd[ind];")
    #load val for df2_dqd
    self.gen_add_code_line("s_df2[" + str(n*n*n*2) + "+ ind] = s_d2tau_dqd[ind];")
    self.gen_add_end_control_flow()
    self.gen_add_sync(use_thread_group)

    #fill in mat mult of dM_dq + Minv
    #fix minv 
    for row_m in range(n):
        for col_m in range(n):
            if row_m > col_m:
                self.gen_add_code_line("s_Minv["+ str(row_m + col_m*n) + "] = s_Minv["+ str(row_m*n + col_m) + "];")

    self.gen_add_sync(use_thread_group)
    
    self.gen_add_parallel_loop("ind",str(n*n*n),use_thread_group)
    #fill in mat mult of dM_dq + df_dqd
    self.gen_add_code_line("int page = ind / " + str(n*n) + ";")
    self.gen_add_code_line("int row = ind % " + str(n) + ";")
    self.gen_add_code_line("int col = ind % " + str(n*n) + " / " + str(n) + ";")
    # self.gen_add_code_line("dM_dqxfd_dq[" + str(i*n*n) + "+ ind] = dot_prod<T,7,7,1>(&s_dm_dq" + str(i) + "[row], &s_df_dq[" + str(n) + "*row + col]);")
    self.gen_add_code_line("s_df2[" + str(n*n*n*3) + "+ ind] = dot_prod<T," + str(n) + "," + str(n) + ",1>(&s_dm_dq[" + str(n*n) + "*page + row], &s_Minv[" + str(n) + "*col]);")
    self.gen_add_end_control_flow()
    self.gen_add_sync(use_thread_group)
    

    self.gen_add_code_line("T *s_df2_temp = &s_di_du[0];")
    

    self.gen_add_parallel_loop("ind",str(n*n*n*4),use_thread_group)
    #fill in mat mult of everything w minv
    self.gen_add_code_line("int page = ind / " + str(n*n) + ";")
    self.gen_add_code_line("int row = ind % " + str(n) + ";")
    self.gen_add_code_line("int col = ind % " + str(n*n) + " / " + str(n) + ";")
    self.gen_add_code_line("s_df2_temp[ind] = dot_prod<T," + str(n) + "," + str(n) + ",1>(&s_Minv[row], &s_df2[" + str(n*n) + "*page + " + str(n) + "*col]);")
    self.gen_add_end_control_flow()
    self.gen_add_sync(use_thread_group)

    self.gen_add_parallel_loop("ind",str(n*n*n*4),use_thread_group)
    #mult everything by -1
    self.gen_add_code_line("s_df2[ind] = s_df2_temp[ind];")
    self.gen_add_code_line("s_df2[ind] *= (-1);")
    self.gen_add_end_control_flow()
    self.gen_add_sync(use_thread_group)

    self.gen_add_end_function()

    

    
def gen_fdsva_so_inner_temp_mem_size(self):
    n = self.robot.get_num_pos()
    return n*n*n*4*3
    
def gen_fdsva_so_inner_function_call(self, use_thread_group = False, updated_var_names = None):
    var_names = dict( \
        s_df2_name = "s_df2", \
        s_q_name = "s_q", \
        s_qd_name = "s_qd", \
        s_qdd_name = "s_qdd", \
        s_tau_name = "s_tau", \
        s_temp_name = "s_temp", \
        gravity_name = "gravity"
    )
    if updated_var_names is not None:
        for key,value in updated_var_names.items():
            var_names[key] = value
    fdsva_so_code_start = "fdsva_so_inner<T>(" + var_names["s_df2_name"] + ", " + var_names["s_q_name"] + ", " + var_names["s_qd_name"] + ", " + var_names["s_qdd_name"] + ", " + var_names["s_tau_name"] + ", " 
    fdsva_so_code_end = var_names["s_temp_name"] + ", " + var_names["gravity_name"] + ");"
    if use_thread_group:
        id_code_start = id_code_start.replace("(","(tgrp, ")
    fdsva_so_code_middle = self.gen_insert_helpers_function_call()
    fdsva_so_code = fdsva_so_code_start + fdsva_so_code_middle + fdsva_so_code_end
    self.gen_add_code_line(fdsva_so_code)

def gen_fdsva_so_device_temp_mem_size(self):
    n = self.robot.get_num_pos()
    wrapper_size = self.gen_topology_helpers_size() 
    return self.gen_fdsva_so_inner_temp_mem_size() + wrapper_size

def gen_fdsva_so_device(self, use_thread_group = False):
    n = self.robot.get_num_pos()
    # construct the boilerplate and function definition
    func_params = ["s_q is the vector of joint positions", \
                   "s_qd is the vector of joint velocities", \
                   "s_qdd is the vector of joint accelerations", \
                   "s_tau is the vector of joint torques", \
                   "d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)", \
                   "gravity is the gravity constant"]
    func_notes = []
    func_def_start = "void fdsva_so_device("
    func_def_middle = "const T *s_q, const T *s_qd, const T *s_qdd, const T *s_tau,"
    func_def_end = "const robotModel<T> *d_robotModel, const T gravity) {"
    if use_thread_group:
        func_def_start += "cgrps::thread_group tgrp, "
        func_params.insert(0,"tgrp is the handle to the thread_group running this function")
    func_def = func_def_start + func_def_middle + func_def_end

    # then generate the code
    self.gen_add_func_doc("Compute the FDSVA_SO (Second Order of Forward Dyamics with Spacial Vector Algebra)",\
                          func_notes,func_params,None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__device__")
    self.gen_add_code_line(func_def, True)

    # add the shared memory variables
    shared_mem_size = self.gen_fdsva_so_device_temp_mem_size() if not self.use_dynamic_shared_mem_flag else None
    self.gen_XImats_helpers_temp_shared_memory_code(shared_mem_size)

    self.gen_add_code_line("extern __shared__ T s_df2[" + str(3*n*3*n*n) + "];")
    
    # then load/update XI and run the algo
    self.gen_load_update_XImats_helpers_function_call(use_thread_group)
    self.gen_fdsva_so_inner_function_call(use_thread_group)
    self.gen_add_end_function()

def gen_fdsva_so_kernel(self, use_thread_group = False, single_call_timing = False):
    n = self.robot.get_num_pos()
    # define function def and params
    func_params = ["d_q_qd_qdd_tau is the vector of joint positions, velocities, accelerations", \
                    "stride_q_qd_qdd is the stride between each q, qd, qdd", \
                    "d_robotModel is the pointer to the initialized model specific helpers on the GPU (XImats, topology_helpers, etc.)", \
                    "d_tau is the vector of joint torques", \
                    "gravity is the gravity constant", \
                    "num_timesteps is the length of the trajectory points we need to compute over (or overloaded as test_iters for timing)"]
    func_notes = []
    func_def_start = "void fdsva_so_kernel(T *d_df2, const T *d_q_qd_qdd_tau, const int stride_q_qd_qdd, "
    func_def_end = "const robotModel<T> *d_robotModel, const T gravity, const int NUM_TIMESTEPS) {"
    func_def = func_def_start + func_def_end
    if single_call_timing:
        func_def = func_def.replace("kernel(", "kernel_single_timing(")
    
    # then generate the code
    self.gen_add_func_doc("Compute the FDSVA_SO (Second Order of Forward Dynamics with Spacial Vector Algebra)", \
                            func_notes, func_params, None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__global__")
    self.gen_add_code_line(func_def, True)

    # add shared memory variables
    shared_mem_vars = ["__shared__ T s_df2[" + str(4*n*n*n) + "];", \
                        "__shared__ T s_q_qd_qdd_tau[4*" + str(n) + "]; T *s_q = s_q_qd_qdd_tau; T *s_qd = &s_q_qd_qdd_tau[" + str(n) + "]; T *s_qdd = &s_q_qd_qdd_tau[2 * " + str(n) + "]; T *s_tau = &s_q_qd_qdd_tau[3 * " + str(n) + "];",\
                        ]
    self.gen_add_code_lines(shared_mem_vars)
    shared_mem_size = self.gen_fdsva_so_inner_temp_mem_size() if not self.use_dynamic_shared_mem_flag else None
    self.gen_XImats_helpers_temp_shared_memory_code(shared_mem_size)
    if use_thread_group:
        self.gen_add_code_line("cgrps::thread_group tgrp = TBD;")
    if not single_call_timing:
        # load to shared mem and loop over blocks to compute all requested comps
        self.gen_add_parallel_loop("k","NUM_TIMESTEPS",use_thread_group,block_level = True)
        self.gen_kernel_load_inputs("q_qd_qdd_tau","stride_q_qd_qdd",str(3*n),use_thread_group)
        # compute
        self.gen_add_code_line("// compute")
        self.gen_load_update_XImats_helpers_function_call(use_thread_group)
        self.gen_fdsva_so_inner_function_call(use_thread_group)
        self.gen_add_sync(use_thread_group)
        # save to global
        self.gen_kernel_save_result("df2","1",str(4*n*n*n),use_thread_group)
        self.gen_add_end_control_flow()
    else:
        # repurpose NUM_TIMESTEPS for number of timing reps
        self.gen_kernel_load_inputs_single_timing("q_qd_qdd_tau",str(4*n),use_thread_group)
        # then compute in loop for timing
        self.gen_add_code_line("// compute with NUM_TIMESTEPS as NUM_REPS for timing")
        self.gen_add_code_line("for (int rep = 0; rep < NUM_TIMESTEPS; rep++){", True)
        self.gen_load_update_XImats_helpers_function_call(use_thread_group)
        self.gen_fdsva_so_inner_function_call(use_thread_group)
        self.gen_add_end_control_flow()
        # save to global
        self.gen_kernel_save_result_single_timing("df2",str(4*n*n*n),use_thread_group)
    self.gen_add_end_function()

def gen_fdsva_so_host(self, mode = 0):
    n = self.robot.get_num_pos()
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
    func_def_start = "void fdsva_so(gridData<T> *hd_data, const robotModel<T> *d_robotModel, const T gravity, const int num_timesteps,"
    func_def_end =   "                      const dim3 block_dimms, const dim3 thread_dimms, cudaStream_t *streams) {"
    if single_call_timing:
        func_def_start = func_def_start.replace("(", "_single_timing(")
        func_def_end = "              " + func_def_end
    if compute_only:
        func_def_start = func_def_start.replace("(", "_compute_only(")
        func_def_end = "             " + func_def_end.replace(", cudaStream_t *streams", "")
    # then generate the code
    self.gen_add_func_doc("Compute the FDSVA_SO (Second Order of Forward Dynamics with Spacial Vector Algebra)",\
                          func_notes,func_params,None)
    self.gen_add_code_line("template <typename T>")
    self.gen_add_code_line("__host__")
    self.gen_add_code_line(func_def_start)
    self.gen_add_code_line(func_def_end, True)

    func_call_start = "fdsva_so_kernel<T><<<block_dimms,thread_dimms,FDSVA_SO_SHARED_MEM_COUNT*sizeof(T)>>>(hd_data->d_df2,hd_data->d_q_qd_u,stride_q_qd_qdd,"
    func_call_end = "d_robotModel,gravity,num_timesteps);"
    self.gen_add_code_line("int stride_q_qd_qdd = 3*NUM_JOINTS;")
    if single_call_timing:
        func_call_start = func_call_start.replace("kernel<T>","kernel_single_timing<T>")
    if not compute_only:
        # start code with memory transfer
        self.gen_add_code_lines(["// start code with memory transfer", \
                                 "gpuErrchk(cudaMemcpyAsync(hd_data->d_q_qd_u,hd_data->h_q_qd_u,stride_q_qd_qdd" + \
                                    ("*num_timesteps" if not single_call_timing else "") + "*sizeof(T),cudaMemcpyHostToDevice,streams[0]));", \
                                 "gpuErrchk(cudaDeviceSynchronize());"])
    # then compute:
    self.gen_add_code_line("// call the kernel")
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
                                "gpuErrchk(cudaMemcpy(hd_data->h_df2,hd_data->d_df2," + \
                                ("num_timesteps*" if not single_call_timing else "") + str(3*n*3*n*n) + "*sizeof(T),cudaMemcpyDeviceToHost));",
                                "gpuErrchk(cudaDeviceSynchronize());"])
    # finally report out timing if requested
    if single_call_timing:
        self.gen_add_code_line("printf(\"Single Call FDSVA_SO %fus\\n\",time_delta_us_timespec(start,end)/static_cast<double>(num_timesteps));")
    self.gen_add_end_function()

def gen_fdsva_so(self, use_thread_group = False):
    # first generate the inner helper
    self.gen_fdsva_so_inner(use_thread_group)
    # then generate the device wrapper
    self.gen_fdsva_so_device(use_thread_group)
    # then generate the kernels
    self.gen_fdsva_so_kernel(use_thread_group, True)
    self.gen_fdsva_so_kernel(use_thread_group, False)
    # then generate the host wrappers
    self.gen_fdsva_so_host(0)
    self.gen_fdsva_so_host(1)
    self.gen_fdsva_so_host(2)
    
