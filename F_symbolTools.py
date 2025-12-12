
import numpy as

def triple_line_to_linear_index(X:np.array, M:np.array, N:np.array, j:np.array, shapes):
    """
        M
    _________
    ____X____|(j) ----- (XMN,j):= A
    _________|
        N
    """

    #A=np.ravel((X, M, N, j), shapes)
    A = np.ravel_multi_index((X, M, N, j), dims=shapes)
    shape_A = np.prod(shapes)
    return A, shape_A

def MPO_tensor(F):
    X, N_1, Y, M_2, M_1, N_2, i, k, l, j = F.coords
    X_size, N_1_size, Y_size, M_2_size, M_1_size, N_2_size, i_size, k_size, l_size, j_size = F.shape

    A, shape_A = triple_line_to_linear_index(X, M_1, N_1, i, (X_size, M_1_size, N_1_size, i_size))
    B, shape_B = triple_line_to_linear_index(Y, M_1, M_2, l, (Y_size, M_1_size, M_2_size, l_size))
    C, shape_C = triple_line_to_linear_index(X, M_2, N_2, j, (X_size, M_2_size, N_2_size, j_size))
    D, shape_D = triple_line_to_linear_index(Y, N_1, N_2, k, (Y_size, N_1_size, N_2_size, k_size))

    coords = [A, B, C, D]
    shape = [shape_A, shape_B, shape_C, shape_D]
    MPO_F = sparse.COO(coords, F.data, shape)

    return MPO_F

def fusion_tensor(F):
    M_1, Y_1, Y_2, M_2, M_3, Y_3, i, k, l, j = F.coords
    M_1_size, Y_1_size, Y_2_size, M_2_size, M_3_size, Y_3_size, i_size, k_size, l_size, j_size = F.shape

    A, shape_A = triple_line_to_linear_index(Y_1, M_1, M_3, i, (Y_1_size, M_1_size, M_3_size, i_size))
    B, shape_B = triple_line_to_linear_index(Y_3, M_1, M_2, j, (Y_3_size, M_1_size, M_2_size, j_size))
    C, shape_C = triple_line_to_linear_index(Y_2, M_2, M_3, k, (Y_2_size, M_2_size, M_3_size, k_size))

    coords = [l, A, B, C]
    shape = [l_size, shape_A, shape_B, shape_C]
    X = sparse.COO(coords, F.data, shape)

    return X


def test_F_symbol_unitarity(ω):
    #test = oe.contract('abcdefghij, klmnofpqij', ω, ω.conj())
    test = np.einsum('abcdefghij, abcdofpqij->abcd eo gp hq', ω, ω.conj())
    test = remove_zeros(test)
    return test

def commutator(A, B):
    return A@B - B@A

def build_f_abc_from_cayley_table(cayley_table, return_tube_compat_f = False, ):
    order = cayley_table.shape[0] 
    f = []
    dimension_dict = {}
    for a in range(order):
        for b in range(order):
            c = int(cayley_table[a,b]-1)
            f.append([a,b,c])
            dimension_dict[(a,b,c)]=(1,1,1)
    coords = np.array(f).T
    padding = np.ones_like(coords)-1
    #print(np.vstack([coords,padding]))
    vals = np.ones(coords.shape[1])
    shape = (order,order,order)
    f_sparse = sparse.COO(coords, vals, shape)

    return f_sparse, dimension_dict

def dim_dict_vec_G(cayley_table):
    order = cayley_table.shape[0]
    dim_dict = {}
    for g1 in range(order):
        for g2 in range(order):
            #g3 = cayley_table[g1,g2]-1
            for g3 in range(order):
                dim_dict[(g1,g2,g3)] = (1,1,1)

    return dim_dict

def F_vec_G(cayley_table):  
    """ Is this technically Vec over Vec(G) module cat F symbol? Or Rep(G) """
    order = cayley_table.shape[0]
    coords = []
    for g1 in range(order):
        for g2 in range(order):
            g12 = cayley_table[g1,g2]-1
            for g3 in range(order):
                g23 = cayley_table[g2,g3]-1
                g123 = cayley_table[g1,int(g23)]-1
                coords.append([g1,g2,g3,g123,g12,g23,0,0,0,0])
    vals = np.ones(len(coords))
    coords = np.array(coords).T.astype(int)
    shape = 6*[order]+4*[1]

    F_vec_G = sparse.COO(coords, vals, shape)

    return F_vec_G

def mod_cat_Vec_Vec_G(cayley_table):
    """ Module category Vec over Vec(G) F -symbol """
    order = cayley_table.shape[0]
    coords = []
    for g1 in range(order):
        for g2 in range(order):
            g1g2 = int(cayley_table[g1,g2]-1)
            coords.append([0,g1,g2,0,0,g1g2,0,0,0,0])
    coords = np.array(coords).T
    
    vals = np.ones(coords.shape[1])
    shape = (1,order,order,1,1,order,1,1,1,1)
    
    F_Vec_Vec_G = sparse.COO(coords, vals, shape)

    return F_Vec_Vec_G