
# Standard Library
import string
from typing import Dict
from functools import lru_cache

# Third Party Library
import scipy.io
import scipy as sp
import numpy as np
import pandas as pd
import sparse
import opt_einsum as oe

# --- Slicing Function for Sparse High Rank Tensors ---
import numpy as np
import opt_einsum as oe

def slice_tensor(F:sparse.SparseArray, slices:Dict):
    """
    Slice a high-rank tensor using opt_einsum selector vectors

    Args:
        F (sparse.SparseArray): F symbol/input tensor of rank N
        slices (DICT): Keys are axis indices, values are the value of the axis we want to slice at (i.e. the index to fix)
            - E.g: {0: N_1, 3: N_3, 4: N_2}

    Returns:
        sliced (sparse.SparseArray): Sliced tensor 
    """
    rank = F.ndim
    letters = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    input_labels = letters[:rank]
    output_labels = []

    tensors = [F]
    einsum_str = "".join(input_labels)

    def one_hot_sparse(length, index, dtype=float):
        """ Makes a sparse 1-hot vector of zeroes of a given length with a 1 at 'index' """
        coords = [[index]]        
        data = [1.0]              
        return sparse.COO(coords, data, shape=(length,))

    for axis in range(rank):
        if axis in slices:
            sel = one_hot_sparse(F.shape[axis], slices[axis], dtype=F.dtype)
            tensors.append(sel)
            einsum_str += "," + input_labels[axis] 
        else:
            output_labels.append(input_labels[axis])

    out_str = "".join(output_labels)
    einsum_eq = f"{''.join(input_labels)}{einsum_str[len(input_labels):]}->{out_str}"
    ##print(einsum_eq)
    sliced = oe.contract(einsum_eq, *tensors, backend="sparse")
    return sliced

_slice_tensor_cache = {}
def slice_tensor_cached(F: sparse.SparseArray, slices: dict):
    """
    Cached version of slice_tensor.
    """
    key = (id(F), tuple(sorted(slices.items())))
    
    if key in _slice_tensor_cache:
        return _slice_tensor_cache[key]
    else:
        result = slice_tensor(F, slices)
        _slice_tensor_cache[key] = result
        return result


# --- Tuple <-> Index ---
def tuple_to_index(tup, shape):
    return np.ravel_multi_index(tup, shape)


def index_to_tuple(idx, shape):
    return np.unravel_index(idx, shape)

def remove_zeros(sparse_tensor, tol= 1e-12):
        mask = np.abs(sparse_tensor.data) >= tol
        return  sparse.COO(sparse_tensor.coords[:, mask], sparse_tensor.data[mask], shape=sparse_tensor.shape)

# --- Fusion Rules Factory ---
def make_fusion_rules(F: sparse.SparseArray, size: Dict):
    @lru_cache(maxsize=None)
    def fusion_rules(M_3: int, M_1: int, tol: float = 1e-10) -> pd.DataFrame:
        """
        Args:
            F (sparse.SparseArray): F-symbol for module category
            M_3, M_1 (int): module indices
            size (Dict): size is a dictionary containing the number of fusion objects, the number of module objects and the 

        Returns:
            fusion_rules (np.ndarray): fusion rules for N_M_3Y1_M_1
            nonzero_M_3M_1_df (pd.DataFrame): given an M_3, M_1 gives the Y, alpha indices of nonzero tube elements


        The fusion rules of a fusion category C are embedded in the module category M over a fusion category C.
        We extract them by considering diagrams:

           M_3  1  Y2        M_3  1  Y2
            |  /  /          |  \  /
            | /  /           |   \/k
          í |/  /            |   /
            |  /      =~ F   |  / Y3
            | /              | /
           j|/              l|/
            |                |
            M_1               M_1

        i = k = 0 since multiplicity space one dimensional for identity object of fusion category
        j = l as F is diagonal for this fusion (later I call j and k alpha and beta for some reason)
        
        Find the rank of F to find the dimension of each multipliicty index i.e. each hom space (the fusion rules)
        - Most should be one dimensional
        """
        #print(f"Finding fusion rules for {M_3, M_1}")
        nonzero_M_3M_1 = []
        fusion_rules_dict = {}
        Y1 = 0
        #F(M1,Y1,Y2,M2,M3,Y3,i,k,l,j)
        for Y2 in range(size['fusion_label']):
            #hom_space = F[M_3, Y1, Y2, M_1, M_3, Y2, 0, :, 0, :] # Multiplicity space of identity element is always one dimensional
            #print(f"M_3, Y1, Y2, M_1, M_3, Y2 = {M_3, Y1, Y2, M_1, M_3, Y2}")
            hom_space = slice_tensor(F, {0:M_3, 
                                         1:Y1, 
                                         2:Y2, 
                                         3:M_1, 
                                         4:M_3, 
                                         5:Y2, 
                                         6:0, 
                                         8:0 })
            
            # List index of nonzero fusion elements
            mask = np.abs(hom_space.data) > tol
            N_M_3Y1_M_1 = np.count_nonzero(mask)
            nonzero_tubes = hom_space.coords[:, mask].T

            if N_M_3Y1_M_1 > 0:
                fusion_rules_dict[M_3, M_1, Y2] = N_M_3Y1_M_1
                for a in nonzero_tubes:
                    nonzero_M_3M_1.append([Y2, a[0]])
        
        nonzero_M_3M_1_df = pd.DataFrame(nonzero_M_3M_1, columns=['Y2', 'alpha'])
        return nonzero_M_3M_1_df
        """
        return np.array(nonzero_M_3M_1, dtype=np.int64)
        """
    return fusion_rules


# --- Tubes Factory ---
def make_tubes_ij(fusion_rules, F1, F2, size):
    @lru_cache(maxsize=None)
    def tubes_ij(N_2:int, N_1:int, M_1:int, M_2:int):  
        """
        Connect the module strands via the fusion strand for all possible allowed matchings ->i.e. computes list of labels of nonzero tube elements
        - i, j specify module objects N_1, M_1, N_2, M_2
        Args:
            F1 (SparseTensor): rank 10 sparse tensor holding module catgeory F-symbol date with expected index ordering
                - ['N_1', 'Y1', 'Y2', 'N_3', 'N_2', 'Y3', 's', 't', 'l', 'j'] (M are module labels, Y are fusion labels and lattin lower case are multiplicity labels)
            
            F2 (SparseTensor): rank 10 sparse tensor holding module catgeory F-symbol date with similar index ordering

            size (Dict):  dictionary containing number of module and fusion elements in mod cat

        Returns:
            nonzero_tubes_ij (pd.DataFrame): List of nonzero tube algebra labels corresponding to (i,j) block
        """
    
        nonzero_N_Y_N = fusion_rules(N_1, N_2)
        nonzero_M_Y_M = fusion_rules(M_1, M_2).rename(columns={"alpha": "beta"})
       
        nonzero_tubes_ij = pd.merge(nonzero_N_Y_N, nonzero_M_Y_M, how="inner", on="Y2")
        nonzero_tubes_ij["linear index"] = nonzero_tubes_ij.index
        nonzero_tubes_ij = nonzero_tubes_ij.set_index(["Y2", "alpha", "beta"])

        return nonzero_tubes_ij

    return tubes_ij

# --- f_ijk_sparse Factory ---
def make_f_ijk_sparse(F: sparse.SparseArray, F_trans: sparse.SparseArray, F_quantum_dims: np.ndarray, size: Dict, tubes_ij):
    @lru_cache(maxsize=None)
    def f_ijk_sparse(i, j, k):
        """
        Args:
            F (sparse.SparseArray): F-symbol for module category
            M_3, M_1 (int): module indices
            size (Dict): size is a dictionary containing the number of fusion objects, the number of module objects and the 

        Returns:
            fusion_rules (np.ndarray): fusion rules for N_M_3Y1_M_1
            nonzero_M_3M_1_df (pd.DataFrame): given an M_3, M_1 gives the Y, alpha indices of nonzero tube elements


        The fusion rules of a fusion category C are embedded in the module category M over a fusion category C.
        We extract them by considering diagrams:

          M_3  1  Y2         M_3  1  Y2
            |  /  /          |  \  /
            | /  /           |   \/k
          í |/  /            |   /
            |  /      =~ F   |  / Y3
            | /              | /
           j|/              l|/
            |                |
           M_1               M_1

        i = k = 0 since multiplicity space one dimensional for identity object of fusion category
        j = l as F is diagonal for this fusion (later I call j and k alpha and beta for some reason)
        
        Find the rank of F to find the dimension of each multipliicty index i.e. each hom space (the fusion rules)
        - Most should be one dimensional
        """
        #print(f"Finding f_ijk for: {i,j,k}")
        # --- Decode i,j,k flattened indices --- 

        M_1, N_1 = index_to_tuple(i, (size['module_label'], size['module_label']))
        M_2, N_2 = index_to_tuple(j, (size['module_label'], size['module_label']))
        M_3, N_3 = index_to_tuple(k, (size['module_label'], size['module_label']))
        
        ##print(f" M_3:{M_3}, M_1:{M_1}, M_2:{M_2}, N_1:{N_1}, N_3:{N_3}, N_2:{N_2} ")
        
        # --- Slice tensors ---
        """
        F_axis_names = ['N_1', 'Y1', 'Y2', 'N_3', 'N_2', 'Y3', 'n_1', 'n_2', 'n_3', 'n_4']  
        F_trans_axis_names = ['M_1', 'Y1', 'Y2', 'M_3', 'M_2', 'Y3', 'm_1', 'm_2', 'm_3', 'm_4']  
        """
      
        #F_slice = remove_zeros(F[N_1, :, :, N_3, N_2, :, :, :, :, :])
        F_slice = slice_tensor(F, {0: N_1, 3: N_3, 4: N_2})
       
        #Fts_slice = remove_zeros(F_trans[M_1, :, :, M_3, M_2, :, :, :, :, :])
        Fts_slice = slice_tensor(F_trans, {0: M_1, 3: M_3, 4: M_2})

        """
        F_slice = F[N_1, :, :, N_3, N_2, :, :, :, :, :]
        Fts_slice = F_trans[M_1, :, :, M_3, M_2, :, :, :, :, :]
        """
        
        # --- Quantum dimension prefactors ---
        sqrtd = np.sqrt(F_quantum_dims[0].astype(np.float64))
        dY1 = sparse.COO.from_numpy(sqrtd)
        dY2 = sparse.COO.from_numpy(sqrtd)
        dY3 = sparse.COO.from_numpy(1.0 / sqrtd)

        # --- Contractions With Einsum ---
        """
        Naming scheme:
        Y1:y, Y2:p, Y3:x
        M_'s sliced out
        Multiplicity indices: rsnamb
        """

        #einsum_str = "y,   p,   x, ypxrsln,ypxambl->ypxrsnamb"
        einsum_str = "y,   p,   x, ypxrsln,ypxamlb->ypxrsnamb"

        dxdxpdy_F_trans_dot_F = sparse.einsum(einsum_str, dY1, dY2, dY3, 
                                              F_slice, Fts_slice, optimize=True)
        
        # --- Remove zeros ---
        tol = 1e-10
        mask = np.abs(dxdxpdy_F_trans_dot_F.data) >= tol
        dxdxpdy_F_trans_dot_F = sparse.COO( dxdxpdy_F_trans_dot_F.coords[:, mask], dxdxpdy_F_trans_dot_F.data[mask], shape=dxdxpdy_F_trans_dot_F.shape )
        
        # --- Reindex using linear index (index the elements in each (i,j) tube subalgebra) ---
        #Y1, Y2, Y3, s, t, j_, i_, k_, q = dxdxpdy_F_trans_dot_F.coords
        Y1, Y2, Y3, n1, n2, n4, m1, m2, m3 = dxdxpdy_F_trans_dot_F.coords
        #print(f"(M_2, M_1, N_1, N_2) = {M_2, M_1, N_1, N_2}")
        idx_a_map = tubes_ij(M_2, M_1, N_1, N_2)
        #print(f"(M_3, M_2, N_2, N_3) = {M_3, M_2, N_2, N_3}")
        idx_b_map = tubes_ij(M_3, M_2, N_2, N_3)
        #print(f"(M_3, M_1, N_1, N_3) = {M_3, M_1, N_1, N_3}")
        idx_c_map = tubes_ij(M_3, M_1, N_1, N_3)
        
        map_a = idx_a_map["linear index"].to_dict()
        map_b = idx_b_map["linear index"].to_dict()
        map_c = idx_c_map["linear index"].to_dict()

        size_a, size_b, size_c = len(map_a), len(map_b), len(map_c)

        index_a = np.array([map_a[tuple] for tuple in zip(Y1, m1, n1)])
        index_b = np.array([map_b[tuple] for tuple in zip(Y2, m2, n2)])
        index_c = np.array([map_c[tuple] for tuple in zip(Y3, m3, n4)])

        reindexed_coords = np.vstack([index_a, index_b, index_c])
        reindexed_shape = (size_a, size_b, size_c)

        reindexed_f_symbol = sparse.COO(reindexed_coords, dxdxpdy_F_trans_dot_F.data, shape=reindexed_shape)
        return reindexed_f_symbol
    return f_ijk_sparse


def make_dim_ijk(size, tubes_ij):
    @lru_cache(maxsize=None)
    def dim_ijk(i,j,k):
        N_Mcat = size["module_label"]
        N_Ncat = size["module_label"]
        M_1, N_1 = index_to_tuple(i, (N_Mcat, N_Ncat))
        M_2, N_2 = index_to_tuple(j, (N_Mcat, N_Ncat))
        map_a = tubes_ij(M_2, M_1, N_1, N_2)#["linear index"].to_dict()
        size_a = len(map_a)
        if size_a==0:
            return None
        M_3, N_3 = index_to_tuple(k, (N_Mcat, N_Ncat))
                
        map_b = tubes_ij(M_3, M_2, N_2, N_3)#["linear index"]#.to_dict()
        size_b = len(map_b)
        if size_b==0:
            return None


        map_c = tubes_ij(M_3, M_1, N_1, N_3)#["linear index"]#.to_dict()
        size_c =  len(map_c)
        if size_c==0:
            return None
        else:
            return (size_a, size_b, size_c)
        
    return dim_ijk

def is_associative(f, tol=1e-9):

    left  = np.einsum('ijk,klm->ijlm', f, f)
    right = np.einsum('jlk,ikm->ijlm', f, f)
    test = left - right
    max_violation = np.max(np.abs(test))
    return (max_violation <= tol), max_violation


def compute_dim_dict(size, tubes_ij):
    N_Mcat = size["module_label"]
    N_Ncat = size["module_label"]

    N_blocks = N_Ncat*N_Mcat
    dim_dict = {}
    for i in range(N_blocks):
        for j in range(N_blocks):
            M_1, N_1 = index_to_tuple(i, (N_Mcat, N_Ncat))
            M_2, N_2 = index_to_tuple(j, (N_Mcat, N_Ncat))
            map_a = tubes_ij(M_2, M_1, N_1, N_2)#["linear index"].to_dict()
            size_a = len(map_a)
            if size_a==0:
                continue

            for k in range(N_blocks):
                
                M_3, N_3 = index_to_tuple(k, (N_Mcat, N_Ncat))
                
                map_b = tubes_ij(M_3, M_2, N_2, N_3)#["linear index"]#.to_dict()
                size_b = len(map_b)
                if size_b==0:
                    continue

                map_c = tubes_ij(M_3, M_1, N_1, N_3)#["linear index"]#.to_dict()
                size_c =  len(map_c)
                if size_c==0:
                    continue
                else:
                    dim_dict[(i,j,k)] = (size_a, size_b, size_c)

    return dim_dict



def construct_irreps(algebra:TubeAlgebra, irrep_projectors_dict:List[Dict], size:Dict, tubes_ij:Callable, q_dim_a, return_df=False, return_rho_list=False):
    """
    Args:
      
    Returns:
    rho[irrep,i,j,a] (2D np.array) --> rho[irrep,i,j,a, s,t]
        
    """

    irrep_coords = []
    irrep_vals = []

    rho_irrep_dict = {}
    N_irrep = len(irrep_projectors_dict)
    for irrep in range(N_irrep):
        first_key = next(iter(irrep_projectors_dict[irrep].keys()))
        k = first_key[1]
        N_blocks = algebra.N_diag_blocks
        rho_ijk_dict = {} 
        for i in range(N_blocks):
            for j in range(N_blocks):
                if (i,j,k) not in algebra.dimension_dict.keys():
                    continue
                if (i, k) not in irrep_projectors_dict[irrep] or (j, k) not in irrep_projectors_dict[irrep]:
                    continue
                
                Q_irrep_ik = irrep_projectors_dict[irrep][(i,k)]
                Q_irrep_jk = irrep_projectors_dict[irrep][(j,k)]

                T_a_ijk = algebra.create_left_ijk_basis(i,j,k).basis
                d_a = len(T_a_ijk)

                M_1, N_1 = index_to_tuple(i, (size['module_label'], size['module_label']))
                M_2, N_2 = index_to_tuple(j, (size['module_label'], size['module_label']))
               
                idx_a_map = tubes_ij(M_2, M_1, N_1, N_2)
                reverse_idx_a_map = {linear_idx: idx 
                                     for idx, linear_idx in idx_a_map["linear index"].items() }
                
                rho_ijk_list = []
                for a in range(d_a):
                    Y, m, n = reverse_idx_a_map[a]
                    rho = 1.0/np.sqrt(q_dim_a[0][Y].astype(np.float64)) * ( Q_irrep_ik.conj().T @ T_a_ijk[a] @ Q_irrep_jk)#.conj().T 
                    rho_ijk_list.append(rho)
                    rows, cols = np.nonzero(rho)
                    new_vals = rho[rows, cols]
                    N_nz = len(new_vals)
    
                    new_coords = np.vstack([np.full(N_nz, irrep),
                                            np.full(N_nz, M_1),
                                            np.full(N_nz, Y),
                                            np.full(N_nz, N_2),
                                            np.full(N_nz, N_1),
                                            np.full(N_nz, M_2),
                                            rows,
                                            np.full(N_nz, n),
                                            np.full(N_nz, m),
                                            cols])

                    irrep_coords.append(new_coords)
                    irrep_vals.append(new_vals)

                rho_ijk_dict[(i,j,k)] = rho_ijk_list
            rho_irrep_dict[irrep] = rho_ijk_dict

    coords = np.hstack(irrep_coords)
    vals = np.hstack(irrep_vals)

    """  Expected Final Shape?????  """
    shape_lazy = np.max(coords, axis = 1) + 1
    F = sparse.COO(coords, vals, shape_lazy)
    return F


def FP_dimension(d_quantum):
    FP_dim = np.sum(d_quantum**2)
    return FP_dim

fusion_cat_quantum_dims = F_quantum_dims[0]

def sparse_clebsch_gordon_coefficients(ω, fusion_cat_quantum_dims):

    FP_dim = FP_dimension(fusion_cat_quantum_dims)
    U_abc = {}
    X_dim = ω.shape[0]
    for a in range(X_dim):
        ω_a = ω[a,::]
        for b in range(X_dim):
            ω_b = ω[b,::]
            P_ab = oe.contract('ayegc hfbd, jycal ibkm -> ayegc hfd jl ikm', ω_a, ω_b )
            for c in range(X_dim):
                P_abc = oe.contract('ayegc hfd jl ikm, jyegl sfkn, y -> ge ac jl hd im sn ',P_ab, ω[c,::].conj(), fusion_cat_quantum_dims)/FP_dim
                trace_P_abc = oe.contract('gg aa jj hh ii ss->', P_abc)
                N_eigs = int(np.abs(np.round(trace_P_abc)))
                if np.abs(N_eigs)> 1e-10:
                    print(f"abc=({a, b, c}) has N_eigs= {N_eigs}")
                    P_abc_permuted = np.transpose(P_abc, (0, 6, 2, 8, 4, 10, 1, 7, 3, 9, 5, 11))
                    P_abc_permuted_old_shape = P_abc_permuted.shape

                    # Can probably precompute these things but lets not for now
                    half = len(P_abc_permuted_old_shape) // 2
                    d_l = np.prod(P_abc_permuted_old_shape[:half])
                    d_r = np.prod(P_abc_permuted_old_shape[half:])

                    P_abc_reshaped = P_abc_permuted.reshape((d_l, d_r))

                    eig_val, eig_abc = eigs(P_abc_reshaped, k=N_eigs)
                    eig_abc_rehaped = eig_abc.reshape(P_abc_permuted_old_shape[:half] + (eig_abc.shape[1],))

                    U_abc[(a,b,c)] = eig_abc_rehaped
                    
                    """not_herm = len(remove_zeros(P_abc_reshaped.conj().T - P_abc_reshaped).data)>0
                    if not_herm == True:
                        pass
                        #print(remove_zeros(P_abc_reshaped.conj().T - P_abc_reshaped).data)
                    print(f"Is P_abc hermitian ? {not not_herm}")
                    test_proj = len(remove_zeros(P_abc_reshaped @ P_abc_reshaped - P_abc_reshaped).data)>0
                    print(f"Is P_abc a projector? {not test_proj}")"""

    return U_abc

U_abc = sparse_clebsch_gordon_coefficients(ω_fixed, fusion_cat_quantum_dims)