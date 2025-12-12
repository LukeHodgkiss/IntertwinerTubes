import numpy as np

# --- Eigen Decomposition Functions ---

def eig(A: np.ndarray):
    """
    Checks if matrix is 1 dimenionsla before performing ED, if it is then we can skip that irrep as the ED is trivial else we do a full ED. 

    eigenvalues (…, M) array: The eigenvalues, each repeated according to its multiplicity. The eigenvalues are not necessarily ordered. 
                              The resulting array will be of complex type, unless the imaginary part is zero in which case it will be cast to a real type. 
                              When a is real the resulting eigenvalues will be real (0 imaginary part) or occur in conjugate pairs
    eigenvectors (…, M, M) array: The normalized (unit “length”) eigenvectors, such that the column eigenvectors[:,i] is the eigenvector corresponding to the eigenvalue eigenvalues[i].
    
    Args:
        A (np.array): matrix

    Returns:
        (eigenvalues, eigenvectors): as a named tuple
    
    """

    shape = np.shape(A)
    if shape[0] == 1 and shape[-1] == 1:
        return A[0][0], np.array(np.array([1.0])) # (eigenvalue, eigenvector)

    else:
        return np.linalg.eig(A)
    
def eigen_decomposition_subalgebra_block(algebra: TubeAlgebra, i: int, rng :np.random._generator.Generator = None):
    """
    Finds the eigendecompostion of the i'th block of the (i,i) subalgebra of the tube algebra

    Args:
        algebra (TubeAlgebra):
        i (int):

    Returns:
        tuple(EigVec): 
    """
    
    if rng is None:
            rng = np.random.default_rng()  # independent generator
    
    LX_iii = algebra.random_left_linear_combination_ijk(i, i, i, rng)
    eigenvalues_iii, eigenvectors_iii  = np.linalg.eig(LX_iii.LX)
    
    eig_results_iii = [ EigVec(vector=v.flatten(), eigenvalue=val, subalgebra=(i,i), irrep_index=i) for v, val in zip(eigenvectors_iii.T, eigenvalues_iii) ]
    return eig_results_iii

def eigen_decomposition_subalgebra(algebra: TubeAlgebra, i: int, rng :np.random._generator.Generator = None):
    """
    Finds the eigendecompostion of the i'th block of the(i,i) subalgebra of the tube algebra

    Args:
        algebra (TubeAlgebra):
        i (int):

    Returns:
        irreps_ii (list(int)): labels the coresponding irrep each Eigen decomp belongs to
        list(eigenvalues, eigenvectors): eigenvalues (np.array) & eigenvectors (np.array) such that the column eigenvectors[:,i] is the eigenvector corresponding to the eigenvalue eigenvalues[i]
    """
    
    if rng is None:
            rng = np.random.default_rng()  # independent generator
    
    X_ii = algebra.random_left_linear_combination_ij(i, i, rng )

    irreps_ii = X_ii.irreps
    eig_results_ii = []     # list of lists [..., [eigenvalues, eigenvectors], ...]
    for LX_iik in  X_ii.LX:
        eigenvalues, eigenvectors = eig(LX_iik)
        eig_results_ii.append([eigenvalues, eigenvectors])
    
    return irreps_ii, eig_results_ii

# --- Constructing Irreps Functions ---

def build_block(v:EigVec, L_X: Callable[[int, int, int], SubAlgebraElementBlockL], i, rng, tol_rank=1e-10)->np.ndarray:
    """
    Builds out block in the subalgebra (i,j) from a starting eigenvector by repeate dmultilucation of randommelements L_X.

    Args:
        v (EigVec): 
        L_X (AlgebraElementL): Algebra element used to generate new vectors via L_X @ v
        i (int):
        tol_rank (float): Numerical threshold for rank to be considered zero

    Returns:
        Q_old (np.ndarray):
    """

    # Move from (i,i)_i subalgebra to (j,i)_i subalgebra
    v_list = [(L_X(i, v.subalgebra[0], v.subalgebra[1] ) @ v).vector]
    Q_old, R = np.linalg.qr(np.stack(v_list))
    rank_old = 1
    rank_new = 1
    
    while True:
        # Find new candidate vector
        L = L_X(i, v.subalgebra[0],  v.subalgebra[1], rng=rng)
        new_vec = L @ v  
        # Add candidate vector to list and test rank
        M_new = np.stack(v_list + [new_vec.vector], axis=1)
        Q_new, R = np.linalg.qr(M_new)
        rank_new = np.sum(np.abs(np.diag(R)) > tol_rank)

        if rank_new > rank_old:
            v_list.append(new_vec.vector)
            rank_old = rank_new
            Q_old = Q_new
           
        else:
            break

    return Q_old


def remove_overlapping_evec(algebra:TubeAlgebra, ED_ii:List[EigVec], ED:List[EigVec], tol_overlap = 1e-9)->List[EigVec]:
    """
    Given a set of eigenvectors ED_ii remove any eigenvectros from this set that have an overlap with any vector in ED

    Args:
        ED_ii (List[EigVec]): list of vectors to trim
        ED (List[EigVec]): list of vectors to compare against

    Returns:
        ED_ii_trimmed 
    
    """
    ED_ii_trimmed = []

    for evec in ED_ii:
        overlaps = 0
        s,t = evec.subalgebra 
        for old_evec in ED:
            i,j = old_evec.subalgebra
            if (j, t, s) in algebra.dimension_dict and (s, i, j)  in algebra.dimension_dict:
                
                LX_sij = algebra.random_left_linear_combination_ijk(s,i,j)
                RX_jts = algebra.random_right_linear_combination_ijk(j,t,s)
                
                vl = RX_jts @ evec
                vr = LX_sij @ old_evec
                overlaps = inner_product(vl, vr)
                
            else:
                overlaps = 0

            if np.abs(overlaps) > tol_overlap:
                break

        if np.abs(overlaps) < tol_overlap: 
            ED_ii_trimmed.append(evec)

    return ED_ii_trimmed

def remove_evec_in_same_block_ii(ED_ii:List[EigVec], RX_iii:SubAlgebraElementBlockR, LX_iii:SubAlgebraElementBlockL, tol_overlap = 1e-12)->List[EigVec]:
    """

    Args:
        ED_ii (List[EigVec]): list of vectors to trim

    Returns:

    """
    ED_ii_trimmed = []

    for evec_1 in ED_ii:
        # check if this vector is non-overlapping with all previously selected
        if all(np.abs(inner_product(RX_iii @ evec_1, LX_iii @ evec_2)) < tol_overlap for evec_2 in ED_ii_trimmed):
            ED_ii_trimmed.append(evec_1)

    return ED_ii_trimmed

def build_out_irrep( v: EigVec, i: int, algebra: TubeAlgebra, rng):# -> Dict[Tuple[int]:np.ndarray]:
    """
    Starting from a vector in subalgebra (i, i), builds out the irrep by applying 
    left multiplication from related subalgebra blocks (j, i)

    # --- Rough Method Outline --- #

        Loop over each block j
            Given a vec v_ii in (i,i) apply LX_ii on v until space built out -> Q_ii
            Next apply LX_{i+1}i on v_ii until space built out -> Q_{i+1}i
            ...
            Next apply LX_ji on v_ii until space built out -> Q_ji  
            ...
            Finally apply LX_ji on v_ii until space built out -> Q_ji
        Stop here or earlier if the dimension of the algebra has been saturated

    # ---------------------------- #
        
    Args:
        v (EigVec): Initial eigenvector in block (i, i)
        i (int): Fixed right index of subalgebra blocks
        algebra (TubeAlgebra): Algebra object containing subalgebra elements

    Returns:
        List[np.ndarray]: List of matrices forming the irrep space built out from eigenvector v
    """
    irrep_blocks = {}
    dim_iii = algebra.dimension_dict[(i, i, i)][0]

    for j in range(algebra.N_diag_blocks): 
        if (j, v.subalgebra[0], v.subalgebra[1]) in algebra.dimension_dict:
            Q_v = build_block(v, algebra.random_left_linear_combination_ijk, j, rng=rng)

            irrep_blocks[j, v.subalgebra[1]] = Q_v
            

    return irrep_blocks


def d_sum(d_irrep_list):
    return np.sum(np.array(d_irrep_list)**2)

def find_idempotents(algebra: TubeAlgebra): #-> Dict[Dict[np.ndarray]]:
    """
    Finds a set of orthogonal idempotents by:
    - Decomposing each (i, i) block into eigenvectors
    - Removing overlapping eigenvectors
    - Grouping by eigenvalue
    - Building out full irreps from non-overlapping eigenvectors

    Args:
        algebra (TubeAlgebra): The full algebra object

    Returns:
        List[Dict[Tuple[Int, int]:np.ndarray]]: A list of irreducible representation blocks (idempotents)
    """

    irrep_projector_list = []
    d_irrep_list = []
    ED_global = []
    d_algebra_squared = algebra.d_algebra_squared

    rng = np.random.default_rng(seed=42)  

    for ii in range(algebra.N_diag_blocks):  # Loop over diagonal subalgebras (i = j)

        # Step 1: Diagonalize subalgebra block (ii, ii)
        ED_ii = eigen_decomposition_subalgebra_block(algebra, ii)  

        # Step 2: Remove overlapping eigenvectors (between different subalgebra ED)
        ED_ii_ortho = remove_overlapping_evec(algebra, ED_ii, ED_global)

        # Step 3: Remove overlapping eigenvectors (within subalgebra ED)
        RX_iii = algebra.random_right_linear_combination_ijk(ii,ii,ii,rng=rng) 
        LX_iii = algebra.random_left_linear_combination_ijk(ii,ii, ii,rng=rng)
        ED_ii_ortho_trimmed = remove_evec_in_same_block_ii(ED_ii_ortho, RX_iii, LX_iii)
        ED_global.extend(ED_ii_ortho_trimmed)

        # Step 5: Build irreps from each unique eigenvalue
        
        d_irrep_list = []
        for vec in ED_ii_ortho_trimmed:
            
            irrep = build_out_irrep(vec, ii, algebra, rng=rng)
            d_irrep = sum([Q_ij.shape[1] for Q_ij in irrep.values()])
            irrep_projector_list.append(irrep)
            d_irrep_list.append(d_irrep)
            
            # Track dimension to terminate early
            d_irrep = sum([Q_ij.shape[1] for Q_ij in irrep.values()])
            d_irrep_list.append(d_irrep)
            if d_sum(d_irrep_list) >= d_algebra_squared:
                return irrep_projector_list

    return irrep_projector_list
