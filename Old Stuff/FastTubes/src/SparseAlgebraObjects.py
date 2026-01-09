# Standard libraries
import os
import math
from typing import List, Tuple, Callable
from functools import lru_cache

# Third-party
import numpy as np
import pandas as pd
from scipy.linalg import block_diag



# --- Vector Definition ---

class Vec:
    """
    A class to represent an vector with attribute: irrep block label, subalgebra label, vector
    """
    def __init__(self, vector, subalgebra=None, irrep_index=None, alpha=None):
        """
        Args:
            vector (np.array(complex_128)): eigenvector in k irrep of ij subalgebra
            subalgebra (tuple[int]): (i, j) subalgebra index of full algebra
            irrep_index (int): irrep-k of the (i, j) subalgebra

        """
        self.vector = vector # / np.linalg.norm(vector)
        self.subalgebra = subalgebra  # (i, j) subalgebra
        self.irrep_index = irrep_index            # irrep-k of the (i, j) subalgebra
        self.alpha = alpha                        # # Block label (irreducible representation) - Is this just the k index? might be the case if blocks dont mix between basis elements in subalg
    
    def __add__(self, other):
        if isinstance(other, Vec):
            if self.subalgebra == other.vector:
                result_vector = self.vector + other.vector
                result = Vec(result_vector, self.vector)
            else:
                result = AlgebraVec({self.subalgebra: self.vector, other.subalgebra: other.vector})
            return result 
        
        if isinstance(other, AlgebraVec):
            if self.subalgebra in other.vecs:
                other.vecs[self.subalgebra] = other.vecs[self.subalgebra] + self
            else:
                other.vecs[self.subalgebra] = self

        else:
            return NotImplemented
  

    def to_numpy_in_algebra_repspace(self, rep_structure):
        """
        Expand vector to full Hilbert space based on (i, j, k) structure from rep_structure.
        Assumes vector lives in space of shape (d_b^k, ) and expands to the full (d_b_total,) = sum_k (d_b^k, ) by placing 
        the Vec.vector in the Vec.irrep_index block.

        Args:
            rep_structure (grouped pandas series): rep_structure[(i, j)] gives a pandas Series of dimensions columns (d_b, d_c) indexed by irrep label k

        Returns:
            full_vec (np.array[complex_128]):eigenvector in ful rep space
        """

        vec_subalg = self.to_numpy_in_subalgebra_repspace(rep_structure)

        blocks = []
        for i in sorted(set(i for (i, j, _) in rep_structure.keys() if i == j)): # Loop over the A_ii subalgebras
            if i == self.subalgebra[0]:
                blocks.append(vec_subalg)
            else:
                sub_dims = rep_structure[(i, i)]
                total_dim = sum(dim_c for _, dim_c in sub_dims.to_numpy()) # finds total dimension of the subalgebra rep space
                blocks.append(np.zeros(total_dim))
        return np.concatenate(blocks)
    
    def to_numpy_in_subalgebra_repspace(self, rep_structure: pd.Series):
            """
            Converts a vector in irrep space of A_ij: V_{ijk} into the full subalgebra rep. space.: V_ij

            Args:
                vec (SubAlgebraVec): vectroe in the irrep k of the ij subalgebra rep space

                rep_space_structure (PandasDataFrame) = {
                                                            (i, j): {
                                                                k: (dim_b, dim_c)
                                                                for k in irreps_present
                                                            }
                                                            for i, j in index_pairs
                                                        }

            Returns:
                v_subalg (np.array[complex_128]): full subalgebra repspace vectir as numpy array
            """
            rep_structure_ij = rep_structure[self.subalgebra]
            blocks = []
            for k, (dim_b, dim_c) in rep_structure_ij.items():
                #print(k, self.irrep_index)
                if k == self.irrep_index:
                    blocks.append(self.vector)
                else:
                    blocks.append(np.zeros(dim_b, dtype=np.complex128))
            return np.concatenate(blocks)


    def __repr__(self):
        return f"Vec(dim={np.shape(self.vector)}, subalgebra={self.subalgebra}, irrep={self.irrep_index})"
    
    @classmethod
    def zero(cls, dim: int, irrep_index: int, subalgebra: tuple):
        """
        Create a zero vector in the given representation space.

        Args:
            dim (int): Dimension of the vector space
            irrep_index (int): Irrep label
            subalgebra (tuple): (i, j) index of the subalgebra

        Returns:
            Vec: a zero-valued vector in the appropriate rep space
        """
        return cls(vector=np.zeros(dim, dtype=np.complex128),
                   irrep_index=irrep_index,
                   subalgebra=subalgebra)
    
# --- Eigen Vector Definition ---

class EigVec(Vec):
    """
    A class to represent an eigenvector with attribute: irrep block label, eigenval label, degeneracy label, eigenvalue, subalgebra label
    EigVec is a subclass of Vec and represnt eigenvetros of random elemennt in the irrepspace of subalgebras
    """
    def __init__(self, vector, eigenvalue, eigindex=None, degenindex=None, subalgebra=None, irrep_index=None, alpha=None):
        super().__init__(vector, subalgebra=subalgebra, irrep_index=irrep_index, alpha=alpha)
        self.eigenvalue = eigenvalue             # nmerical value of eigenvalue
        self.eigindex = eigindex                 # Unique eigenvalue label within a block
        self.degenindex = degenindex             # Degeneracy label (for vectors sharing the same eigenvalue)

    def __repr__(self):
        return (f"EigVec(eigenvalue={self.eigenvalue:.4f}, dim_irrep_space={np.shape(self.vector)} "
                f"deg={self.degenindex}, subalg={self.subalgebra}, irrep={self.irrep_index})")
    

class SubAlgebraVec:
    """
    Represents a collection of Vecs (one per irrep index k) in a single (i, j) subalgebra space.
    """

    def __init__(self, vecs: dict[int, Vec]):
        """
        Args:
            vecs (dict[int, Vec]): keys are irrep indices k, values are Vecs in V_{ijk}
        """
        self.vecs = vecs

        # Sanity check: all vecs should belong to the same (i, j)
        subalgebras = {vec.subalgebra for vec in vecs.values()}
        if len(subalgebras) != 1:
            raise ValueError(f"Inconsistent subalgebra indices: {subalgebras}")
        self.subalgebra = next(iter(subalgebras))

    def __add__(self, other):
        if self.subalgebra != other.subalgebra:
            raise ValueError("Can't add SubAlgebraVecs from different subalgebras")

        new_vecs = {}
        for k, v1 in self.vecs.items():
            v2 = other.vecs.get(k)
            if v2:
                new_vecs[k] = Vec(v1.vector + v2.vector, subalgebra=self.subalgebra, irrep_index=k)
            else:
                new_vecs[k] = v1

        for k, v2 in other.vecs.items():
            if k not in new_vecs:
                new_vecs[k] = v2

        return SubAlgebraVec(new_vecs)

    def to_numpy(self, rep_structure):
        """
        Expands the SubAlgebraVec to a full vector in the V_ij representation space.
        """
        blocks = []
        rep_structure_ij = rep_structure[self.subalgebra]
        for k, (dim_b, _) in rep_structure_ij.items():
            vec = self.vecs.get(k)
            if vec:
                blocks.append(vec.vector)
            else:
                blocks.append(np.zeros(dim_b, dtype=np.complex128))
        return np.concatenate(blocks)

    def __repr__(self):
        irreps = list(self.vecs.keys())
        return f"SubAlgebraVec(subalgebra={self.subalgebra}, irreps={irreps})"

class AlgebraVec:
    def __init__(self, vecs: dict[tuple[int, int], Vec]):
        self.vecs = vecs  # Dictionary of (i,j): Vec

    def __add__(self, other):
        if not isinstance(other, AlgebraVec):
            return NotImplemented

        result = self.vecs.copy()

        for (i,j), vec in other.vecs.items():
            if (i,j) in result:
                result[(i,j)] = result[(i,j)] + vec  # Vec addition
            else:
                result[(i,j)] = vec

        return AlgebraVec(result)

    def __repr__(self):
        return f"AlgebraVec({self.vecs})"

    @classmethod
    def zero(cls, index_set: list[tuple[int, int]], dim_lookup: Callable[[tuple[int, int]], int], 
             irrep_lookup: Callable[[tuple[int, int]], int]) -> "AlgebraVec":
        """
        Creates a zero AlgebraVec with keys from `index_set`.
        `dim_lookup((i, j))` -> dimension of vector at (i,j)
        `irrep_lookup((i, j))` -> irrep index of vector at (i,j)
        """
        zero_vecs = {}
        for ij in index_set:
            dim = dim_lookup(ij)
            irrep = irrep_lookup(ij)
            zero_vecs[ij] = Vec.zero(dim=dim, irrep_index=irrep, subalgebra_index=ij)
        return cls(zero_vecs)


# --- Sub-Algebra Irrep Base Class Definition ---

class SubalgebraIrrepBase:
    """
    SubalgebraIrrep: Represents all basis matrices of a single irrep (i,j,k).
    """
    rep_type: str = "Base"  # Default

    def __init__(self, i:int, j:int, k:int, T_aijk_list:List[np.ndarray]):
        """
        i, j: integers specifying the subalgebra (i,j) of the tube algebra
        k: integer specifying the irrep index of the subalgebra
        T_aijk_list: list of numpy arrays corresponding to the irrep-k blocks of each basis element a
        """
        self.subalgebra = (i, j)
        self.irrep = k
        self.basis = T_aijk_list  # List of numpy arrays (one matrix for each basis element in irrep k of the (i,j) subalgebra)
        self.d_a = len(T_aijk_list)
        self.irrep_space_shape = tuple(int(x) for x in np.shape(T_aijk_list[0]) )  # dimension of the representation of the irrep of subalgebra

    def __repr__(self):
        return f"<SubalgebraIrrep(subalgebra={self.subalgebra}, irrep={self.irrep}, basis_dim={len(self.basis)}, irrep_space_dim={self.irrep_space_shape}, rep_type={self.rep_type})>"

    def linear_combination(self, coeffs):
        """Returns a linear combination of the basis matrices"""
        return np.tensordot(coeffs, self.basis, axes = ([0], [0]))

# --- Sub-Algebra Irrep Left And Right Class Definition ---

class SubalgebraIrrepL(SubalgebraIrrepBase):
    """
    Left regular representation
    """
    rep_type: str = "L" 
    

class SubalgebraIrrepR(SubalgebraIrrepBase):
    """
    Right regular representation
    """
    rep_type: str = "R" 

# --- Sub-Algebra Block-k Element Base Class Definition ---

class SubAlgebraElementBlockBase:
    """
    SubAlgebraElementBlockBase: represnet an eleemnt in a block of a subalgebra of the tube algebra
    """
    rep_type: str = "Base"  

    def __init__(self, i: int, j:int, k:int, LX_ijk:np.ndarray):
        self.subalgebra = (i, j)
        self.irrep = k
        self.LX = LX_ijk  
        
    def __repr__(self):
        return (f"<SubAlgebraElementBlockBase (i={self.subalgebra[0]}, j={self.subalgebra[1]}, "
                f"k={self.irrep}) | shape={self.LX.shape}, rep_type={self.rep_type}>")

    def copy(self):
        return self.__class__(self.subalgebra[0], self.subalgebra[1],
                              self.irrep, self.LX.copy())

    def to_matrix(self):
        return self.LX
    
    def __add__(self, other):
        if not isinstance(other, SubAlgebraElementBlockBase):
            return NotImplemented

        if self.subalgebra != other.subalgebra:
            raise ValueError(f"Cannot add: different subalgebras {self.subalgebra} vs {other.subalgebra}")

        if self.irrep != other.irrep:
            raise ValueError(f"Cannot add: different irreps {self.irrep} vs {other.irrep}")

        if self.LX.shape != other.LX.shape:
            raise ValueError(f"Shape mismatch: {self.LX.shape} vs {other.LX.shape}")

        LX_sum = self.LX + other.LX
        return self.__class__(self.subalgebra[0], self.subalgebra[1], self.irrep, LX_sum)
    
    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)

    def dagger(self):
        """
        Returns the Hermitian conjugate of this subalgebra block element.
        i.e Swaps indices (i, j) → (j, i) and conjugate-transposes the matrix
        """
        i, j = self.subalgebra
        LX_dag = self.LX.conj().T  

        return self.__class__(j, i, self.irrep, LX_dag)
    
    def conj(self):
        """
        Returns the  conjugate of this subalgebra block element.
        i.e Swaps indices (i, j) → (j, i) and conjugate-transposes the matrix
        """
        i, j = self.subalgebra
        LX_conj = self.LX.conj()

        return self.__class__(i, j, self.irrep, LX_conj)
    
# --- Sub-Algebra Block-k Element Left And Right Class Definition ---
    
class SubAlgebraElementBlockL(SubAlgebraElementBlockBase):
    rep_type: str = "L"

    def __matmul__(self, other):
        if isinstance(other, Vec):
            if self.subalgebra[1] != other.subalgebra[0]:
                print(f"Cannot act: L_({self.subalgebra}) on Vec in subalgebra ({other.subalgebra})")
                print(f"Irrep mismatch: {self.irrep} vs {other.irrep_index}")
                return Vec.zero(dim=self.LX.shape[0], irrep_index=self.irrep, subalgebra=(self.subalgebra[0], other.subalgebra[1]) )
                
            new_vector = self.LX @ other.vector
            return Vec(vector=new_vector, irrep_index=other.subalgebra[1],
                       subalgebra=(self.subalgebra[0], other.subalgebra[1]))

        return super().__matmul__(other)
    
class SubAlgebraElementBlockR(SubAlgebraElementBlockBase):
    rep_type: str = "R"

    def __matmul__(self, other):
        if isinstance(other, Vec):
            if self.subalgebra[1] != other.subalgebra[1]:# or self.irrep != other.irrep_index:
                print(f"Cannot act:R_({self.subalgebra}) on Vec in subalgebra ({other.subalgebra}) ")
                return Vec.zero(dim=self.LX.shape[0], irrep_index=self.irrep, subalgebra=(other.subalgebra[0], self.subalgebra[0]) )
            
            new_vector = other.vector @ self.LX

            return Vec(vector=new_vector, irrep_index=other.subalgebra[0],
                       subalgebra=(other.subalgebra[0], self.subalgebra[0]))

        return super().__matmul__(other)
    
# --- Tube Algebra Class Definition ---

class TubeAlgebra:
    """ 
    TubeAlgebra: Stores all irreps, provides subalgebra access and algebraic operations
    """
    def __init__(self, dimension_dict:dict, f_ijk_sparse:Callable[[int, int, int], np.ndarray], df_fabc:pd.DataFrame=None):
        """
        A class to represent the tube algebra. Given the nonzero structure constants, splits into subalgebra....

        Build the TubeAlgebra from a dataframe and a dimension dict.
        dimension_dict[(i,j,k)] = (d_a, d_b, d_c)

        Args:

        Returns:

        """
        self.f_ijk_sparse = f_ijk_sparse
        self.dimension_dict = dimension_dict
        self.N_diag_blocks = max(k[0] for k in dimension_dict.keys()) +1
        self.d_algebra_squared = self._find_dim_algebra()
            
        if df_fabc != None:
            self.df_fabc = df_fabc
            self.ij_subalgebras_L = self._find_ij_subalgebras_L()
            self.jk_subalgebras_R = self._find_jk_subalgebras_R()
            self.left_irrep_df = self._build_left_irrep_df(df_fabc)
            self.right_irrep_df = self._build_right_irrep_df(df_fabc)
            self.d_algebra_squared = self._find_dim_algebra()

    def _find_dim_algebra(self):
        subalgebra_shapes = np.array([shape[1] for (i,j,k), shape in self.dimension_dict.items() if i==j ])
        d_algebra_squared = np.sum(subalgebra_shapes, axis = 0)
        return d_algebra_squared

    def _find_ij_subalgebras_L(self):
        """
        List all the non-zero dimensional (i, j) subalgebras given the dictionary of all subalgebra dimension in self.dimension_dict

        Args:
            dimension_dict (dict): Dictionary where keys are tuples (i, j, k) and  values are tuples (d_a, d_b, d_c), representing 
                                   the size of the basis and dimension of the irrep vector space for subalgebra (i, j)
        Returns:
            ii_subalgebras (list of tuples): List of unique (i, j) subalgebras where the dimension d_a > 0.
        """
        ij_subalgebras_L = set() # Using set avoids duplicates (the irony)

        for (i, j, k), (d_a, d_b, d_c) in self.dimension_dict.items():
            if d_a > 0:
                ij_subalgebras_L.add((int(i), int(j)))

        return sorted(ij_subalgebras_L)
    
    def _find_jk_subalgebras_R(self):
        """
        List all the non-zero dimensional (i, j) subalgebras given the dictionary of all subalgebra dimension in self.dimension_dict

        Args:
            dimension_dict (dict): Dictionary where keys are tuples (i, j, k) and  values are tuples (d_a, d_b, d_c), representing 
                                   the size of the basis and dimension of the irrep vector space for subalgebra (i, j)
        Returns:
            ii_subalgebras (list of tuples): List of unique (i, j) subalgebras where the dimension d_a > 0.
        """
        jk_subalgebras_R = set() # Using set avoids duplicates (the irony)

        for (i, j, k), (d_a, d_b, d_c) in self.dimension_dict.items():
            if d_b > 0:
                jk_subalgebras_R.add((int(j), int(k)))

        return sorted(jk_subalgebras_R)

    @lru_cache(maxsize=None)
    def create_left_ijk_basis(self, i:int, j:int, k:int):
        d_a, d_b, d_c = self.dimension_dict[(i, j, k)]
        f_ijk_dense = self.f_ijk_sparse(i,j,k).todense() 
        T_a_ijk = [ f_ijk_dense[a, :, :].T for a in range(d_a)]
        
        """
        for a in range(d_a):
            mat = np.zeros((d_c, d_b), dtype=np.complex64)
            mat[df_a["c"].to_numpy(), df_a["b"].to_numpy()] = df_a["value"].to_numpy()
            pass
        """
        ijk_irrep_basis = SubalgebraIrrepL(int(i), int(j), int(k), T_a_ijk)

        return ijk_irrep_basis
    
    @lru_cache(maxsize=None)
    def create_right_ijk_basis(self, i:int, j:int, k:int):
        """
        Note that i, j ,k subalgebra indices are flipped to j,k,i since we technically are working with the tranpsose right regular representation
        """
        d_a, d_b, d_c = self.dimension_dict[(i, j, k)]

        f_ijk_dense = self.f_ijk_sparse(i,j,k).todense() 

        T_b_ijk = [ f_ijk_dense[:, b, :] for b in range(d_b)]
        
        ijk_irrep_basis = SubalgebraIrrepR(int(i), int(j), int(k), T_b_ijk)
       
        return ijk_irrep_basis

    def get_left_irrep(self, i:int, j:int, k:int)->SubalgebraIrrepL:
        return self.left_irrep_df.loc[(i, j, k), "irrep"]

    def get_right_irrep(self, j:int, k:int, i:int)->SubalgebraIrrepR:
        return self.right_irrep_df.loc[(j, k, i,), "irrep"]
    
    ### Random elements across the k block of the (i,j) subalgebra in left (right) regular representation

    def random_left_linear_combination_ijk(self, i: int, j: int,  k: int, isHermitian=True, rng :np.random._generator.Generator = None) -> SubAlgebraElementBlockL:
        """
        Constructs a random left representation element L_ij^k from the basis.

        Args:
            i, j (int): Subalgebra indices
            k (int): Irrep index
            isHermitian (bool): Whether to return a Hermitian combination (only valid if i == j)
            rng (np.random.Generator): Optional random number generator

        Returns:
            SubAlgebraElementBlockR: Random linear combination in the left regular representation
        """
        #block_basis_ijk = self.get_subalgebra_L(i, j)["irrep"][k]
        block_basis_ijk = self.create_left_ijk_basis(i,j,k)

        if rng is None:
            rng = np.random.default_rng()

        d_a = self.dimension_dict[(i,j,k)][0]
        x_a = rng.uniform(-1, 1, size=d_a)
        #print(x_a)
        LX_ijk = block_basis_ijk.linear_combination(x_a)
        LX_ijk = SubAlgebraElementBlockL(i, j, k, LX_ijk)
        
        if isHermitian==True and i==j: # Only need hermitian element for eigenvector calculation
            LX_ijk = LX_ijk + LX_ijk.conj() #.dagger()
        
        return LX_ijk
    
    def random_right_linear_combination_ijk(self, i: int, j: int, k: int, isHermitian=True, rng: np.random.Generator = None) -> SubAlgebraElementBlockR:
        """
        Constructs a random right representation element R_ij^k from the basis.

        Args:
            i, j (int): Subalgebra indices
            k (int): Irrep index
            isHermitian (bool): Whether to return a Hermitian combination (only valid if i == j)
            rng (np.random.Generator): Optional random number generator

        Returns:
            SubAlgebraElementBlockR: Random linear combination in the right regular representation
        """
        #block_basis_ijk = self.get_subalgebra_R(i, j)["irrep"][k]
        block_basis_ijk = self.create_right_ijk_basis(i, j, k)

        if rng is None:
            rng = np.random.default_rng()

        d_a, d_b, d_c = self.dimension_dict[(i, j, k)]

        x_b = rng.uniform(-1, 1, size=d_c)
        
        RX_ijk = np.tensordot(x_b, block_basis_ijk.basis, axes = ([0], [1]))
        RX_ijk = SubAlgebraElementBlockR(i, j, k, RX_ijk)

        if isHermitian and i == j:
            RX_ijk = RX_ijk + RX_ijk.dagger()

        return RX_ijk

    