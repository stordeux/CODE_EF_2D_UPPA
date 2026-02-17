import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

class COOMatrix:
    def __init__(self,nb_lig,nb_col,nnz):
        self.rows = np.zeros(nnz, dtype=np.int64)
        self.cols = np.zeros(nnz, dtype=np.int64)
        self.data = np.zeros(nnz,dtype=np.complex128)
        self.nb_lig=nb_lig
        self.nb_col=nb_col
        self.l=0
        self.nnz=nnz
    
    def ajout(self,i,j,val):
        if self.l >= self.nnz:
            raise ValueError("Nombre d'éléments non nuls dépassé")
        self.rows[self.l] = i
        self.cols[self.l] = j
        self.data[self.l] = np.complex128(val)
        self.l=self.l+1
    def ajout_rapide(self,row,col,nnz_val,val):
        if self.l + nnz_val > self.nnz:
            raise ValueError("Nombre d'éléments non nuls dépassé")
        self.rows[self.l:self.l+nnz_val] = row
        self.cols[self.l:self.l+nnz_val] = col
        self.data[self.l:self.l+nnz_val] = val
        self.l += nnz_val
    def print(self):
        print('nb lignes',self.nb_lig)
        print('nb colonnes',self.nb_col)
        self.create_COO()
        print(self.coo)
        print(self.to_dense())
        
    def create_COO(self):
        self.coo = coo_matrix((self.data[:self.l], (self.rows[:self.l], self.cols[:self.l])), 
                              shape=(self.nb_lig, self.nb_col))
    def to_dense(self):
        A=np.zeros((self.nb_lig,self.nb_col),dtype=np.complex128)
        for i in range(self.l):
            row=self.rows[i].astype(int)
            col=self.cols[i].astype(int)
            dat=self.data[i]
            A[row,col]+=dat
        return(A)
    def to_csc(self):
        self.create_COO()
        self.csc=self.coo.tocsc()
    def to_csr(self):
        self.create_COO()
        self.csr=self.coo.tocsr()
    def lu(self):
        self.to_csc()
        self.lu_decomp = splu(self.csc)
    def solveLU(self, F):
        F = np.asarray(F).reshape(-1)
        x = self.lu_decomp.solve(F)
        return x #.reshape((self.nb_lig, 1))    
    def solve(self, F):
        self.to_csc()
        self.lu_decomp = splu(self.csc)
        F = np.asarray(F).reshape(-1)
        x = self.lu_decomp.solve(F)
        return x #.reshape((self.nb_lig, 1))    


    def __add__(self, other):
        """
        Ajoute à self la matrice other

        Parameters:
        -----------
        other : COOMatrix
            Autre matrice COO à additionner
            
        
        """
        if self.nb_lig != other.nb_lig or self.nb_col != other.nb_col:
            raise ValueError("Les matrices doivent avoir les mêmes dimensions")
        
                
        # Ajouter les entrées de la deuxième matrice
        for i in range(other.l):
            self.ajout(other.rows[i], other.cols[i], other.data[i])
        return self

    def __sub__(self, other):
        """
        Soustrait la matrice other de self

        Parameters:
        -----------
        other : COOMatrix
            Autre matrice COO à soustraire
            
        
        """
        if self.nb_lig != other.nb_lig or self.nb_col != other.nb_col:
            raise ValueError("Les matrices doivent avoir les mêmes dimensions")
        
                
        # Ajouter les entrées de la deuxième matrice avec un signe négatif
        for i in range(other.l):
            self.ajout(other.rows[i], other.cols[i], -other.data[i]) 
        return self       
    
    def sesquilinear_form(self, V, U):
        """
        Calcule la forme sesquilinéaire V^* A U
        où V^* est le conjugué transposé de V
        
        Parameters:
        -----------
        V : array (n,) ou (n,1)
            Premier vecteur
        U : array (n,) ou (n,1)
            Second vecteur
            
        Returns:
        --------
        result : complex ou float
            Valeur de la forme sesquilinéaire V^* A U
        """
        # Assurer que V et U sont des vecteurs 1D
        V = np.asarray(V).flatten()
        U = np.asarray(U).flatten()
        
        # Vérifier les dimensions
        if len(V) != self.nb_lig:
            raise ValueError(f"V doit avoir {self.nb_lig} éléments, mais a {len(V)}")
        if len(U) != self.nb_col:
            raise ValueError(f"U doit avoir {self.nb_col} éléments, mais a {len(U)}")
        
        # Calculer V^* A U en parcourant les éléments non-nuls de A
        result = 0.0 + 0.0j
        for idx in range(self.l):
            i = int(self.rows[idx])
            j = int(self.cols[idx])
            aij = self.data[idx]
            # V^* A U = sum_ij conj(V_i) * A_ij * U_j
            result += np.conj(V[i]) * aij * U[j]
        
        return result
    
    def spy(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10,10))
        sizes = (np.ones(self.l, dtype=float) * 1000.0 / max(1, np.max(self.cols[:self.l])))

        
        scatter = ax.scatter(self.cols[:self.l], self.rows[:self.l], s=sizes, c=np.abs(self.data[:self.l]), 
                           cmap='RdBu_r', alpha=0.7)
        ax.set_aspect('equal')
        ax.set_title(f"spy de la matrice")
        ax.set_xlabel('j')
        ax.set_ylabel('i')
        plt.colorbar(scatter, ax=ax, label='Valeur')
        ax.grid(True, alpha=0.4)


        plt.tight_layout()
        plt.show()
        return fig, ax
    def __matmul__(self, U):
        """
        Produit matrice-vecteur : F = A @ U
        
        Parameters:
        -----------
        U : array (n,) ou (n,1)
            Vecteur à multiplier
            
        Returns:
        --------
        F : array (m,)
            Résultat du produit A @ U
        """
        # Recréer la matrice COO scipy (car elle peut avoir changé lors de l'assemblage)
        self.create_COO()
        
        # Utiliser le @ de scipy
        return self.coo @ U
        
    def produit_gauche(self, U):
        """
        Produit vecteur-matrice : F = U @ A
        
        Parameters:
        -----------
        U : array (m,) ou (1,m)
            Vecteur à multiplier (à gauche)
            
        Returns:
        --------
        F : array (n,)
            Résultat du produit U @ A
        """
        # Assurer que U est un vecteur 1D
        U = np.asarray(U).flatten()
        
        # Vérifier les dimensions
        if len(U) != self.nb_lig:
            raise ValueError(f"U doit avoir {self.nb_lig} éléments, mais a {len(U)}")
        
        # Initialiser le résultat
        F = np.zeros(self.nb_col,dtype=np.complex128)
        
        # Calculer U @ A en parcourant les éléments non-nuls de A
        # (U @ A)_j = sum_i U_i * A_ij
        for idx in range(self.l):
            i = int(self.rows[idx])
            j = int(self.cols[idx])
            aij = self.data[idx]
            # F_j += U_i * A_ij
            F[j] += U[i] * aij
        
        return F
    def is_symmetric(self, tol=1e-10):
        """
        Vérifie si la matrice est symétrique
        
        Parameters:
        -----------
        tol : float, optional
            Tolérance pour la comparaison (défaut: 1e-10)
            
        Returns:
        --------
        symmetric : bool
            True si la matrice est symétrique, False sinon
        """
        # Vérifier d'abord que la matrice est carrée
        if self.nb_lig != self.nb_col:
            return False
        
        # Créer la matrice COO scipy
        self.create_COO()
        
        # Calculer A - A^T
        diff = self.coo - self.coo.T
        
        # Vérifier si la norme est proche de zéro
        norm_diff = np.abs(diff.data).max() if diff.nnz > 0 else 0.0
        
        return norm_diff < tol
    def __mul__(self, scalar):
        """
        Multiplie la matrice par un scalaire
        
        Parameters:
        -----------
        scalar : float ou complex
            Scalaire à multiplier
        
        Returns:
        --------
        result : COOMatrix
            Nouvelle matrice résultante (self * scalar)
        """
        # Créer une nouvelle matrice avec les mêmes dimensions
        result = COOMatrix(self.nb_lig, self.nb_col, self.l)
        
        # Copier les indices
        result.rows[:self.l] = self.rows[:self.l]
        result.cols[:self.l] = self.cols[:self.l]
        
        # Multiplier les données par le scalaire
        result.data[:self.l] = self.data[:self.l] * scalar
        
        # Mettre à jour le compteur d'éléments
        result.l = self.l
        
        return result

    def __rmul__(self, scalar):
        """
        Permet la multiplication à gauche: scalar * matrix
        """
        return self.__mul__(scalar)

    def __truediv__(self, scalar):
        """
        Divise la matrice par un scalaire
        
        Parameters:
        -----------
        scalar : float ou complex
            Scalaire diviseur
            
        Returns:
        --------
        result : COOMatrix
            Nouvelle matrice résultante (self / scalar)
        """
        if scalar == 0:
            raise ValueError("Division par zéro")
        return self.__mul__(1.0 / scalar)
    
    @property
    def shape(self):
        return (self.nb_lig, self.nb_col)

    def __repr__(self):
        return f"COOMatrix({self.nb_lig}x{self.nb_col}, nnz={self.l}/{self.nnz})"
    
    def copy(self):
        B = COOMatrix(self.nb_lig, self.nb_col, self.nnz)
        B.rows[:self.l] = self.rows[:self.l]
        B.cols[:self.l] = self.cols[:self.l]
        B.data[:self.l] = self.data[:self.l]
        B.l = self.l
        return B
    
    def is_zero(self, tol=1e-10):
        """
        Teste si la matrice assemblée est numériquement nulle.
        (après réduction des contributions répétées)
        """
        if self.l == 0:
            return True

        # Passage par SciPy pour réduire les doublons
        self.create_COO()
        csr = self.coo.tocsr()

        if csr.nnz == 0:
            return True

        return np.max(np.abs(csr.data)) <= tol