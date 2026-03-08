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


    def __iadd__(self, other):
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

    def to_csr_clean(self, tol=0.0):
        from scipy.sparse import coo_matrix, csr_matrix

        rows = self.rows[:self.l]
        cols = self.cols[:self.l]
        data = self.data[:self.l]

        if tol > 0.0:
            mask = np.abs(data) > tol
            rows = rows[mask]
            cols = cols[mask]
            data = data[mask]

        coo = coo_matrix((data, (rows, cols)),
                        shape=(self.nb_lig, self.nb_col))

        return csr_matrix(coo)

    def spy_hyperbo(self, d, tol: float = 0.0, secondes: float = 0.0):
        """
        Affiche le spy de la matrice avec visualisation des blocs hyperboliques.

        Parameters
        ----------
        d : int
            Nombre de composantes du système hyperbolique.
        tol : float, optional
            Seuil sous lequel les coefficients sont ignorés.
        secondes : float, optional
            Ferme automatiquement la figure après ce temps (0 = pas de fermeture).
        """

        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        self.create_COO()

        if d <= 0:
            raise ValueError("d doit être > 0.")

        if self.nb_lig % d != 0:
            raise ValueError(
                f"Dimension incompatible avec d={d} "
                f"(nb_lig={self.nb_lig} non divisible par d)."
            )

        # Taille d’un bloc
        Nglob = self.nb_lig // d

        # --- Vérifie que le COO existe ---
        if not hasattr(self, "coo") or self.coo is None:
            raise RuntimeError("La matrice COO n'est pas construite.")

        csr = self.coo.tocsr()

        # --- nettoyage numérique ---
        if tol > 0:
            csr.data[np.abs(csr.data) < tol] = 0.0
            csr.eliminate_zeros()
        
        rows, cols = csr.nonzero()

        fig, ax = plt.subplots(figsize=(8, 8))

        # spy rapide (beaucoup plus performant que plt.spy)
        ax.plot(cols, rows, '.', color='black', markersize=1, linestyle='None')

        # convention matricielle
        ax.set_ylim(self.nb_lig, 0)
        ax.set_xlim(0, self.nb_col)
        ax.set_aspect('equal')

        # --- dessin des blocs hyperboliques ---
        for m in range(d):
            for n in range(d):
                x0 = n * Nglob
                y0 = m * Nglob

                rect = Rectangle(
                    (x0, y0),
                    Nglob,
                    Nglob,
                    fill=False,
                    edgecolor='red',
                    linewidth=1.2
                )
                ax.add_patch(rect)

        ax.set_title("Spy + structure hyperbolique")
        ax.set_xlabel("j")
        ax.set_ylabel("i")

        plt.tight_layout()

        # --- fermeture automatique ---
        timer = None
        if secondes > 0:
            timer = fig.canvas.new_timer(interval=int(1000 * secondes))
            timer.add_callback(lambda: plt.close(fig))
            timer.start()

        plt.show()

        return fig, ax

    def spy(self, tol=0.0):
        import matplotlib.pyplot as plt

        csr = self.to_csr_clean(tol)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.spy(csr, markersize=2) # type: ignore[arg-type] sur la ligne du spy.

        ax.set_title("Spy (CSR nettoyée)")
        ax.set_xlabel("j")
        ax.set_ylabel("i")

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
    

    def is_hermitian(self, tol=1e-10):
        """
        Vérifie si la matrice est hermitienne à une tolérance près.
        """
        if self.nb_lig != self.nb_col:
            return False

        self.create_COO()

        diff = (self.coo - self.coo.getH()).tocsr()
        diff.sum_duplicates()

        if diff.nnz == 0:
            return True

        diff.data[np.abs(diff.data) <= tol] = 0.0
        diff.eliminate_zeros()

        return diff.nnz == 0
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
    
    def is_positive(self, tol=1e-12):
        """
        Teste de UTAU pour quelques vecteurs aléatoires (méthode rapide mais pas très robuste)
        """

        self.to_csr()
        A = self.csr
        N= self.nb_lig


        ntests = 20
        lam_min = +np.inf

        for _ in range(ntests):
            x = np.random.randn(N)
            x /= np.linalg.norm(x)

            Ax = A @ x
            val = np.vdot(x, Ax).real  # énergie

            lam_min = min(lam_min, val)

            if val < -tol:
                return False, val

        return True
    
    def is_zero(self, tol=1e-12):
        """
        Teste si la matrice est nulle à une tolérance près,
        après sommation des contributions répétées.
        """
        csr = self.to_csr_clean(tol=0.0)
        csr.sum_duplicates()

        if csr.nnz == 0:
            return True

        return np.all(np.abs(csr.data) <= tol)

    def is_equal(self, other, tol=1e-8):
        """
        Teste si self et other sont égales à une tolérance près
        (après réduction des contributions répétées)
        """
        if self.nb_lig != other.nb_lig or self.nb_col != other.nb_col:
            return False
        
        MAT= COOMatrix(self.shape[0], self.shape[1], self.l + other.l)
        MAT += self
        MAT -= other
        return MAT.is_zero(tol=tol)

    def check_positive_definite(self, tol=1e-12):
        """
        Teste de UTAU pour quelques vecteurs aléatoires (méthode rapide mais pas très robuste)
        """
        self.to_csr()
        A = self.csr
        N= self.nb_lig

        # Utilisation de eigsh pour estimer la plus petite valeur propre
        from scipy.sparse.linalg import eigsh
        try:
            eigvals, _ = eigsh(A, k=1, which='SA', tol=tol) # pyright: ignore[reportArgumentType]
            lam_min = eigvals[0].real
            ok = lam_min > tol
        except Exception as e:
            print("Erreur lors du calcul des valeurs propres :", e)
            return False, None, None
        return ok, lam_min, eigvals
    
    def is_positive_slow(self, tol=1e-12):
        """
        Teste de UTAU pour quelques vecteurs aléatoires (méthode rapide mais pas très robuste)
        """
        self.to_csr()
        A = self.csr
        N= self.nb_lig

        # Utilisation de eigsh pour estimer la plus petite valeur propre
        from scipy.sparse.linalg import eigsh
        try:
            eigvals, _ = eigsh(A, k=5, which='SA', tol=tol) # pyright: ignore[reportArgumentType]
            lam_min = eigvals[0].real
            ok = lam_min > -tol
        except Exception as e:
            print("Erreur lors du calcul des valeurs propres :", e)
            return False, None, None
        return ok, lam_min, eigvals