def assemble_hyperbo(m,n,A,Mat,Nglob):
    """
    Assemble le block sparse A dans la matrice globale Mat
    aux positions (m,n) du système hyperbolique
    
    Parameters:
    -----------
    m, n : int
        Indices de block (composantes du système hyperbolique)
    A : COOMatrix
        Matrice block sparse à assembler
    Mat : COOMatrix
        Matrice globale où assembler
    Nglob : int
        Dimension d'un block
    """
    for i in range(A.l):
        Mat.ajout(m*Nglob + A.rows[i], n*Nglob + A.cols[i], A.data[i])