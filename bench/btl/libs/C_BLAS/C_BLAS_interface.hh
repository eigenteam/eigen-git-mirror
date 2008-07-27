//=====================================================
// File   :  C_BLAS_interface.hh
// Author :  L. Plagne <laurent.plagne@edf.fr)>
// Copyright (C) EDF R&D,  lun sep 30 14:23:28 CEST 2002
//=====================================================
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
//
#ifndef C_BLAS_PRODUIT_MATRICE_VECTEUR_HH
#define C_BLAS_PRODUIT_MATRICE_VECTEUR_HH

#include "f77_interface.hh"

extern "C"
{
#include "cblas.h"
#ifdef HAS_LAPACK
  // Cholesky Factorization
  void spotrf_(const char* uplo, const int* n, float *a, const int* ld, int* info);
  void dpotrf_(const char* uplo, const int* n, double *a, const int* ld, int* info);
#endif
}

#define MAKE_STRING2(S) #S
#define MAKE_STRING(S) MAKE_STRING2(S)

template<class real>
class C_BLAS_interface : public f77_interface_base<real>
{
public :

  typedef typename f77_interface_base<real>::gene_matrix gene_matrix;
  typedef typename f77_interface_base<real>::gene_vector gene_vector;

  static inline std::string name( void )
  {
    return MAKE_STRING(CBLASNAME);
  }

  static  inline void matrix_vector_product(gene_matrix & A, gene_vector & B, gene_vector & X, int N)
  {
    cblas_dgemv(CblasColMajor,CblasNoTrans,N,N,1.0,A,N,B,1,0.0,X,1);
  }

  static  inline void atv_product(gene_matrix & A, gene_vector & B, gene_vector & X, int N)
  {
    cblas_dgemv(CblasColMajor,CblasTrans,N,N,1.0,A,N,B,1,0.0,X,1);
  }

  static  inline void matrix_matrix_product(gene_matrix & A, gene_matrix & B, gene_matrix & X, int N){
    cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,N,N,N,1.0,A,N,B,N,0.0,X,N);
  }

  static  inline void transposed_matrix_matrix_product(gene_matrix & A, gene_matrix & B, gene_matrix & X, int N){
    cblas_dgemm(CblasColMajor,CblasTrans,CblasTrans,N,N,N,1.0,A,N,B,N,0.0,X,N);
  }

  static  inline void ata_product(gene_matrix & A, gene_matrix & X, int N){
    cblas_dgemm(CblasColMajor,CblasTrans,CblasNoTrans,N,N,N,1.0,A,N,A,N,0.0,X,N);
  }

  static  inline void aat_product(gene_matrix & A, gene_matrix & X, int N){
    cblas_dgemm(CblasColMajor,CblasNoTrans,CblasTrans,N,N,N,1.0,A,N,A,N,0.0,X,N);
  }

  static  inline void axpy(real coef, const gene_vector & X, gene_vector & Y, int N){
    cblas_daxpy(N,coef,X,1,Y,1);
  }

  static inline void axpby(real a, const gene_vector & X, real b, gene_vector & Y, int N){
    cblas_dscal(N,b,Y,1);
    cblas_daxpy(N,a,X,1,Y,1);
  }

};

template<>
class C_BLAS_interface<float> : public f77_interface_base<float>
{
public :

  static inline std::string name( void )
  {
    return MAKE_STRING(CBLASNAME);
  }

  static inline void matrix_vector_product(gene_matrix & A, gene_vector & B, gene_vector & X, int N){
    cblas_sgemv(CblasColMajor,CblasNoTrans,N,N,1.0,A,N,B,1,0.0,X,1);
  }

  static inline void atv_product(gene_matrix & A, gene_vector & B, gene_vector & X, int N){
    cblas_sgemv(CblasColMajor,CblasTrans,N,N,1.0,A,N,B,1,0.0,X,1);
  }

  static inline void matrix_matrix_product(gene_matrix & A, gene_matrix & B, gene_matrix & X, int N){
    cblas_sgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,N,N,N,1.0,A,N,B,N,0.0,X,N);
  }

  static inline void transposed_matrix_matrix_product(gene_matrix & A, gene_matrix & B, gene_matrix & X, int N){
    cblas_sgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,N,N,N,1.0,A,N,B,N,0.0,X,N);
  }

  static inline void ata_product(gene_matrix & A, gene_matrix & X, int N){
    cblas_sgemm(CblasColMajor,CblasTrans,CblasNoTrans,N,N,N,1.0,A,N,A,N,0.0,X,N);
  }

  static inline void aat_product(gene_matrix & A, gene_matrix & X, int N){
    cblas_sgemm(CblasColMajor,CblasNoTrans,CblasTrans,N,N,N,1.0,A,N,A,N,0.0,X,N);
  }

  static inline void axpy(float coef, const gene_vector & X, gene_vector & Y, int N){
    cblas_saxpy(N,coef,X,1,Y,1);
  }

  static inline void axpby(float a, const gene_vector & X, float b, gene_vector & Y, int N){
    cblas_sscal(N,b,Y,1);
    cblas_saxpy(N,a,X,1,Y,1);
  }

  #ifdef HAS_LAPACK
  static inline void cholesky(const gene_vector & X, gene_vector & C, int N){
    cblas_scopy(N*N, X, 1, C, 1);
    char uplo = 'L';
    int info = 0;
    spotrf_(&uplo, &N, C, &N, &info);
//     float tmp[N];
//     for (int j = 1; j < N; ++j)
//     {
//       int endSize = N-j-1;
//       if (endSize>0) {
          //cblas_sgemv(CblasColMajor, CblasNoTrans, N-j-1, j, ATL_rnone, A+j+1, lda, A+j, lda, ATL_rone, Ac+j+1,       1);
//          cblas_sgemv(CblasColMajor, CblasNoTrans,endSize,j, 1.0,     &(C[j+1]),N, &(C[j]),N, 0.0,      &(C[j+1+N*j]),1);
//       }
//     }
  }
  #endif

  static inline void trisolve_lower(const gene_matrix & L, const gene_vector& B, gene_vector & X, int N){
    cblas_scopy(N, B, 1, X, 1);
    cblas_strsv(CblasColMajor, CblasLower, CblasNoTrans, CblasNonUnit, N, L, N, X, 1);
  }

};


#endif



