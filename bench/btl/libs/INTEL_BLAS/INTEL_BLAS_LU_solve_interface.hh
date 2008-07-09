//=====================================================
// File   :  INTEL_BLAS_LU_solve_interface.hh
// Author :  L. Plagne <laurent.plagne@edf.fr)>        
// Copyright (C) EDF R&D,  lun sep 30 14:23:29 CEST 2002
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
#ifndef INTEL_BLAS_LU_solve_interface_HH
#define INTEL_BLAS_LU_solve_interface_HH
#include "INTEL_BLAS_interface.hh"
extern "C"
{
//   void dgetrf_(int *M, int *N, double *A, int *LDA, int *IPIV, int *INFO);
//   void dgetrs_(char *TRANS, int *N, int *NRHS, double *A, int *LDA, int *IPIV, double *B, int *LDB, int *INFO);
//   void sgetrf_(int *M, int *N, float *A, int *LDA, int *IPIV, int *INFO);
//   void sgetrs_(char *TRANS, int *N, int *NRHS, float *A, int *LDA, int *IPIV, double *B, int *LDB, int *INFO);
#include "mkl_lapack.h"

}

template<class real>
class INTEL_BLAS_LU_solve_interface : public INTEL_BLAS_interface<real>
{
public :

  typedef typename INTEL_BLAS_interface<real>::gene_matrix gene_matrix;
  typedef typename INTEL_BLAS_interface<real>::gene_vector gene_vector;

  typedef int *  Pivot_Vector;
  
  inline static void new_Pivot_Vector(Pivot_Vector & pivot, int N)
  {
    
    pivot = new int[N];

  }

  inline static void free_Pivot_Vector(Pivot_Vector & pivot)
  {
    
    delete pivot;

  }
  

  inline static void LU_factor(gene_matrix & LU, Pivot_Vector & pivot, int N)
  {
    
    int info;
    DGETRF(&N,&N,LU,&N,pivot,&info);    

  }

  inline static void  LU_solve(const gene_matrix & LU, const Pivot_Vector pivot, const gene_vector &B, gene_vector X, int N)
  {
    int info;
    int one=1;

    char * transpose="N";

    copy_vector(B,X,N);
    DGETRS(transpose,&N,&one,LU,&N,pivot,X,&N,&info);
    
  }

};

template<>
class INTEL_BLAS_LU_solve_interface<float> : public INTEL_BLAS_interface<float>
{
public :

  typedef int *  Pivot_Vector;
  
  inline static void new_Pivot_Vector(Pivot_Vector & pivot, int N)
  {
    
    pivot = new int[N];

  }

  inline static void free_Pivot_Vector(Pivot_Vector & pivot)
  {
    
    delete pivot;

  }
  

  inline static void LU_factor(gene_matrix & LU, Pivot_Vector & pivot, int N)
  {
    
    int info;
    SGETRF(&N,&N,LU,&N,pivot,&info);    

  }

  inline static void  LU_solve(const gene_matrix & LU, const Pivot_Vector pivot, const gene_vector &B, gene_vector X, int N)
  {
        
    char * transpose="N";
    int info;
    int one=1;
    copy_vector(B,X,N);
    SGETRS(transpose,&N,&one,LU,&N,pivot,X,&N,&info);
    
  }

};


#endif



