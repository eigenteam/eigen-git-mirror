// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2012 Désiré Nuentsa-Wakam <desire.nuentsa_wakam@inria.fr>
//
// Eigen is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 3 of the License, or (at your option) any later version.
//
// Alternatively, you can redistribute it and/or
// modify it under the terms of the GNU General Public License as
// published by the Free Software Foundation; either version 2 of
// the License, or (at your option) any later version.
//
// Eigen is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License or the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License and a copy of the GNU General Public License along with
// Eigen. If not, see <http://www.gnu.org/licenses/>.

#ifndef SPARSELU_KERNEL_BMOD_H
#define SPARSELU_KERNEL_BMOD_H

/**
 * \brief Performs numeric block updates from a given supernode to a single column
 * 
 * \param segsize Size of the segment (and blocks ) to use for updates
 * \param [in,out]dense Packed values of the original matrix
 * \param tempv temporary vector to use for updates
 * \param lusup array containing the supernodes
 * \param nsupr Number of rows in the supernode
 * \param nrow Number of rows in the rectangular part of the supernode
 * \param lsub compressed row subscripts of supernodes
 * \param lptr pointer to the first column of the current supernode in lsub
 * \param no_zeros Number of nonzeros elements before the diagonal part of the supernode
 * \return 0 on success
 */
template <typename BlockScalarVector, typename ScalarVector, typename IndexVector>
int LU_kernel_bmod(const int segsize, BlockScalarVector& dense, ScalarVector& tempv, ScalarVector& lusup, int& luptr, const int nsupr, const int nrow, IndexVector& lsub, const int lptr, const int no_zeros)
{
  typedef typename ScalarVector::Scalar Scalar; 
  // First, copy U[*,j] segment from dense(*) to tempv(*)
   // The result of triangular solve is in tempv[*]; 
    // The result of matric-vector update is in dense[*]
  int isub = lptr + no_zeros; 
  int i, irow;
  for (i = 0; i < segsize; i++)
  {
    irow = lsub(isub); 
    tempv(i) = dense(irow); 
    ++isub; 
  }
  // Dense triangular solve -- start effective triangle
  luptr += nsupr * no_zeros + no_zeros; 
  // Form Eigen matrix and vector 
  Map<Matrix<Scalar,Dynamic,Dynamic>, 0, OuterStride<> > A( &(lusup.data()[luptr]), segsize, segsize, OuterStride<>(nsupr) );
  VectorBlock<ScalarVector> u(tempv, 0, segsize);
  
  u = A.template triangularView<UnitLower>().solve(u); 
  
  // Dense matrix-vector product y <-- A*x 
  luptr += segsize; 
  new (&A) Map<Matrix<Scalar,Dynamic, Dynamic>, 0, OuterStride<> > ( &(lusup.data()[luptr]), nrow, segsize, OuterStride<>(nsupr) ); 
  VectorBlock<ScalarVector> l(tempv, segsize, nrow); 
  l= A * u;
  
  // Scatter tempv[] into SPA dense[] as a temporary storage 
  isub = lptr + no_zeros; 
  for (i = 0; i < segsize; i++)
  {
    irow = lsub(isub); 
    dense(irow) = tempv(i); 
    tempv(i) =  Scalar(0.0); 
    ++isub;
  }
  
  // Scatter l into SPA dense[]
  for (i = 0; i < nrow; i++)
  {
    irow = lsub(isub); 
    dense(irow) -= l(i); 
    l(i) = Scalar(0.0); 
    ++isub; 
  }
  
  return 0; 
}
#endif