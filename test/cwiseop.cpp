// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2006-2008 Benoit Jacob <jacob@math.jussieu.fr>
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

#include "main.h"
#include <iostream>
#include <cmath>
#include <cstdlib>

namespace Eigen {

template<typename Scalar> struct AddIfNull {
    static Scalar op(const Scalar a, const Scalar b) {return a<1e-3 ? b : a;}
};

template<typename MatrixType> void cwiseops(const MatrixType& m)
{
  typedef typename MatrixType::Scalar Scalar;
  typedef Matrix<Scalar, MatrixType::Traits::RowsAtCompileTime, 1> VectorType;
  
  int rows = m.rows();
  int cols = m.cols();
  
  MatrixType m1 = MatrixType::random(rows, cols),
             m2 = MatrixType::random(rows, cols),
             m3(rows, cols),
             mzero = MatrixType::zero(rows, cols),
             mones = MatrixType::ones(rows, cols),
             identity = Matrix<Scalar, MatrixType::Traits::RowsAtCompileTime, MatrixType::Traits::RowsAtCompileTime>
                              ::identity(rows, rows),
             square = Matrix<Scalar, MatrixType::Traits::RowsAtCompileTime, MatrixType::Traits::RowsAtCompileTime>
                              ::random(rows, rows);
  VectorType v1 = VectorType::random(rows),
             v2 = VectorType::random(rows),
             vzero = VectorType::zero(rows);

   
   m2 = m2.cwise<AddIfNull>(mones);
   //m2 = cwise<AddIfNull>(m2,mones);
  std::cout << m2 << "\n";
  
  VERIFY_IS_APPROX(               mzero,    m1-m1);
  VERIFY_IS_APPROX(               m2,       m1+m2-m1);
  VERIFY_IS_APPROX(               mones,    m2.cwiseQuotient(m2));
  VERIFY_IS_APPROX(               m1.cwiseProduct(m2),    m2.cwiseProduct(m1));
  //VERIFY_IS_APPROX(               m1,       m2.cwiseProduct(m1).cwiseQuotient(m2));
  
//   VERIFY_IS_APPROX( cwiseMin(m1,m2), cwiseMin(m2,m1) );
//   VERIFY_IS_APPROX( cwiseMin(m1,m1+mones), m1 );
//   VERIFY_IS_APPROX( cwiseMin(m1,m1-mones), m1-mones );
}

void EigenTest::testCwiseops()
{
  for(int i = 0; i < 1/*m_repeat*/ ; i++) {
//     cwiseops(Matrix<float, 1, 1>());
//     cwiseops(Matrix4d());
//     cwiseops(MatrixXcf(3, 3));
    cwiseops(MatrixXi(8, 12));
//     cwiseops(MatrixXcd(20, 20));
  }
}

} // namespace Eigen
