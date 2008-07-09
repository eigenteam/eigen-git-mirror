//=====================================================
// Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
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
#ifndef EIGEN2_INTERFACE_HH
#define EIGEN2_INTERFACE_HH

#include <Eigen/Core>
#include <vector>

using namespace Eigen;

template<class real, int SIZE=Dynamic>
class eigen2_interface
{

public :

  typedef real real_type ;

  typedef std::vector<real> stl_vector;
  typedef std::vector<stl_vector> stl_matrix;

  typedef Eigen::Matrix<real,SIZE,SIZE> gene_matrix;
  typedef Eigen::Matrix<real,SIZE,1> gene_vector;

  static inline std::string name( void )
  {
    #if defined(EIGEN_VECTORIZE_SSE)
    if (SIZE==Dynamic) return "eigen2_SSE"; else return "tiny_eigen2_SSE";
    #elif defined(EIGEN_VECTORIZE_ALTIVEC)
    if (SIZE==Dynamic) return "eigen2_AltiVec"; else return "tiny_eigen2_AltiVec";
    #else
    if (SIZE==Dynamic) return "eigen2"; else return "tiny_eigen2";
    #endif
  }

  static void free_matrix(gene_matrix & A, int N) {}

  static void free_vector(gene_vector & B) {}

  static inline void matrix_from_stl(gene_matrix & A, stl_matrix & A_stl){
    A.resize(A_stl[0].size(), A_stl.size());

    for (int j=0; j<A_stl.size() ; j++){
      for (int i=0; i<A_stl[j].size() ; i++){
        A.coeffRef(i,j) = A_stl[j][i];
      }
    }
  }

  static inline void vector_from_stl(gene_vector & B, stl_vector & B_stl){
    B.resize(B_stl.size(),1);

    for (int i=0; i<B_stl.size() ; i++){
      B.coeffRef(i) = B_stl[i];
    }
  }

  static inline void vector_to_stl(gene_vector & B, stl_vector & B_stl){
    for (int i=0; i<B_stl.size() ; i++){
      B_stl[i] = B.coeff(i);
    }
  }

  static inline void matrix_to_stl(gene_matrix & A, stl_matrix & A_stl){
    int N=A_stl.size();

    for (int j=0;j<N;j++){
      A_stl[j].resize(N);
      for (int i=0;i<N;i++){
        A_stl[j][i] = A.coeff(i,j);
      }
    }
  }

  static inline void matrix_matrix_product(const gene_matrix & A, const gene_matrix & B, gene_matrix & X, int N){
    X = (A*B).lazy();
  }

  static inline void transposed_matrix_matrix_product(const gene_matrix & A, const gene_matrix & B, gene_matrix & X, int N){
    X = (A.transpose()*B.transpose()).lazy();
  }

  static inline void ata_product(const gene_matrix & A, gene_matrix & X, int N){
    X = (A.transpose()*A).lazy();
  }

  static inline void aat_product(const gene_matrix & A, gene_matrix & X, int N){
    X = (A*A.transpose()).lazy();
  }

  static inline void matrix_vector_product(gene_matrix & A, gene_vector & B, gene_vector & X, int N){
    X = (A*B).lazy();
  }

  static inline void atv_product(gene_matrix & A, gene_vector & B, gene_vector & X, int N){
    X = (A*B).lazy();
  }

  static inline void axpy(const real coef, const gene_vector & X, gene_vector & Y, int N){
    Y += coef * X;
  }

  static inline void copy_matrix(const gene_matrix & source, gene_matrix & cible, int N){
    cible = source;
  }

  static inline void copy_vector(const gene_vector & source, gene_vector & cible, int N){
    cible = source;
  }

};

#endif
