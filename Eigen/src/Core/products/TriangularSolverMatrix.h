// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Gael Guennebaud <g.gael@free.fr>
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

#ifndef EIGEN_TRIANGULAR_SOLVER_MATRIX_H
#define EIGEN_TRIANGULAR_SOLVER_MATRIX_H

/* Optimized triangular solver with multiple right hand side (_TRSM)
 */
template <typename Scalar,
          int LhsStorageOrder,
          int RhsStorageOrder,
          int Mode>
struct ei_triangular_solve_matrix//<Scalar,LhsStorageOrder,RhsStorageOrder>
{

  static EIGEN_DONT_INLINE void run(
    int size, int cols,
    const Scalar* _lhs, int lhsStride,
          Scalar* _rhs, int rhsStride)
  {
    Map<Matrix<Scalar,Dynamic,Dynamic,LhsStorageOrder> > lhs(_lhs, size, size);
    Map<Matrix<Scalar,Dynamic,Dynamic,RhsStorageOrder> > rhs(_rhs, size, cols);
    //ei_const_blas_data_mapper<Scalar, LhsStorageOrder> lhs(_lhs,lhsStride);
    //ei_const_blas_data_mapper<Scalar, RhsStorageOrder> rhs(_rhs,rhsStride);

    typedef ei_product_blocking_traits<Scalar> Blocking;
    enum {
      SmallPanelWidth   = EIGEN_ENUM_MAX(Blocking::mr,Blocking::nr),
      IsLowerTriangular = (Mode&LowerTriangular) == LowerTriangular
    };

    int kc = 8;//std::min<int>(Blocking::Max_kc,size);  // cache block size along the K direction
    int mc = 8;//std::min<int>(Blocking::Max_mc,size);  // cache block size along the M direction

    Scalar* blockA = ei_aligned_stack_new(Scalar, kc*mc);
    Scalar* blockB = ei_aligned_stack_new(Scalar, kc*cols*Blocking::PacketSize);

    ei_gebp_kernel<Scalar, Blocking::mr, Blocking::nr, ei_conj_helper<false,false> > gebp_kernel;

    for(int k2=0; k2<size; k2+=kc)
    {
      const int actual_kc = std::min(k2+kc,size)-k2;

      // We have selected and packed a big horizontal panel R1 of rhs. Let B be the packed copy of this panel,
      // and R2 the remaining part of rhs. The corresponding vertical panel of lhs is split into
      // A11 (the triangular part) and A21 the remaining rectangular part.
      // Then the high level algorithm is:
      //  - B = R1                    => general block copy
      //  - R1 = L1^-1 B              => tricky part
      //  - update B from the new R1  => actually this has to performed continuously during the above step
      //  - R2 = L2 * B               => GEPP

      // B = R1
      ei_gemm_pack_rhs<Scalar,Blocking::nr,RhsStorageOrder>()
        (blockB, &rhs(k2,0), rhsStride, -1, actual_kc, cols);

        Map<MatrixXf>(blockB,Blocking::PacketSize*Blocking::nr*actual_kc, cols/Blocking::nr+(cols%Blocking::nr)).setZero();

      // The tricky part: R1 = L1^-1 B while updating B from R1
      // The idea is to split L1 into multiple small vertical panels.
      // Each panel can be split into a small triangular part A1 which is processed without optimization,
      // and the remaining small part A2 which is processed using gebp with appropriate block strides
      {
        // pack L1
//         ei_gemm_pack_lhs<Scalar,Blocking::mr,LhsStorageOrder>()
//           (blockA, &lhs(k2, k2), lhsStride, actual_kc, actual_kc);

        // for each small vertical panels of lhs
        for (int k1=0; k1<actual_kc; k1+=SmallPanelWidth)
        {
          int actualPanelWidth = std::min<int>(SmallPanelWidth,actual_kc-k1);
          // tr solve
          for (int k=0; k<actualPanelWidth; ++k)
          {
            int i = k2+k1+k;
            if(!(Mode & UnitDiagBit))
              rhs.row(i) /= lhs(i,i);

            int rs = actualPanelWidth - k - 1; // remaining size
            //std::cerr << i << " ; " << k << " " << rs << "\n";
            if (rs>0)
            {
              rhs.block(i+1,0,rs,cols) -=
                    lhs.col(i).segment(IsLowerTriangular ? i+1 : i-rs, rs) * rhs.row(i);
            }
          }
          // update the respective row of B from rhs
          {
            const Scalar* lr = _rhs+k2+k1;
            int packet_cols = (cols/Blocking::nr) * Blocking::nr;
            int count = 0;
            for(int j2=0; j2<packet_cols; j2+=Blocking::nr)
            {
              // skip what we have before
              count += Blocking::PacketSize * Blocking::nr * (k1-k2);
              const Scalar* b0 = &lr[(j2+0)*rhsStride];
              const Scalar* b1 = &lr[(j2+1)*rhsStride];
              const Scalar* b2 = &lr[(j2+2)*rhsStride];
              const Scalar* b3 = &lr[(j2+3)*rhsStride];
              for(int k=0; k<actualPanelWidth; k++)
              {
                ei_pstore(&blockB[count+0*Blocking::PacketSize], ei_pset1(-b0[k]));
                ei_pstore(&blockB[count+1*Blocking::PacketSize], ei_pset1(-b1[k]));
                if (Blocking::nr==4)
                {
                  ei_pstore(&blockB[count+2*Blocking::PacketSize], ei_pset1(-b2[k]));
                  ei_pstore(&blockB[count+3*Blocking::PacketSize], ei_pset1(-b3[k]));
                }
                count += Blocking::nr*Blocking::PacketSize;
              }
              // skip what we have after
              count += Blocking::PacketSize * Blocking::nr * (actual_kc-k1-actualPanelWidth);
            }
            // copy the remaining columns one at a time (nr==1)
            for(int j2=packet_cols; j2<cols; ++j2)
            {
              count += Blocking::PacketSize * (k1-k2);
              const Scalar* b0 = &lr[(j2+0)*rhsStride];
              for(int k=0; k<actualPanelWidth; k++)
              {
                ei_pstore(&blockB[count], ei_pset1(-b0[k]));
                count += Blocking::PacketSize;
              }
              count += Blocking::PacketSize * (actual_kc-k1-actualPanelWidth);
            }
          }

//           std::cerr << Map<MatrixXf>(blockB,Blocking::PacketSize*Blocking::nr*actual_kc, cols/Blocking::nr+(cols%Blocking::nr)) << "\n\n";

//           MatrixXf aux(Blocking::PacketSize*Blocking::nr*actual_kc, cols/Blocking::nr+(cols%Blocking::nr));
//           aux.setZero();

//           ei_gemm_pack_rhs<Scalar,Blocking::nr,RhsStorageOrder>()
//                           (aux.data(), &rhs(k2,0), rhsStride, -1, actual_kc, cols);

//           std::cerr << Map<MatrixXf>(blockB,Blocking::PacketSize*Blocking::nr*actual_kc, cols/Blocking::nr+(cols%Blocking::nr)) - aux << "\n\n";


          // gebp
          int i = k1+actualPanelWidth;
          int rs = actual_kc-i;

//           ei_gemm_pack_rhs<Scalar,Blocking::nr,RhsStorageOrder>()
//                           (blockB, &rhs(k1,0), rhsStride, -1, actualPanelWidth, cols);

          ei_gemm_pack_lhs<Scalar,Blocking::mr,LhsStorageOrder>()
                          (blockA, &lhs(k2+i, k2+k1), lhsStride, actualPanelWidth, rs);

          if (rs>0)
            rhs.block(i,0,actual_kc-i,cols) -= lhs.block(i,k1,rs,actualPanelWidth) * rhs.block(k1,0,actualPanelWidth,cols);

//             gebp_kernel(_rhs+i+k2, rhsStride,
//                       blockA/*+actual_kc*i+k1*rs*/, blockB/*+k1*Blocking::PacketSize*Blocking::nr*/, rs, actualPanelWidth, cols, actualPanelWidth/*actual_kc*/, actual_kc, 0, k1*Blocking::PacketSize);

//           gebp_kernel(_rhs+i, rhsStride,
//                       blockA+actual_kc*i+k1*rs, blockB+k1*Blocking::PacketSize*Blocking::nr, rs, actualPanelWidth, cols, actual_kc, actual_kc);

//           gebp_kernel(_rhs+k2+i, rhsStride,
//                       blockA+actual_kc*i+k1, blockB+k1*Blocking::PacketSize, actual_kc-i, actualPanelWidth, cols, actual_kc, actual_kc);
        }
      }

      //  - R2 = A2 * B => GEPP
      for(int i2=k2+kc; i2<size; i2+=mc)
      {
        const int actual_mc = std::min(i2+mc,size)-i2;
        ei_gemm_pack_lhs<Scalar,Blocking::mr,LhsStorageOrder>()
          (blockA, &lhs(k2, i2), lhsStride, actual_kc, actual_mc);

        std::cerr << i2 << " sur " << actual_mc << " -= " << i2 << "x" << k2 << "+" << actual_mc<<"," <<actual_kc << " * " << k2 << " sur " << actual_kc << "\n";
        rhs.block(i2,0,actual_mc,cols) -= lhs.block(i2,k2,actual_mc,actual_kc) * rhs.block(k2,0,actual_kc,cols);

//         gebp_kernel(_rhs+i2, rhsStride, blockA, blockB, actual_mc, actual_kc, cols);
      }
    }

    ei_aligned_stack_delete(Scalar, blockA, kc*mc);
    ei_aligned_stack_delete(Scalar, blockB, kc*cols*Blocking::PacketSize);
  }
};

#endif // EIGEN_TRIANGULAR_SOLVER_MATRIX_H
