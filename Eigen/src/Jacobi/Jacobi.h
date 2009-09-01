// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Benoit Jacob <jacob.benoit.1@gmail.com>
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

#ifndef EIGEN_JACOBI_H
#define EIGEN_JACOBI_H

/** Applies the counter clock wise 2D rotation of angle \c theta given by its
  * cosine \a c and sine \a s to the set of 2D vectors of cordinates \a x and \a y:
  * \f$ x = c x - s' y \f$
  * \f$ y = s x + c y \f$
  *
  * \sa MatrixBase::applyJacobiOnTheLeft(), MatrixBase::applyJacobiOnTheRight()
  */
template<typename VectorX, typename VectorY, typename JacobiScalar>
void ei_apply_rotation_in_the_plane(VectorX& _x, VectorY& _y, JacobiScalar c, JacobiScalar s);

/** Applies a rotation in the plane defined by \a c, \a s to the rows \a p and \a q of \c *this.
  * More precisely, it computes B = J' * B, with J = [c s ; -s' c] and B = [ *this.row(p) ; *this.row(q) ]
  * \sa MatrixBase::applyJacobiOnTheRight(), ei_apply_rotation_in_the_plane()
  */
template<typename Derived>
template<typename JacobiScalar>
inline void MatrixBase<Derived>::applyJacobiOnTheLeft(int p, int q, JacobiScalar c, JacobiScalar s)
{
  RowXpr x(row(p));
  RowXpr y(row(q));
  ei_apply_rotation_in_the_plane(x, y, c, s);
}

/** Applies a rotation in the plane defined by \a c, \a s to the columns \a p and \a q of \c *this.
  * More precisely, it computes B = B * J, with J = [c s ; -s' c] and B = [ *this.col(p) ; *this.col(q) ]
  * \sa MatrixBase::applyJacobiOnTheLeft(), ei_apply_rotation_in_the_plane()
  */
template<typename Derived>
template<typename JacobiScalar>
inline void MatrixBase<Derived>::applyJacobiOnTheRight(int p, int q, JacobiScalar c, JacobiScalar s)
{
  ColXpr x(col(p));
  ColXpr y(col(q));
  ei_apply_rotation_in_the_plane(x, y, c, -ei_conj(s));
}

/** Computes the cosine-sine pair (\a c, \a s) such that its associated
  * rotation \f$ J = ( \begin{array}{cc} c & \overline s \\ -s & \overline c \end{array} )\f$
  * applied to both the right and left of the 2x2 matrix
  * \f$ B = ( \begin{array}{cc} x & y \\ * & z \end{array} )\f$ yields
  * a diagonal matrix A: \f$ A = J^* B J \f$
  */
template<typename Scalar>
bool ei_makeJacobi(typename NumTraits<Scalar>::Real x, Scalar y, typename NumTraits<Scalar>::Real z, Scalar *c, Scalar *s)
{
  typedef typename NumTraits<Scalar>::Real RealScalar;
  if(y == Scalar(0))
  {
    *c = Scalar(1);
    *s = Scalar(0);
    return false;
  }
  else
  {
    RealScalar tau = (x-z)/(RealScalar(2)*ei_abs(y));
    RealScalar w = ei_sqrt(ei_abs2(tau) + 1);
    RealScalar t;
    if(tau>0)
    {
      t = RealScalar(1) / (tau + w);
    }
    else
    {
      t = RealScalar(1) / (tau - w);
    }
    RealScalar sign_t = t > 0 ? 1 : -1;
    RealScalar n = RealScalar(1) / ei_sqrt(ei_abs2(t)+1);
    *s = - sign_t * (ei_conj(y) / ei_abs(y)) * ei_abs(t) * n;
    *c = n;
    return true;
  }
}

template<typename Derived>
inline bool MatrixBase<Derived>::makeJacobi(int p, int q, Scalar *c, Scalar *s) const
{
  return ei_makeJacobi(ei_real(coeff(p,p)), coeff(p,q), ei_real(coeff(q,q)), c, s);
}

template<typename VectorX, typename VectorY, typename JacobiScalar>
void /*EIGEN_DONT_INLINE*/ ei_apply_rotation_in_the_plane(VectorX& _x, VectorY& _y, JacobiScalar c, JacobiScalar s)
{
  typedef typename VectorX::Scalar Scalar;
  ei_assert(_x.size() == _y.size());
  int size = _x.size();
  int incrx = size ==1 ? 1 : &_x.coeffRef(1) - &_x.coeffRef(0);
  int incry = size ==1 ? 1 : &_y.coeffRef(1) - &_y.coeffRef(0);

  Scalar* EIGEN_RESTRICT x = &_x.coeffRef(0);
  Scalar* EIGEN_RESTRICT y = &_y.coeffRef(0);

  if((VectorX::Flags & VectorY::Flags & PacketAccessBit) && incrx==1 && incry==1)
  {
    // both vectors are sequentially stored in memory => vectorization
    typedef typename ei_packet_traits<Scalar>::type Packet;
    enum { PacketSize = ei_packet_traits<Scalar>::size, Peeling = 2 };

    int alignedStart = ei_alignmentOffset(y, size);
    int alignedEnd = alignedStart + ((size-alignedStart)/PacketSize)*PacketSize;

    const Packet pc = ei_pset1(Scalar(c));
    const Packet ps = ei_pset1(Scalar(s));
    ei_conj_helper<NumTraits<Scalar>::IsComplex,false> cj;

    for(int i=0; i<alignedStart; ++i)
    {
      Scalar xi = x[i];
      Scalar yi = y[i];
      x[i] = c * xi + ei_conj(s) * yi;
      y[i] = - s * xi + ei_conj(c) * yi;
    }

    Scalar* px = x + alignedStart;
    Scalar* py = y + alignedStart;

    if(ei_alignmentOffset(x, size)==alignedStart)
    {
      for(int i=alignedStart; i<alignedEnd; i+=PacketSize)
      {
        Packet xi = ei_pload(px);
        Packet yi = ei_pload(py);
        ei_pstore(px, ei_padd(ei_pmul(pc,xi),cj.pmul(ps,yi)));
        ei_pstore(py, ei_psub(ei_pmul(pc,yi),ei_pmul(ps,xi)));
        px += PacketSize;
        py += PacketSize;
      }
    }
    else
    {
      int peelingEnd = alignedStart + ((size-alignedStart)/(Peeling*PacketSize))*(Peeling*PacketSize);
      for(int i=alignedStart; i<peelingEnd; i+=Peeling*PacketSize)
      {
        Packet xi   = ei_ploadu(px);
        Packet xi1  = ei_ploadu(px+PacketSize);
        Packet yi   = ei_pload (py);
        Packet yi1  = ei_pload (py+PacketSize);
        ei_pstoreu(px, ei_padd(ei_pmul(pc,xi),cj.pmul(ps,yi)));
        ei_pstoreu(px+PacketSize, ei_padd(ei_pmul(pc,xi1),cj.pmul(ps,yi1)));
        ei_pstore (py, ei_psub(ei_pmul(pc,yi),ei_pmul(ps,xi)));
        ei_pstore (py+PacketSize, ei_psub(ei_pmul(pc,yi1),ei_pmul(ps,xi1)));
        px += Peeling*PacketSize;
        py += Peeling*PacketSize;
      }
      if(alignedEnd!=peelingEnd)
      {
        Packet xi = ei_ploadu(x+peelingEnd);
        Packet yi = ei_pload (y+peelingEnd);
        ei_pstoreu(x+peelingEnd, ei_padd(ei_pmul(pc,xi),cj.pmul(ps,yi)));
        ei_pstore (y+peelingEnd, ei_psub(ei_pmul(pc,yi),ei_pmul(ps,xi)));
      }
    }

    for(int i=alignedEnd; i<size; ++i)
    {
      Scalar xi = x[i];
      Scalar yi = y[i];
      x[i] = c * xi + ei_conj(s) * yi;
      y[i] = -s * xi + ei_conj(c) * yi;
    }
  }
  else
  {
    for(int i=0; i<size; ++i)
    {
      Scalar xi = *x;
      Scalar yi = *y;
      *x = c * xi + ei_conj(s) * yi;
      *y = -s * xi + ei_conj(c) * yi;
      x += incrx;
      y += incry;
    }
  }
}

#endif // EIGEN_JACOBI_H
