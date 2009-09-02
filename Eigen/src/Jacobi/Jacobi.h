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

/** \ingroup Jacobi
  * \class PlanarRotation
  * \brief Represents a rotation in the plane from a cosine-sine pair.
  *
  * This class represents a Jacobi or Givens rotation.
  * This is a 2D clock-wise rotation in the plane \c J of angle \f$ \theta \f$ defined by
  * its cosine \c c and sine \c s as follow:
  * \f$ J = \left ( \begin{array}{cc} c & \overline s \\ -s  & \overline c \end{array} \right ) \f$
  *
  * \sa MatrixBase::makeJacobi(), MatrixBase::applyOnTheLeft(), MatrixBase::applyOnTheRight()
  */
template<typename Scalar> class PlanarRotation
{
  public:
    typedef typename NumTraits<Scalar>::Real RealScalar;

    /** Default constructor without any initialization. */
    PlanarRotation() {}

    /** Construct a planar rotation from a cosine-sine pair (\a c, \c s). */
    PlanarRotation(const Scalar& c, const Scalar& s) : m_c(c), m_s(s) {}

    Scalar& c() { return m_c; }
    Scalar c() const { return m_c; }
    Scalar& s() { return m_s; }
    Scalar s() const { return m_s; }

    /** Concatenates two planar rotation */
    PlanarRotation operator*(const PlanarRotation& other)
    {
      return PlanarRotation(m_c * other.m_c - ei_conj(m_s) * other.m_s,
                            ei_conj(m_c * ei_conj(other.m_s) + ei_conj(m_s) * ei_conj(other.m_c)));
    }

    /** Returns the transposed transformation */
    PlanarRotation transpose() const { return PlanarRotation(m_c, -ei_conj(m_s)); }

    /** Returns the adjoint transformation */
    PlanarRotation adjoint() const { return PlanarRotation(ei_conj(m_c), -m_s); }

    template<typename Derived>
    bool makeJacobi(const MatrixBase<Derived>&, int p, int q);
    bool makeJacobi(RealScalar x, Scalar y, RealScalar z);

    void makeGivens(const Scalar& p, const Scalar& q, Scalar* z=0);

  protected:
    void makeGivens(const Scalar& p, const Scalar& q, Scalar* z, ei_meta_true);
    void makeGivens(const Scalar& p, const Scalar& q, Scalar* z, ei_meta_false);

    Scalar m_c, m_s;
};

/** Makes \c *this as a Jacobi rotation \a J such that applying \a J on both the right and left sides of the 2x2 matrix
  * \f$ B = \left ( \begin{array}{cc} x & y \\ * & z \end{array} \right )\f$ yields
  * a diagonal matrix \f$ A = J^* B J \f$
  *
  * \sa MatrixBase::makeJacobi(), MatrixBase::applyOnTheLeft(), MatrixBase::applyOnTheRight()
  */
template<typename Scalar>
bool PlanarRotation<Scalar>::makeJacobi(RealScalar x, Scalar y, RealScalar z)
{
  typedef typename NumTraits<Scalar>::Real RealScalar;
  if(y == Scalar(0))
  {
    m_c = Scalar(1);
    m_s = Scalar(0);
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
    m_s = - sign_t * (ei_conj(y) / ei_abs(y)) * ei_abs(t) * n;
    m_c = n;
    return true;
  }
}

/** Makes \c *this as a Jacobi rotation \c J such that applying \a J on both the right and left sides of the 2x2 matrix
  * \f$ B = \left ( \begin{array}{cc} \text{this}_{pp} & \text{this}_{pq} \\ * & \text{this}_{qq} \end{array} \right )\f$ yields
  * a diagonal matrix \f$ A = J^* B J \f$
  *
  * \sa PlanarRotation::makeJacobi(RealScalar, Scalar, RealScalar), MatrixBase::applyOnTheLeft(), MatrixBase::applyOnTheRight()
  */
template<typename Scalar>
template<typename Derived>
inline bool PlanarRotation<Scalar>::makeJacobi(const MatrixBase<Derived>& m, int p, int q)
{
  return makeJacobi(ei_real(m.coeff(p,p)), m.coeff(p,q), ei_real(m.coeff(q,q)));
}

/** Makes \c *this as a Givens rotation \c G such that applying \f$ G^* \f$ to the left of the vector
  * \f$ V = \left ( \begin{array}{c} p \\ q \end{array} \right )\f$ yields:
  * \f$ G^* V = \left ( \begin{array}{c} z \\ 0 \end{array} \right )\f$.
  *
  * The value of \a z is returned if \a z is not null (the default is null).
  * Also note that G is built such that the cosine is always real.
  *
  * \sa MatrixBase::applyOnTheLeft(), MatrixBase::applyOnTheRight()
  */
template<typename Scalar>
void PlanarRotation<Scalar>::makeGivens(const Scalar& p, const Scalar& q, Scalar* z)
{
  makeGivens(p, q, z, typename ei_meta_if<NumTraits<Scalar>::IsComplex, ei_meta_true, ei_meta_false>::ret());
}


// specialization for complexes
template<typename Scalar>
void PlanarRotation<Scalar>::makeGivens(const Scalar& p, const Scalar& q, Scalar* z, ei_meta_true)
{
  RealScalar scale, absx, absxy;
  if(q==Scalar(0))
  {
    // return identity
    m_c = Scalar(1);
    m_s = Scalar(0);
    if(z) *z = p;
  }
  else
  {
    scale = ei_norm1(p);
    absx = scale * ei_sqrt(ei_abs2(p/scale));
    scale = ei_abs(scale) + ei_norm1(q);
    absxy = scale * ei_sqrt((absx/scale)*(absx/scale) + ei_abs2(q/scale));
    m_c = Scalar(absx / absxy);
    Scalar np = p/absx;
    m_s = -ei_conj(np) * q / absxy;
    if(z) *z = np * absxy;
  }
}

// specialization for reals
template<typename Scalar>
void PlanarRotation<Scalar>::makeGivens(const Scalar& p, const Scalar& q, Scalar* z, ei_meta_false)
{
  // from Golub's "Matrix Computations", algorithm 5.1.3
  if(q==0)
  {
    m_c = 1; m_s = 0;
  }
  else if(ei_abs(q)>ei_abs(p))
  {
    Scalar t = -p/q;
    m_s = Scalar(1)/ei_sqrt(1+t*t);
    m_c = m_s * t;
  }
  else
  {
    Scalar t = -q/p;
    m_c = Scalar(1)/ei_sqrt(1+t*t);
    m_s = m_c * t;
  }
}

/****************************************************************************************
*   Implementation of MatrixBase methods
****************************************************************************************/

/** Applies the clock wise 2D rotation \a j to the set of 2D vectors of cordinates \a x and \a y:
  * \f$ \left ( \begin{array}{cc} x \\ y \end{array} \right )  =  J \left ( \begin{array}{cc} x \\ y \end{array} \right ) \f$
  *
  * \sa MatrixBase::applyOnTheLeft(), MatrixBase::applyOnTheRight()
  */
template<typename VectorX, typename VectorY, typename OtherScalar>
void ei_apply_rotation_in_the_plane(VectorX& _x, VectorY& _y, const PlanarRotation<OtherScalar>& j);

/** Applies the rotation in the plane \a j to the rows \a p and \a q of \c *this, i.e., it computes B = J * B,
  * with \f$ B = \left ( \begin{array}{cc} \text{*this.row}(p) \\ \text{*this.row}(q) \end{array} \right ) \f$.
  *
  * \sa class PlanarRotation, MatrixBase::applyOnTheRight(), ei_apply_rotation_in_the_plane()
  */
template<typename Derived>
template<typename OtherScalar>
inline void MatrixBase<Derived>::applyOnTheLeft(int p, int q, const PlanarRotation<OtherScalar>& j)
{
  RowXpr x(row(p));
  RowXpr y(row(q));
  ei_apply_rotation_in_the_plane(x, y, j);
}

/** Applies the rotation in the plane \a j to the columns \a p and \a q of \c *this, i.e., it computes B = B * J
  * with \f$ B = \left ( \begin{array}{cc} \text{*this.col}(p) & \text{*this.col}(q) \end{array} \right ) \f$.
  *
  * \sa class PlanarRotation, MatrixBase::applyOnTheLeft(), ei_apply_rotation_in_the_plane()
  */
template<typename Derived>
template<typename OtherScalar>
inline void MatrixBase<Derived>::applyOnTheRight(int p, int q, const PlanarRotation<OtherScalar>& j)
{
  ColXpr x(col(p));
  ColXpr y(col(q));
  ei_apply_rotation_in_the_plane(x, y, j.transpose());
}


template<typename VectorX, typename VectorY, typename OtherScalar>
void /*EIGEN_DONT_INLINE*/ ei_apply_rotation_in_the_plane(VectorX& _x, VectorY& _y, const PlanarRotation<OtherScalar>& j)
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

    const Packet pc = ei_pset1(Scalar(j.c()));
    const Packet ps = ei_pset1(Scalar(j.s()));
    ei_conj_helper<NumTraits<Scalar>::IsComplex,false> cj;

    for(int i=0; i<alignedStart; ++i)
    {
      Scalar xi = x[i];
      Scalar yi = y[i];
      x[i] =  j.c() * xi + ei_conj(j.s()) * yi;
      y[i] = -j.s() * xi + ei_conj(j.c()) * yi;
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
      x[i] =  j.c() * xi + ei_conj(j.s()) * yi;
      y[i] = -j.s() * xi + ei_conj(j.c()) * yi;
    }
  }
  else
  {
    for(int i=0; i<size; ++i)
    {
      Scalar xi = *x;
      Scalar yi = *y;
      *x =  j.c() * xi + ei_conj(j.s()) * yi;
      *y = -j.s() * xi + ei_conj(j.c()) * yi;
      x += incrx;
      y += incry;
    }
  }
}

#endif // EIGEN_JACOBI_H
