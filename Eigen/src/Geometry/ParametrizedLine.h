// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
// Copyright (C) 2008 Benoit Jacob <jacob@math.jussieu.fr>
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

#ifndef EIGEN_PARAMETRIZEDLINE_H
#define EIGEN_PARAMETRIZEDLINE_H

/** \geometry_module \ingroup GeometryModule
  *
  * \class ParametrizedLine
  *
  * \brief A parametrized line
  *
  * \param _Scalar the scalar type, i.e., the type of the coefficients
  * \param _AmbientDim the dimension of the ambient space, can be a compile time value or Dynamic.
  *             Notice that the dimension of the hyperplane is _AmbientDim-1.
  */
template <typename _Scalar, int _AmbientDim>
class ParametrizedLine
  #ifdef EIGEN_VECTORIZE
  : public ei_with_aligned_operator_new<_Scalar,_AmbientDim>
  #endif
{
  public:

    enum { AmbientDimAtCompileTime = _AmbientDim };
    typedef _Scalar Scalar;
    typedef typename NumTraits<Scalar>::Real RealScalar;
    typedef Matrix<Scalar,AmbientDimAtCompileTime,1> VectorType;

    /** Default constructor without initialization */
    inline explicit ParametrizedLine(int _dim = AmbientDimAtCompileTime)
      : m_origin(_dim), m_direction(_dim)
    {}
    
    ParametrizedLine(const VectorType& origin, const VectorType& direction)
      : m_origin(origin), m_direction(direction) {}
    explicit ParametrizedLine(const Hyperplane<_Scalar, _AmbientDim>& hyperplane);

    ~ParametrizedLine() {}

    /** \returns the dimension in which the line holds */
    inline int dim() const { return m_direction.size(); }

    const VectorType& origin() const { return m_origin; }
    VectorType& origin() { return m_origin; }

    const VectorType& direction() const { return m_direction; }
    VectorType& direction() { return m_direction; }

    /** \returns the squared distance of a point \a p to its projection onto the line \c *this.
      * \sa distance()
      */
    RealScalar squaredDistance(const VectorType& p) const
    {
      VectorType diff = p-origin();
      return (diff - diff.dot(direction())* direction()).norm2();
    }
    /** \returns the distance of a point \a p to its projection onto the line \c *this.
      * \sa squaredDistance()
      */
    RealScalar distance(const VectorType& p) const { return ei_sqrt(squaredDistance(p)); }

    /** \returns the projection of a point \a p onto the line \c *this.
      */
    VectorType projection(const VectorType& p) const
    { return origin() + (p-origin()).dot(direction()) * direction(); }

    Scalar intersection(const Hyperplane<_Scalar, _AmbientDim>& hyperplane);

  protected:

    VectorType m_origin, m_direction;
};

/** Construct a parametrized line from a 2D hyperplane
  *
  * \warning the ambient space must have dimension 2 such that the hyperplane actually describes a line
  */
template <typename _Scalar, int _AmbientDim>
inline ParametrizedLine<_Scalar, _AmbientDim>::ParametrizedLine(const Hyperplane<_Scalar, _AmbientDim>& hyperplane)
{
  EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(VectorType, 2);
  direction() = hyperplane.normal().unitOrthogonal();
  origin() = -hyperplane.normal()*hyperplane.offset();
}

/** \returns the parameter value of the intersection between *this and the given hyperplane
  */
template <typename _Scalar, int _AmbientDim>
inline _Scalar ParametrizedLine<_Scalar, _AmbientDim>::intersection(const Hyperplane<_Scalar, _AmbientDim>& hyperplane)
{
  return -(hyperplane.offset()+origin().dot(hyperplane.normal()))
          /(direction().dot(hyperplane.normal()));
}

#endif // EIGEN_PARAMETRIZEDLINE_H
