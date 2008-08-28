// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2008 Gael Guennebaud <g.gael@free.fr>
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

#ifndef EIGEN_HYPERPLANE_H
#define EIGEN_HYPERPLANE_H

/** \geometry_module \ingroup GeometryModule
  *
  * \class HyperPlane
  *
  * \brief Represents an hyper plane in any dimensions
  *
  * \param _Scalar the scalar type, i.e., the type of the coefficients
  * \param _Dim the  dimension of the space, can be a compile time value or Dynamic
  *
  * This class represents an hyper-plane as the zero set of the implicit equation
  * \f$ n \cdot x + d = 0 \f$ where \f$ n \f$ is the normal of the plane (linear part)
  * and \f$ d \f$ is the distance (offset) to the origin.
  *
  */

// FIXME default to 3 (because plane => dim=3, or default to Dynamic ?)
template <typename _Scalar, int _Dim = 3>
class HyperPlane
{

  public:

    enum { DimAtCompileTime = _Dim };
    typedef _Scalar Scalar;
    typedef typename NumTraits<Scalar>::Real RealScalar;
    typedef Matrix<Scalar,DimAtCompileTime,1> VectorType;

    HyperPlane(int _dim = DimAtCompileTime)
      : m_normal(_dim)
    {}
    
    /** Construct a plane from its normal \a normal and a point \a e onto the plane.
      * \warning the vector normal is assumed to be normalized.
      */
    HyperPlane(const VectorType& normal, const VectorType e)
        : m_normal(normal), m_offset(-e.dot(normal))
    {}
    
    /** Constructs a plane from its normal \a normal and distance to the origin \a d.
      * \warning the vector normal is assumed to be normalized.
      */
    HyperPlane(const VectorType& normal, Scalar d)
      : m_normal(normal), m_offset(d)
    {}
    
    ~HyperPlane() {}

    /** \returns the dimension in which the plane holds */
    int dim() const { return m_normal.size(); }

    /** normalizes \c *this */
    void normalize(void)
    {
      RealScalar l = Scalar(1)/m_normal.norm();
      m_normal *= l;
      m_offset *= l;
    }
    
    /** \returns the signed distance between the plane \c *this and a point \a p.
      */
    inline Scalar distanceTo(const VectorType& p) const
    {
      return p.dot(m_normal) + m_offset;
    }
    
    /** \returns the projection of a point \a p onto the plane \c *this.
      */
    inline VectorType project(const VectorType& p) const
    {
      return p - distanceTo(p) * m_normal;
    }

    /**  \returns the normal of the plane, which corresponds to the linear part of the implicit equation. */
    inline const VectorType& normal(void) const { return m_normal; }

    /** \returns the distance to the origin, which is also the constant part
      * of the implicit equation */
    inline Scalar offset(void) const { return m_offset; }
    
    /** Set the normal of the plane.
      * \warning the vector normal is assumed to be normalized. */
    inline void setNormal(const VectorType& normal) { m_normal = normal; }

    /** Set the distance to origin */
    inline void setOffset(Scalar d) { m_offset = d; }

    /** \returns a pointer the coefficients c_i of the plane equation:
      *  c_0*x_0 + ... + c_d-1*x_d-1 + offset = 0
      * \warning this is only for fixed size dimensions !
      */
    inline const Scalar* equation(void) const { return m_normal.data(); }
    
    /** \brief Plane/ray intersection.
        Returns the parameter value of the intersection between the plane \a *this
        and the parametric ray of origin \a rayOrigin and axis \a rayDir
    */
    Scalar rayIntersection(const VectorType& rayOrigin, const VectorType& rayDir)
    {
      return -(m_offset+rayOrigin.dot(m_normal))/(rayDir.dot(m_normal));
    }

    // TODO some convenient functions to fit a 3D plane on 3 points etc...
//     void makePassBy(const VectorType& p0, const VectorType& p1, const VectorType& p2)
//     {
//       EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(3);
//       m_normal = (p2 - p0).cross(p1 - p0).normalized();
//       m_offset = -m_normal.dot(p0);
//     }
// 
//     void makePassBy(const VectorType& p0, const VectorType& p1)
//     {
//       EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(2);
//       m_normal = (p2 - p0).cross(p1 - p0).normalized();
//       m_offset = -m_normal.dot(p0);
//     }

protected:

    VectorType m_normal;
    Scalar m_offset;
};

/** \addtogroup GeometryModule */
//@{
typedef HyperPlane<float, 2> HyperPlane2f;
typedef HyperPlane<double,2> HyperPlane2d;
typedef HyperPlane<float, 3> HyperPlane3f;
typedef HyperPlane<double,3> HyperPlane3d;

typedef HyperPlane<float, 2> Linef;
typedef HyperPlane<double,2> Lined;
typedef HyperPlane<float, 3> Planef;
typedef HyperPlane<double,3> Planed;

typedef HyperPlane<float, Dynamic> HyperPlaneXf;
typedef HyperPlane<double,Dynamic> HyperPlaneXd;
//@}

#endif // EIGEN_HYPERPLANE_H
