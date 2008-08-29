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

#ifndef EIGEN_Hyperplane_H
#define EIGEN_Hyperplane_H

/** \geometry_module \ingroup GeometryModule
  *
  * \class Hyperplane
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

template <typename _Scalar, int _Dim>
class Hyperplane
{

  public:

    enum { DimAtCompileTime = _Dim };
    typedef _Scalar Scalar;
    typedef typename NumTraits<Scalar>::Real RealScalar;
    typedef Matrix<Scalar,DimAtCompileTime,1> VectorType;
    typedef Matrix<Scalar,DimAtCompileTime==Dynamic
                          ? Dynamic
                          : DimAtCompileTime+1,1> Coefficients;
    typedef Block<Coefficients,DimAtCompileTime,1> NormalReturnType;

    /** Default constructor without initialization */
    inline Hyperplane(int _dim = DimAtCompileTime) : m_coeffs(_dim+1) {}
    
    /** Construct a plane from its normal \a n and a point \a e onto the plane.
      * \warning the vector normal is assumed to be normalized.
      */
    inline Hyperplane(const VectorType& n, const VectorType e)
      : m_coeffs(n.size()+1)
    {
      _normal() = n;
      _offset() = -e.dot(n);
    }
    
    /** Constructs a plane from its normal \a n and distance to the origin \a d.
      * \warning the vector normal is assumed to be normalized.
      */
    inline Hyperplane(const VectorType& n, Scalar d)
      : m_coeffs(n.size()+1)
    {
      _normal() = n;
      _offset() = d;
    }
    
    ~Hyperplane() {}

    /** \returns the dimension in which the plane holds */
    inline int dim() const { return DimAtCompileTime==Dynamic ? m_coeffs.size()-1 : DimAtCompileTime; }

    void normalize(void);
    
    /** \returns the signed distance between the plane \c *this and a point \a p.
      */
    inline Scalar distanceTo(const VectorType& p) const { return p.dot(normal()) + offset(); }
    
    /** \returns the projection of a point \a p onto the plane \c *this.
      */
    inline VectorType project(const VectorType& p) const { return p - distanceTo(p) * normal(); }

    /**  \returns the normal of the plane, which corresponds to the linear part of the implicit equation. */
    inline const NormalReturnType normal() const { return NormalReturnType(m_coeffs,0,0,dim(),1); }

    /** \returns the distance to the origin, which is also the constant part
      * of the implicit equation */
    inline Scalar offset() const { return m_coeffs(dim()); }
    
    /** Set the normal of the plane.
      * \warning the vector normal is assumed to be normalized. */
    inline void setNormal(const VectorType& normal) { _normal() = normal; }

    /** Set the distance to origin */
    inline void setOffset(Scalar d) { _offset() = d; }

    /** \returns the coefficients c_i of the plane equation:
      * \f$ c_0*x_0 + ... + c_{d-1}*x_{d-1} + c_d = 0 \f$
      */
    // FIXME name: equation vs coeffs vs coefficients ???
    inline Coefficients equation(void) const { return m_coeffs; }
    
    /** \brief Plane/ray intersection.
        Returns the parameter value of the intersection between the plane \a *this
        and the parametric ray of origin \a rayOrigin and axis \a rayDir
    */
    inline Scalar rayIntersection(const VectorType& rayOrigin, const VectorType& rayDir)
    {
      return -(_offset()+rayOrigin.dot(_normal()))/(rayDir.dot(_normal()));
    }

    template<typename XprType>
    inline Hyperplane operator* (const MatrixBase<XprType>& mat) const
    { return Hyperplane(mat.inverse().transpose() * _normal(), _offset()); }

    template<typename XprType>
    inline Hyperplane& operator*= (const MatrixBase<XprType>& mat) const
    { _normal() = mat.inverse().transpose() * _normal(); return *this; }

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

    inline NormalReturnType _normal() { return NormalReturnType(m_coeffs,0,0,dim(),1); }
    inline Scalar& _offset() { return m_coeffs(dim()); }

    Coefficients m_coeffs;
};

/** \addtogroup GeometryModule */
//@{
typedef Hyperplane<float, 2> Hyperplane2f;
typedef Hyperplane<double,2> Hyperplane2d;
typedef Hyperplane<float, 3> Hyperplane3f;
typedef Hyperplane<double,3> Hyperplane3d;

typedef Hyperplane<float, 3> Planef;
typedef Hyperplane<double,3> Planed;

typedef Hyperplane<float, Dynamic> HyperplaneXf;
typedef Hyperplane<double,Dynamic> HyperplaneXd;
//@}

/** normalizes \c *this */
template <typename _Scalar, int _Dim>
void Hyperplane<_Scalar,_Dim>::normalize(void)
{
  RealScalar l = Scalar(1)/_normal().norm();
  _normal() *= l;
  _offset() *= l;
}

#endif // EIGEN_Hyperplane_H
