// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Guillaume Saupin <guillaume.saupin@cea.fr>
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

#ifndef EIGEN_SKYLINEMATRIX_H
#define EIGEN_SKYLINEMATRIX_H

#include "SkylineStorage.h"
#include "SkylineMatrixBase.h"

/** \ingroup Skyline_Module
 *
 * \class SkylineMatrix
 *
 * \brief The main skyline matrix class
 *
 * This class implements a skyline matrix using the very uncommon storage
 * scheme.
 *
 * \param _Scalar the scalar type, i.e. the type of the coefficients
 * \param _Options Union of bit flags controlling the storage scheme. Currently the only possibility
 *                 is RowMajor. The default is 0 which means column-major.
 *
 *
 */
template<typename _Scalar, int _Options>
struct ei_traits<SkylineMatrix<_Scalar, _Options> > {
    typedef _Scalar Scalar;

    enum {
        RowsAtCompileTime = Dynamic,
        ColsAtCompileTime = Dynamic,
        MaxRowsAtCompileTime = Dynamic,
        MaxColsAtCompileTime = Dynamic,
        Flags = SkylineBit | _Options,
        CoeffReadCost = NumTraits<Scalar>::ReadCost,
    };
};

template<typename _Scalar, int _Options>
class SkylineMatrix
: public SkylineMatrixBase<SkylineMatrix<_Scalar, _Options> > {
public:
    EIGEN_SKYLINE_GENERIC_PUBLIC_INTERFACE(SkylineMatrix)
    EIGEN_SKYLINE_INHERIT_ASSIGNMENT_OPERATOR(SkylineMatrix, +=)
    EIGEN_SKYLINE_INHERIT_ASSIGNMENT_OPERATOR(SkylineMatrix, -=)

    using Base::IsRowMajor;

protected:

    typedef SkylineMatrix<Scalar, (Flags&~RowMajorBit) | (IsRowMajor ? RowMajorBit : 0) > TransposedSkylineMatrix;

    int m_outerSize;
    int m_innerSize;

public:
    int* m_colStartIndex;
    int* m_rowStartIndex;
    SkylineStorage<Scalar> m_data;

public:

    inline int rows() const {
        return IsRowMajor ? m_outerSize : m_innerSize;
    }

    inline int cols() const {
        return IsRowMajor ? m_innerSize : m_outerSize;
    }

    inline int innerSize() const {
        return m_innerSize;
    }

    inline int outerSize() const {
        return m_outerSize;
    }

    inline int upperNonZeros() const {
        return m_data.upperSize();
    }

    inline int lowerNonZeros() const {
        return m_data.lowerSize();
    }

    inline int upperNonZeros(int j) const {
        return m_colStartIndex[j + 1] - m_colStartIndex[j];
    }

    inline int lowerNonZeros(int j) const {
        return m_rowStartIndex[j + 1] - m_rowStartIndex[j];
    }

    inline const Scalar* _diagPtr() const {
        return &m_data.diag(0);
    }

    inline Scalar* _diagPtr() {
        return &m_data.diag(0);
    }

    inline const Scalar* _upperPtr() const {
        return &m_data.upper(0);
    }

    inline Scalar* _upperPtr() {
        return &m_data.upper(0);
    }

    inline const Scalar* _lowerPtr() const {
        return &m_data.lower(0);
    }

    inline Scalar* _lowerPtr() {
        return &m_data.lower(0);
    }

    inline const int* _upperProfilePtr() const {
        return &m_data.upperProfile(0);
    }

    inline int* _upperProfilePtr() {
        return &m_data.upperProfile(0);
    }

    inline const int* _lowerProfilePtr() const {
        return &m_data.lowerProfile(0);
    }

    inline int* _lowerProfilePtr() {
        return &m_data.lowerProfile(0);
    }

    inline Scalar coeff(int row, int col) const {
        const int outer = IsRowMajor ? row : col;
        const int inner = IsRowMajor ? col : row;

        ei_assert(outer < outerSize());
        ei_assert(inner < innerSize());

        if (outer == inner)
            return this->m_data.diag(outer);

        if (IsRowMajor) {
            if (inner > outer) //upper matrix
            {
                const int minOuterIndex = inner - m_data.upperProfile(inner);
                if (outer >= minOuterIndex)
                    return this->m_data.upper(m_colStartIndex[inner] + outer - (inner - m_data.upperProfile(inner)));
                else
                    return Scalar(0);
            }
            if (inner < outer) //lower matrix
            {
                const int minInnerIndex = outer - m_data.lowerProfile(outer);
                if (inner >= minInnerIndex)
                    return this->m_data.lower(m_rowStartIndex[outer] + inner - (outer - m_data.lowerProfile(outer)));
                else
                    return Scalar(0);
            }
            return m_data.upper(m_colStartIndex[inner] + outer - inner);
        } else {
            if (outer > inner) //upper matrix
            {
                const int maxOuterIndex = inner + m_data.upperProfile(inner);
                if (outer <= maxOuterIndex)
                    return this->m_data.upper(m_colStartIndex[inner] + (outer - inner));
                else
                    return Scalar(0);
            }
            if (outer < inner) //lower matrix
            {
                const int maxInnerIndex = outer + m_data.lowerProfile(outer);

                if (inner <= maxInnerIndex)
                    return this->m_data.lower(m_rowStartIndex[outer] + (inner - outer));
                else
                    return Scalar(0);
            }
        }
    }

    inline Scalar& coeffRef(int row, int col) {
        const int outer = IsRowMajor ? row : col;
        const int inner = IsRowMajor ? col : row;

        ei_assert(outer < outerSize());
        ei_assert(inner < innerSize());

        if (outer == inner)
            return this->m_data.diag(outer);

        if (IsRowMajor) {
            if (col > row) //upper matrix
            {
                const int minOuterIndex = inner - m_data.upperProfile(inner);
                ei_assert(outer >= minOuterIndex && "you try to acces a coeff that do not exist in the storage");
                return this->m_data.upper(m_colStartIndex[inner] + outer - (inner - m_data.upperProfile(inner)));
            }
            if (col < row) //lower matrix
            {
                const int minInnerIndex = outer - m_data.lowerProfile(outer);
                ei_assert(inner >= minInnerIndex && "you try to acces a coeff that do not exist in the storage");
                return this->m_data.lower(m_rowStartIndex[outer] + inner - (outer - m_data.lowerProfile(outer)));
            }
        } else {
            if (outer > inner) //upper matrix
            {
                const int maxOuterIndex = inner + m_data.upperProfile(inner);
                ei_assert(outer <= maxOuterIndex && "you try to acces a coeff that do not exist in the storage");
                return this->m_data.upper(m_colStartIndex[inner] + (outer - inner));
            }
            if (outer < inner) //lower matrix
            {
                const int maxInnerIndex = outer + m_data.lowerProfile(outer);
                ei_assert(inner <= maxInnerIndex && "you try to acces a coeff that do not exist in the storage");
                return this->m_data.lower(m_rowStartIndex[outer] + (inner - outer));
            }
        }
    }

    inline Scalar coeffDiag(int idx) const {
        ei_assert(idx < outerSize());
        ei_assert(idx < innerSize());
        return this->m_data.diag(idx);
    }

    inline Scalar coeffLower(int row, int col) const {
        const int outer = IsRowMajor ? row : col;
        const int inner = IsRowMajor ? col : row;

        ei_assert(outer < outerSize());
        ei_assert(inner < innerSize());
        ei_assert(inner != outer);

        if (IsRowMajor) {
            const int minInnerIndex = outer - m_data.lowerProfile(outer);
            if (inner >= minInnerIndex)
                return this->m_data.lower(m_rowStartIndex[outer] + inner - (outer - m_data.lowerProfile(outer)));
            else
                return Scalar(0);

        } else {
            const int maxInnerIndex = outer + m_data.lowerProfile(outer);
            if (inner <= maxInnerIndex)
                return this->m_data.lower(m_rowStartIndex[outer] + (inner - outer));
            else
                return Scalar(0);
        }
    }

    inline Scalar coeffUpper(int row, int col) const {
        const int outer = IsRowMajor ? row : col;
        const int inner = IsRowMajor ? col : row;

        ei_assert(outer < outerSize());
        ei_assert(inner < innerSize());
        ei_assert(inner != outer);

        if (IsRowMajor) {
            const int minOuterIndex = inner - m_data.upperProfile(inner);
            if (outer >= minOuterIndex)
                return this->m_data.upper(m_colStartIndex[inner] + outer - (inner - m_data.upperProfile(inner)));
            else
                return Scalar(0);
        } else {
            const int maxOuterIndex = inner + m_data.upperProfile(inner);
            if (outer <= maxOuterIndex)
                return this->m_data.upper(m_colStartIndex[inner] + (outer - inner));
            else
                return Scalar(0);
        }
    }

    inline Scalar& coeffRefDiag(int idx) {
        ei_assert(idx < outerSize());
        ei_assert(idx < innerSize());
        return this->m_data.diag(idx);
    }

    inline Scalar& coeffRefLower(int row, int col) {
        const int outer = IsRowMajor ? row : col;
        const int inner = IsRowMajor ? col : row;

        ei_assert(outer < outerSize());
        ei_assert(inner < innerSize());
        ei_assert(inner != outer);

        if (IsRowMajor) {
            const int minInnerIndex = outer - m_data.lowerProfile(outer);
            ei_assert(inner >= minInnerIndex && "you try to acces a coeff that do not exist in the storage");
            return this->m_data.lower(m_rowStartIndex[outer] + inner - (outer - m_data.lowerProfile(outer)));
        } else {
            const int maxInnerIndex = outer + m_data.lowerProfile(outer);
            ei_assert(inner <= maxInnerIndex && "you try to acces a coeff that do not exist in the storage");
            return this->m_data.lower(m_rowStartIndex[outer] + (inner - outer));
        }
    }

    inline bool coeffExistLower(int row, int col) {
        const int outer = IsRowMajor ? row : col;
        const int inner = IsRowMajor ? col : row;

        ei_assert(outer < outerSize());
        ei_assert(inner < innerSize());
        ei_assert(inner != outer);

        if (IsRowMajor) {
            const int minInnerIndex = outer - m_data.lowerProfile(outer);
            return inner >= minInnerIndex;
        } else {
            const int maxInnerIndex = outer + m_data.lowerProfile(outer);
            return inner <= maxInnerIndex;
        }
    }

    inline Scalar& coeffRefUpper(int row, int col) {
        const int outer = IsRowMajor ? row : col;
        const int inner = IsRowMajor ? col : row;

        ei_assert(outer < outerSize());
        ei_assert(inner < innerSize());
        ei_assert(inner != outer);

        if (IsRowMajor) {
            const int minOuterIndex = inner - m_data.upperProfile(inner);
            ei_assert(outer >= minOuterIndex && "you try to acces a coeff that do not exist in the storage");
            return this->m_data.upper(m_colStartIndex[inner] + outer - (inner - m_data.upperProfile(inner)));
        } else {
            const int maxOuterIndex = inner + m_data.upperProfile(inner);
            ei_assert(outer <= maxOuterIndex && "you try to acces a coeff that do not exist in the storage");
            return this->m_data.upper(m_colStartIndex[inner] + (outer - inner));
        }
    }

    inline bool coeffExistUpper(int row, int col) {
        const int outer = IsRowMajor ? row : col;
        const int inner = IsRowMajor ? col : row;

        ei_assert(outer < outerSize());
        ei_assert(inner < innerSize());
        ei_assert(inner != outer);

        if (IsRowMajor) {
            const int minOuterIndex = inner - m_data.upperProfile(inner);
            return outer >= minOuterIndex;
        } else {
            const int maxOuterIndex = inner + m_data.upperProfile(inner);
            return outer <= maxOuterIndex;
        }
    }


protected:

public:
    class InnerUpperIterator;
    class InnerLowerIterator;

    class OuterUpperIterator;
    class OuterLowerIterator;

    /** Removes all non zeros */
    inline void setZero() {
        m_data.clear();
        memset(m_colStartIndex, 0, (m_outerSize + 1) * sizeof (int));
        memset(m_rowStartIndex, 0, (m_outerSize + 1) * sizeof (int));
    }

    /** \returns the number of non zero coefficients */
    inline int nonZeros() const {
        return m_data.diagSize() + m_data.upperSize() + m_data.lowerSize();
    }

    /** Preallocates \a reserveSize non zeros */
    inline void reserve(int reserveSize, int reserveUpperSize, int reserveLowerSize) {
        m_data.reserve(reserveSize, reserveUpperSize, reserveLowerSize);
    }

    /** \returns a reference to a novel non zero coefficient with coordinates \a row x \a col.

     *
     * \warning This function can be extremely slow if the non zero coefficients
     * are not inserted in a coherent order.
     *
     * After an insertion session, you should call the finalize() function.
     */
    EIGEN_DONT_INLINE Scalar & insert(int row, int col) {
        const int outer = IsRowMajor ? row : col;
        const int inner = IsRowMajor ? col : row;

        ei_assert(outer < outerSize());
        ei_assert(inner < innerSize());

        if (outer == inner)
            return m_data.diag(col);

        if (IsRowMajor) {
            if (outer < inner) //upper matrix
            {
                int minOuterIndex = 0;
                minOuterIndex = inner - m_data.upperProfile(inner);

                if (outer < minOuterIndex) //The value does not yet exist
                {
                    const int previousProfile = m_data.upperProfile(inner);

                    m_data.upperProfile(inner) = inner - outer;


                    const int bandIncrement = m_data.upperProfile(inner) - previousProfile;
                    //shift data stored after this new one
                    const int stop = m_colStartIndex[cols()];
                    const int start = m_colStartIndex[inner];


                    for (int innerIdx = stop; innerIdx >= start; innerIdx--) {
                        m_data.upper(innerIdx + bandIncrement) = m_data.upper(innerIdx);
                    }

                    for (int innerIdx = cols(); innerIdx > inner; innerIdx--) {
                        m_colStartIndex[innerIdx] += bandIncrement;
                    }

                    //zeros new data
                    memset(this->_upperPtr() + start, 0, (bandIncrement - 1) * sizeof (Scalar));

                    return m_data.upper(m_colStartIndex[inner]);
                } else {
                    return m_data.upper(m_colStartIndex[inner] + outer - (inner - m_data.upperProfile(inner)));
                }
            }

            if (outer > inner) //lower matrix
            {
                const int minInnerIndex = outer - m_data.lowerProfile(outer);
                if (inner < minInnerIndex) //The value does not yet exist
                {
                    const int previousProfile = m_data.lowerProfile(outer);
                    m_data.lowerProfile(outer) = outer - inner;

                    const int bandIncrement = m_data.lowerProfile(outer) - previousProfile;
                    //shift data stored after this new one
                    const int stop = m_rowStartIndex[rows()];
                    const int start = m_rowStartIndex[outer];


                    for (int innerIdx = stop; innerIdx >= start; innerIdx--) {
                        m_data.lower(innerIdx + bandIncrement) = m_data.lower(innerIdx);
                    }

                    for (int innerIdx = rows(); innerIdx > outer; innerIdx--) {
                        m_rowStartIndex[innerIdx] += bandIncrement;
                    }

                    //zeros new data
                    memset(this->_lowerPtr() + start, 0, (bandIncrement - 1) * sizeof (Scalar));
                    return m_data.lower(m_rowStartIndex[outer]);
                } else {
                    return m_data.lower(m_rowStartIndex[outer] + inner - (outer - m_data.lowerProfile(outer)));
                }
            }
        } else {
            if (outer > inner) //upper matrix
            {
                const int maxOuterIndex = inner + m_data.upperProfile(inner);
                if (outer > maxOuterIndex) //The value does not yet exist
                {
                    const int previousProfile = m_data.upperProfile(inner);
                    m_data.upperProfile(inner) = outer - inner;

                    const int bandIncrement = m_data.upperProfile(inner) - previousProfile;
                    //shift data stored after this new one
                    const int stop = m_rowStartIndex[rows()];
                    const int start = m_rowStartIndex[inner + 1];

                    for (int innerIdx = stop; innerIdx >= start; innerIdx--) {
                        m_data.upper(innerIdx + bandIncrement) = m_data.upper(innerIdx);
                    }

                    for (int innerIdx = inner + 1; innerIdx < outerSize() + 1; innerIdx++) {
                        m_rowStartIndex[innerIdx] += bandIncrement;
                    }
                    memset(this->_upperPtr() + m_rowStartIndex[inner] + previousProfile + 1, 0, (bandIncrement - 1) * sizeof (Scalar));
                    return m_data.upper(m_rowStartIndex[inner] + m_data.upperProfile(inner));
                } else {
                    return m_data.upper(m_rowStartIndex[inner] + (outer - inner));
                }
            }

            if (outer < inner) //lower matrix
            {
                const int maxInnerIndex = outer + m_data.lowerProfile(outer);
                if (inner > maxInnerIndex) //The value does not yet exist
                {
                    const int previousProfile = m_data.lowerProfile(outer);
                    m_data.lowerProfile(outer) = inner - outer;

                    const int bandIncrement = m_data.lowerProfile(outer) - previousProfile;
                    //shift data stored after this new one
                    const int stop = m_colStartIndex[cols()];
                    const int start = m_colStartIndex[outer + 1];

                    for (int innerIdx = stop; innerIdx >= start; innerIdx--) {
                        m_data.lower(innerIdx + bandIncrement) = m_data.lower(innerIdx);
                    }

                    for (int innerIdx = outer + 1; innerIdx < outerSize() + 1; innerIdx++) {
                        m_colStartIndex[innerIdx] += bandIncrement;
                    }
                    memset(this->_lowerPtr() + m_colStartIndex[outer] + previousProfile + 1, 0, (bandIncrement - 1) * sizeof (Scalar));
                    return m_data.lower(m_colStartIndex[outer] + m_data.lowerProfile(outer));
                } else {
                    return m_data.lower(m_colStartIndex[outer] + (inner - outer));
                }
            }
        }
    }

    /** Must be called after inserting a set of non zero entries.
     */
    inline void finalize() {
        if (IsRowMajor) {
            if (rows() > cols())
                m_data.resize(cols(), cols(), rows(), m_colStartIndex[cols()] + 1, m_rowStartIndex[rows()] + 1);
            else
                m_data.resize(rows(), cols(), rows(), m_colStartIndex[cols()] + 1, m_rowStartIndex[rows()] + 1);

            //            ei_assert(rows() == cols() && "memory reorganisatrion only works with suare matrix");
            //
            //            Scalar* newArray = new Scalar[m_colStartIndex[cols()] + 1 + m_rowStartIndex[rows()] + 1];
            //            unsigned int dataIdx = 0;
            //            for (unsigned int row = 0; row < rows(); row++) {
            //
            //                const unsigned int nbLowerElts = m_rowStartIndex[row + 1] - m_rowStartIndex[row];
            //                //                std::cout << "nbLowerElts" << nbLowerElts << std::endl;
            //                memcpy(newArray + dataIdx, m_data.m_lower + m_rowStartIndex[row], nbLowerElts * sizeof (Scalar));
            //                m_rowStartIndex[row] = dataIdx;
            //                dataIdx += nbLowerElts;
            //
            //                const unsigned int nbUpperElts = m_colStartIndex[row + 1] - m_colStartIndex[row];
            //                memcpy(newArray + dataIdx, m_data.m_upper + m_colStartIndex[row], nbUpperElts * sizeof (Scalar));
            //                m_colStartIndex[row] = dataIdx;
            //                dataIdx += nbUpperElts;
            //
            //
            //            }
            //            //todo : don't access m_data profile directly : add an accessor from SkylineMatrix
            //            m_rowStartIndex[rows()] = m_rowStartIndex[rows()-1] + m_data.lowerProfile(rows()-1);
            //            m_colStartIndex[cols()] = m_colStartIndex[cols()-1] + m_data.upperProfile(cols()-1);
            //
            //            delete[] m_data.m_lower;
            //            delete[] m_data.m_upper;
            //
            //            m_data.m_lower = newArray;
            //            m_data.m_upper = newArray;
        } else {
            if (rows() > cols())
                m_data.resize(cols(), rows(), cols(), m_rowStartIndex[cols()] + 1, m_colStartIndex[cols()] + 1);
            else
                m_data.resize(rows(), rows(), cols(), m_rowStartIndex[rows()] + 1, m_colStartIndex[rows()] + 1);
        }
    }

    inline void squeeze() {
        finalize();
        m_data.squeeze();
    }

    void prune(Scalar reference, RealScalar epsilon = dummy_precision<RealScalar > ()) {
        //TODO
    }

    /** Resizes the matrix to a \a rows x \a cols matrix and initializes it to zero
     * \sa resizeNonZeros(int), reserve(), setZero()
     */
    void resize(size_t rows, size_t cols) {
        const int diagSize = rows > cols ? cols : rows;
        m_innerSize = IsRowMajor ? cols : rows;

        ei_assert(rows == cols && "Skyline matrix must be square matrix");

        if (diagSize % 2) { // diagSize is odd
            const int k = (diagSize - 1) / 2;

            m_data.resize(diagSize, IsRowMajor ? cols : rows, IsRowMajor ? rows : cols,
                    2 * k * k + k + 1,
                    2 * k * k + k + 1);

        } else // diagSize is even
        {
            const int k = diagSize / 2;
            m_data.resize(diagSize, IsRowMajor ? cols : rows, IsRowMajor ? rows : cols,
                    2 * k * k - k + 1,
                    2 * k * k - k + 1);
        }

        if (m_colStartIndex && m_rowStartIndex) {
            delete[] m_colStartIndex;
            delete[] m_rowStartIndex;
        }
        m_colStartIndex = new int [cols + 1];
        m_rowStartIndex = new int [rows + 1];
        m_outerSize = diagSize;

        m_data.reset();
        m_data.clear();

        m_outerSize = diagSize;
        memset(m_colStartIndex, 0, (cols + 1) * sizeof (int));
        memset(m_rowStartIndex, 0, (rows + 1) * sizeof (int));
    }

    void resizeNonZeros(int size) {
        m_data.resize(size);
    }

    inline SkylineMatrix()
    : m_outerSize(-1), m_innerSize(0), m_colStartIndex(0), m_rowStartIndex(0) {
        resize(0, 0);
    }

    inline SkylineMatrix(size_t rows, size_t cols)
    : m_outerSize(0), m_innerSize(0), m_colStartIndex(0), m_rowStartIndex(0) {
        resize(rows, cols);
    }

    template<typename OtherDerived>
    inline SkylineMatrix(const SkylineMatrixBase<OtherDerived>& other)
    : m_outerSize(0), m_innerSize(0), m_colStartIndex(0), m_rowStartIndex(0) {
        *this = other.derived();
    }

    inline SkylineMatrix(const SkylineMatrix & other)
    : Base(), m_outerSize(0), m_innerSize(0), m_colStartIndex(0), m_rowStartIndex(0) {
        *this = other.derived();
    }

    inline void swap(SkylineMatrix & other) {
        //EIGEN_DBG_SKYLINE(std::cout << "SkylineMatrix:: swap\n");
        std::swap(m_colStartIndex, other.m_colStartIndex);
        std::swap(m_rowStartIndex, other.m_rowStartIndex);
        std::swap(m_innerSize, other.m_innerSize);
        std::swap(m_outerSize, other.m_outerSize);
        m_data.swap(other.m_data);
    }

    inline SkylineMatrix & operator=(const SkylineMatrix & other) {
        std::cout << "SkylineMatrix& operator=(const SkylineMatrix& other)\n";
        if (other.isRValue()) {
            swap(other.const_cast_derived());
        } else {
            resize(other.rows(), other.cols());
            memcpy(m_colStartIndex, other.m_colStartIndex, (m_outerSize + 1) * sizeof (int));
            memcpy(m_rowStartIndex, other.m_rowStartIndex, (m_outerSize + 1) * sizeof (int));
            m_data = other.m_data;
        }
        return *this;
    }

    template<typename OtherDerived>
            inline SkylineMatrix & operator=(const SkylineMatrixBase<OtherDerived>& other) {
        const bool needToTranspose = (Flags & RowMajorBit) != (OtherDerived::Flags & RowMajorBit);
        if (needToTranspose) {
            //         TODO
            //            return *this;
        } else {
            // there is no special optimization
            return SkylineMatrixBase<SkylineMatrix>::operator=(other.derived());
        }
    }

    friend std::ostream & operator <<(std::ostream & s, const SkylineMatrix & m) {

        EIGEN_DBG_SKYLINE(
        std::cout << "upper elements : " << std::endl;
        for (unsigned int i = 0; i < m.m_data.upperSize(); i++)
            std::cout << m.m_data.upper(i) << "\t";
        std::cout << std::endl;
        std::cout << "upper profile : " << std::endl;
        for (unsigned int i = 0; i < m.m_data.upperProfileSize(); i++)
            std::cout << m.m_data.upperProfile(i) << "\t";
        std::cout << std::endl;
        std::cout << "lower startIdx : " << std::endl;
        for (unsigned int i = 0; i < m.m_data.upperProfileSize(); i++)
            std::cout << (IsRowMajor ? m.m_colStartIndex[i] : m.m_rowStartIndex[i]) << "\t";
        std::cout << std::endl;


        std::cout << "lower elements : " << std::endl;
        for (unsigned int i = 0; i < m.m_data.lowerSize(); i++)
            std::cout << m.m_data.lower(i) << "\t";
        std::cout << std::endl;
        std::cout << "lower profile : " << std::endl;
        for (unsigned int i = 0; i < m.m_data.lowerProfileSize(); i++)
            std::cout << m.m_data.lowerProfile(i) << "\t";
        std::cout << std::endl;
        std::cout << "lower startIdx : " << std::endl;
        for (unsigned int i = 0; i < m.m_data.lowerProfileSize(); i++)
            std::cout << (IsRowMajor ? m.m_rowStartIndex[i] : m.m_colStartIndex[i]) << "\t";
        std::cout << std::endl;
        );
        for (unsigned int rowIdx = 0; rowIdx < m.rows(); rowIdx++) {
            for (unsigned int colIdx = 0; colIdx < m.cols(); colIdx++) {
                s << m.coeff(rowIdx, colIdx) << "\t";
            }
            s << std::endl;
        }
        return s;
    }

    /** Destructor */
    inline ~SkylineMatrix() {
        delete[] m_colStartIndex;
        delete[] m_rowStartIndex;
    }

    /** Overloaded for performance */
    Scalar sum() const;
};

template<typename Scalar, int _Options>
class SkylineMatrix<Scalar, _Options>::InnerUpperIterator {
public:

    InnerUpperIterator(const SkylineMatrix& mat, int outer)
    : m_matrix(mat), m_outer(outer),
    m_id(_Options == RowMajor ? mat.m_colStartIndex[outer] : mat.m_rowStartIndex[outer] + 1),
    m_start(m_id),
    m_end(_Options == RowMajor ? mat.m_colStartIndex[outer + 1] : mat.m_rowStartIndex[outer + 1] + 1) {
    }

    inline InnerUpperIterator & operator++() {
        m_id++;
        return *this;
    }

    inline InnerUpperIterator & operator+=(unsigned int shift) {
        m_id += shift;
        return *this;
    }

    inline Scalar value() const {
        return m_matrix.m_data.upper(m_id);
    }

    inline Scalar* valuePtr() {
        return const_cast<Scalar*> (&(m_matrix.m_data.upper(m_id)));
    }

    inline Scalar& valueRef() {
        return const_cast<Scalar&> (m_matrix.m_data.upper(m_id));
    }

    inline int index() const {
        return IsRowMajor ? m_outer - m_matrix.m_data.upperProfile(m_outer) + (m_id - m_start) :
                m_outer + (m_id - m_start) + 1;
    }

    inline int row() const {
        return IsRowMajor ? index() : m_outer;
    }

    inline int col() const {
        return IsRowMajor ? m_outer : index();
    }

    inline size_t size() const {
        return m_matrix.m_data.upperProfile(m_outer);
    }

    inline operator bool() const {
        return (m_id < m_end) && (m_id >= m_start);
    }

protected:
    const SkylineMatrix& m_matrix;
    const int m_outer;
    int m_id;
    const int m_start;
    const int m_end;
};

template<typename Scalar, int _Options>
class SkylineMatrix<Scalar, _Options>::InnerLowerIterator {
public:

    InnerLowerIterator(const SkylineMatrix& mat, int outer)
    : m_matrix(mat),
    m_outer(outer),
    m_id(_Options == RowMajor ? mat.m_rowStartIndex[outer] : mat.m_colStartIndex[outer] + 1),
    m_start(m_id),
    m_end(_Options == RowMajor ? mat.m_rowStartIndex[outer + 1] : mat.m_colStartIndex[outer + 1] + 1) {
    }

    inline InnerLowerIterator & operator++() {
        m_id++;
        return *this;
    }

    inline InnerLowerIterator & operator+=(unsigned int shift) {
        m_id += shift;
        return *this;
    }

    inline Scalar value() const {
        return m_matrix.m_data.lower(m_id);
    }

    inline Scalar* valuePtr() {
        return const_cast<Scalar*> (&(m_matrix.m_data.lower(m_id)));
    }

    inline Scalar& valueRef() {
        return const_cast<Scalar&> (m_matrix.m_data.lower(m_id));
    }

    inline int index() const {
        return IsRowMajor ? m_outer - m_matrix.m_data.lowerProfile(m_outer) + (m_id - m_start) :
                m_outer + (m_id - m_start) + 1;
        ;
    }

    inline int row() const {
        return IsRowMajor ? m_outer : index();
    }

    inline int col() const {
        return IsRowMajor ? index() : m_outer;
    }

    inline size_t size() const {
        return m_matrix.m_data.lowerProfile(m_outer);
    }

    inline operator bool() const {
        return (m_id < m_end) && (m_id >= m_start);
    }

protected:
    const SkylineMatrix& m_matrix;
    const int m_outer;
    int m_id;
    const int m_start;
    const int m_end;
};

#endif // EIGEN_SkylineMatrix_H
