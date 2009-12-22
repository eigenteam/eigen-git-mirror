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

#ifndef EIGEN_SKYLINE_STORAGE_H
#define EIGEN_SKYLINE_STORAGE_H

/** Stores a skyline set of values in three structures :
 * The diagonal elements
 * The upper elements
 * The lower elements
 *
 */
template<typename Scalar>
class SkylineStorage {
    typedef typename NumTraits<Scalar>::Real RealScalar;
public:

    SkylineStorage()
    : m_diag(0),
    m_lower(0),
    m_upper(0),
    m_lowerProfile(0),
    m_upperProfile(0),
    m_diagSize(0),
    m_upperSize(0),
    m_lowerSize(0),
    m_upperProfileSize(0),
    m_lowerProfileSize(0),
    m_allocatedSize(0) {
    }

    SkylineStorage(const SkylineStorage& other)
    : m_diag(0),
    m_lower(0),
    m_upper(0),
    m_lowerProfile(0),
    m_upperProfile(0),
    m_diagSize(0),
    m_upperSize(0),
    m_lowerSize(0),
    m_upperProfileSize(0),
    m_lowerProfileSize(0),
    m_allocatedSize(0) {
        *this = other;
    }

    SkylineStorage & operator=(const SkylineStorage& other) {
        resize(other.diagSize(), other.m_upperProfileSize, other.m_lowerProfileSize, other.upperSize(), other.lowerSize());
        memcpy(m_diag, other.m_diag, m_diagSize * sizeof (Scalar));
        memcpy(m_upper, other.m_upper, other.upperSize() * sizeof (Scalar));
        memcpy(m_lower, other.m_lower, other.lowerSize() * sizeof (Scalar));
        memcpy(m_upperProfile, other.m_upperProfile, m_upperProfileSize * sizeof (int));
        memcpy(m_lowerProfile, other.m_lowerProfile, m_lowerProfileSize * sizeof (int));
        return *this;
    }

    void swap(SkylineStorage& other) {
        std::swap(m_diag, other.m_diag);
        std::swap(m_upper, other.m_upper);
        std::swap(m_lower, other.m_lower);
        std::swap(m_upperProfile, other.m_upperProfile);
        std::swap(m_lowerProfile, other.m_lowerProfile);
        std::swap(m_diagSize, other.m_diagSize);
        std::swap(m_upperSize, other.m_upperSize);
        std::swap(m_lowerSize, other.m_lowerSize);
        std::swap(m_allocatedSize, other.m_allocatedSize);
    }

    ~SkylineStorage() {
        delete[] m_diag;
        delete[] m_upper;
        if (m_upper != m_lower)
            delete[] m_lower;
        delete[] m_upperProfile;
        delete[] m_lowerProfile;
    }

    void reserve(size_t size, size_t upperProfileSize, size_t lowerProfileSize, size_t upperSize, size_t lowerSize) {
        int newAllocatedSize = size + upperSize + lowerSize;
        if (newAllocatedSize > m_allocatedSize)
            reallocate(size, upperProfileSize, lowerProfileSize, upperSize, lowerSize);
    }

    void squeeze() {
        if (m_allocatedSize > m_diagSize + m_upperSize + m_lowerSize)
            reallocate(m_diagSize, m_upperProfileSize, m_lowerProfileSize, m_upperSize, m_lowerSize);
    }

    void resize(size_t diagSize, size_t upperProfileSize, size_t lowerProfileSize, size_t upperSize, size_t lowerSize, float reserveSizeFactor = 0) {
        if (m_allocatedSize < diagSize + upperSize + lowerSize)
            reallocate(diagSize, upperProfileSize, lowerProfileSize, upperSize + size_t(reserveSizeFactor * upperSize), lowerSize + size_t(reserveSizeFactor * lowerSize));
        m_diagSize = diagSize;
        m_upperSize = upperSize;
        m_lowerSize = lowerSize;
        m_upperProfileSize = upperProfileSize;
        m_lowerProfileSize = lowerProfileSize;
    }

    inline size_t diagSize() const {
        return m_diagSize;
    }

    inline size_t upperSize() const {
        return m_upperSize;
    }

    inline size_t lowerSize() const {
        return m_lowerSize;
    }

    inline size_t upperProfileSize() const {
        return m_upperProfileSize;
    }

    inline size_t lowerProfileSize() const {
        return m_lowerProfileSize;
    }

    inline size_t allocatedSize() const {
        return m_allocatedSize;
    }

    inline void clear() {
        m_diagSize = 0;
    }

    inline Scalar& diag(size_t i) {
        return m_diag[i];
    }

    inline const Scalar& diag(size_t i) const {
        return m_diag[i];
    }

    inline Scalar& upper(size_t i) {
        return m_upper[i];
    }

    inline const Scalar& upper(size_t i) const {
        return m_upper[i];
    }

    inline Scalar& lower(size_t i) {
        return m_lower[i];
    }

    inline const Scalar& lower(size_t i) const {
        return m_lower[i];
    }

    inline int& upperProfile(size_t i) {
        return m_upperProfile[i];
    }

    inline const int& upperProfile(size_t i) const {
        return m_upperProfile[i];
    }

    inline int& lowerProfile(size_t i) {
        return m_lowerProfile[i];
    }

    inline const int& lowerProfile(size_t i) const {
        return m_lowerProfile[i];
    }

    static SkylineStorage Map(int* upperProfile, int* lowerProfile, Scalar* diag, Scalar* upper, Scalar* lower, size_t size, size_t upperSize, size_t lowerSize) {
        SkylineStorage res;
        res.m_upperProfile = upperProfile;
        res.m_lowerProfile = lowerProfile;
        res.m_diag = diag;
        res.m_upper = upper;
        res.m_lower = lower;
        res.m_allocatedSize = res.m_diagSize = size;
        res.m_upperSize = upperSize;
        res.m_lowerSize = lowerSize;
        return res;
    }

    inline void reset() {
        memset(m_diag, 0, m_diagSize * sizeof (Scalar));
        memset(m_upper, 0, m_upperSize * sizeof (Scalar));
        memset(m_lower, 0, m_lowerSize * sizeof (Scalar));
        memset(m_upperProfile, 0, m_diagSize * sizeof (int));
        memset(m_lowerProfile, 0, m_diagSize * sizeof (int));
    }

    void prune(Scalar reference, RealScalar epsilon = dummy_precision<RealScalar>()) {
        //TODO
    }

protected:

    inline void reallocate(size_t diagSize, size_t upperProfileSize, size_t lowerProfileSize, size_t upperSize, size_t lowerSize) {

        Scalar* diag = new Scalar[diagSize];
        Scalar* upper = new Scalar[upperSize];
        Scalar* lower = new Scalar[lowerSize];
        int* upperProfile = new int[upperProfileSize];
        int* lowerProfile = new int[lowerProfileSize];

        size_t copyDiagSize = std::min(diagSize, m_diagSize);
        size_t copyUpperSize = std::min(upperSize, m_upperSize);
        size_t copyLowerSize = std::min(lowerSize, m_lowerSize);
        size_t copyUpperProfileSize = std::min(upperProfileSize, m_upperProfileSize);
        size_t copyLowerProfileSize = std::min(lowerProfileSize, m_lowerProfileSize);

        // copy
        memcpy(diag, m_diag, copyDiagSize * sizeof (Scalar));
        memcpy(upper, m_upper, copyUpperSize * sizeof (Scalar));
        memcpy(lower, m_lower, copyLowerSize * sizeof (Scalar));
        memcpy(upperProfile, m_upperProfile, copyUpperProfileSize * sizeof (int));
        memcpy(lowerProfile, m_lowerProfile, copyLowerProfileSize * sizeof (int));



        // delete old stuff
        delete[] m_diag;
        delete[] m_upper;
        delete[] m_lower;
        delete[] m_upperProfile;
        delete[] m_lowerProfile;
        m_diag = diag;
        m_upper = upper;
        m_lower = lower;
        m_upperProfile = upperProfile;
        m_lowerProfile = lowerProfile;
        m_allocatedSize = diagSize + upperSize + lowerSize;
        m_upperSize = upperSize;
        m_lowerSize = lowerSize;
    }

public:
    Scalar* m_diag;
    Scalar* m_upper;
    Scalar* m_lower;
    int* m_upperProfile;
    int* m_lowerProfile;
    size_t m_diagSize;
    size_t m_upperSize;
    size_t m_lowerSize;
    size_t m_upperProfileSize;
    size_t m_lowerProfileSize;
    size_t m_allocatedSize;

};

#endif // EIGEN_COMPRESSED_STORAGE_H
