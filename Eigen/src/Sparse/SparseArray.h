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

#ifndef EIGEN_SPARSE_ARRAY_H
#define EIGEN_SPARSE_ARRAY_H

/** Stores a sparse set of values as a list of values and a list of indices.
  *
  */
template<typename Scalar> class SparseArray
{
  public:
    SparseArray()
      : m_values(0), m_indices(0), m_size(0), m_allocatedSize(0)
    {}

    SparseArray(int size)
      : m_values(0), m_indices(0), m_size(0), m_allocatedSize(0)
    {
      resize(size);
    }

    SparseArray(const SparseArray& other)
    {
      *this = other;
    }

    SparseArray& operator=(const SparseArray& other)
    {
      resize(other.size());
      memcpy(m_values, other.m_values, m_size * sizeof(Scalar));
      memcpy(m_indices, other.m_indices, m_size * sizeof(int));
    }

    void reserve(int size)
    {
      int newAllocatedSize = m_size + size;
      if (newAllocatedSize > m_allocatedSize)
      {
        Scalar* newValues  = new Scalar[newAllocatedSize];
        int* newIndices = new int[newAllocatedSize];
        // copy
        memcpy(newValues,  m_values,  m_size * sizeof(Scalar));
        memcpy(newIndices, m_indices, m_size * sizeof(int));
        // delete old stuff
        delete[] m_values;
        delete[] m_indices;
        m_values = newValues;
        m_indices = newIndices;
        m_allocatedSize = newAllocatedSize;
      }
    }

    void resize(int size, int reserveSizeFactor = 0)
    {
      if (m_allocatedSize<size)
      {
        int newAllocatedSize = size + reserveSizeFactor*size;
        Scalar* newValues  = new Scalar[newAllocatedSize];
        int* newIndices = new int[newAllocatedSize];
        // copy
        memcpy(newValues,  m_values,  m_size * sizeof(Scalar));
        memcpy(newIndices, m_indices, m_size * sizeof(int));
        // delete old stuff
        delete[] m_values;
        delete[] m_indices;
        m_values = newValues;
        m_indices = newIndices;
        m_allocatedSize = newAllocatedSize;
      }
      m_size = size;
    }

    void append(const Scalar& v, int i)
    {
      int id = m_size;
      resize(m_size+1, 1);
      m_values[id] = v;
      m_indices[id] = i;
    }

    int size() const { return m_size; }

    void clear()
    {
      m_size = 0;
    }

    Scalar& value(int i) { return m_values[i]; }
    Scalar value(int i) const { return m_values[i]; }

    int& index(int i) { return m_indices[i]; }
    int index(int i) const { return m_indices[i]; }

  protected:
    Scalar* m_values;
    int* m_indices;
    int m_size;
    int m_allocatedSize;
};

#endif // EIGEN_SPARSE_ARRAY_H
