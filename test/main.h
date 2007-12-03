// This file is part of Eigen, a lightweight C++ template library
// for linear algebra. Eigen itself is part of the KDE project.
//
// Copyright (C) 2006-2007 Benoit Jacob <jacob@math.jussieu.fr>
//
// Eigen is free software; you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation; either version 2 or (at your option) any later version.
//
// Eigen is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
// details.
//
// You should have received a copy of the GNU General Public License along
// with Eigen; if not, write to the Free Software Foundation, Inc., 51
// Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
//
// As a special exception, if other files instantiate templates or use macros
// or functions from this file, or you compile this file and link it
// with other works to produce a work based on this file, this file does not
// by itself cause the resulting work to be covered by the GNU General Public
// License. This exception does not invalidate any other reasons why a work
// based on this file might be covered by the GNU General Public License.

#ifndef EIGEN_TEST_MAIN_H
#define EIGEN_TEST_MAIN_H

#include <QtTest/QtTest>
#include "../src/Core.h"

#include <cstdlib>
#include <ctime>

#define DEFAULT_REPEAT 50
#define REPEAT for(int repeat_iteration = 0; repeat_iteration < m_repeat; repeat_iteration++)

namespace Eigen {

class EigenTest : public QObject
{
    Q_OBJECT

  public:
    EigenTest(int repeat) : m_repeat(repeat) {}

  private slots:
    void testBasicStuff();
    void testAdjoint();
 
  protected:
    int m_repeat;
};

} // end namespace Eigen

#endif // EIGEN_TEST_MAIN_H
