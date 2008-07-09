//=====================================================
// File   :  dump_file_x_y.hh
// Author :  L. Plagne <laurent.plagne@edf.fr)>        
// Copyright (C) EDF R&D,  lun sep 30 14:23:20 CEST 2002
//=====================================================
// 
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.
// 
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
// 
#ifndef DUMP_FILE_X_Y_HH
#define DUMP_FILE_X_Y_HH
#include <fstream>
#include <string>

// The Vector class must satisfy the following part of STL vector concept :
//            resize() method
//            [] operator for seting element
// the vector element must have the << operator define

using namespace std;

template<class Vector_A, class Vector_B>
void dump_file_x_y(const Vector_A & X, const Vector_B & Y, const std::string & filename){
  
  ofstream outfile (filename.c_str(),ios::out) ;
  int size=X.size();
  
  for (int i=0;i<size;i++){

      outfile << X[i] << " " << Y[i] << endl ;

  }

  outfile.close();
} 

#endif
