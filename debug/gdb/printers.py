# -*- coding: utf-8 -*-
# This file is part of Eigen, a lightweight C++ template library
# for linear algebra.
#
# Copyright (C) 2009 Benjamin Schindler <bschindler@inf.ethz.ch>
#
# Eigen is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 3 of the License, or (at your option) any later version.
#
# Alternatively, you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; either version 2 of
# the License, or (at your option) any later version.
#
# Eigen is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License or the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License and a copy of the GNU General Public License along with
# Eigen. If not, see <http://www.gnu.org/licenses/>.

# Pretty printers for Eigen::Matrix
# This is still pretty basic as the python extension to gdb is still pretty basic. 
# It cannot handle complex eigen types and it doesn't support any of the other eigen types
# Such as quaternion or some other type. 
# This code supports fixed size as well as dynamic size matrices

# To use it:
#
# * create a directory and put the file as well as an empty __init__.py in that directory
# * Create a ~/.gdbinit file, that contains the following:


import gdb
import re
import itertools


class EigenMatrixPrinter:
	"Print Eigen Matrix of some kind"

	def __init__(self, val):
		"Extract all the necessary information"
		# The gdb extension does not support value template arguments - need to extract them by hand
		type = val.type
		if type.code == gdb.TYPE_CODE_REF:
			type = type.target()
		self.type = type.unqualified().strip_typedefs()
		tag = self.type.tag
		regex = re.compile('\<.*\>')
		m = regex.findall(tag)[0][1:-1]
		template_params = m.split(',')
		template_params = map(lambda x:x.replace(" ", ""), template_params)

		self.rows = int(template_params[1])
		self.cols = int(template_params[2])

		if self.rows == 10000:
			self.rows = val['m_storage']['m_rows']

		if self.cols == 10000:
			self.cols = val['m_storage']['m_cols']

		self.innerType = self.type.template_argument(0)

		self.val = val
		
		# Fixed size matrices have a struct as their storage, so we need to walk through this
		self.data = self.val['m_storage']['m_data']
		if self.data.type.code == gdb.TYPE_CODE_STRUCT:
			self.data = self.data['array']
			self.data = self.data.cast(self.innerType.pointer())
			
	class _iterator:
		def __init__ (self, rows, cols, dataPtr):
			self.rows = rows
			self.cols = cols
			self.dataPtr = dataPtr
			self.currentRow = 0
			self.currentCol = 0

		def __iter__ (self):
			return self

		def next(self):
			if self.currentCol >= self.cols:
				raise StopIteration

			row = self.currentRow
			col = self.currentCol
			self.currentRow = self.currentRow + 1
			if self.currentRow >= self.rows:
				self.currentRow = 0
				self.currentCol = self.currentCol + 1

			item = self.dataPtr.dereference()
			self.dataPtr = self.dataPtr + 1

			return ('[%d, %d]' % (row, col), item)

	def children(self):
		
		return self._iterator(self.rows, self.cols, self.data)

	def to_string(self):
		return "Eigen::Matrix<%s,%d,%d> (data ptr: %s)" % (self.innerType, self.rows, self.cols, self.data)

def build_eigen_dictionary ():
	pretty_printers_dict[re.compile('^Eigen::Matrix<.*>$')] = lambda val: EigenMatrixPrinter(val)

def register_eigen_printers(obj):
	"Register eigen pretty-printers with objfile Obj"

	if obj == None:
		obj = gdb
	obj.pretty_printers.append(lookup_function)

def lookup_function(val):
	"Look-up and return a pretty-printer that can print va."

	type = val.type

	if type.code == gdb.TYPE_CODE_REF:
		type = type.target()
	
	type = type.unqualified().strip_typedefs()

	typename = type.tag
	if typename == None:
		return None

	for function in pretty_printers_dict:
		if function.search(typename):
			return pretty_printers_dict[function](val)

	return None

pretty_printers_dict = {}

build_eigen_dictionary ()
