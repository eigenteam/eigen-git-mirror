# -*- coding: iso-8859-1 -*-
# Pretty printers for Eigen::Matrix
# Author: Benjamin Schindler <bschindler@inf.ethz.ch>
# This is still pretty basic as the python extension to gdb is still pretty basic. 
# It cannot handle complex eigen types and it doesn't support any of the other eigen types
# Such as quaternion or some other type. 
# This code supports fixed size as well as dynamic size matrices
# Licence - ment to be included in Eigen, so dual GPLv3 or LGPL
# NOTE: This code was only tested with the stable branch of eigen!

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
		data = self.val['m_storage']['m_data']
		# Fixed size matrices have a struct as their storage, so we need to walk through this
		if data.type.code == gdb.TYPE_CODE_STRUCT:
			data = data['array']
			data = data.cast(self.innerType.pointer())
		return self._iterator(self.rows, self.cols, data)

	def to_string(self):
		return "Eigen::Matrix<%s,%d,%d>" % (self.innerType, self.rows, self.cols)

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
