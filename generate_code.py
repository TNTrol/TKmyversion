from typing import List, Dict


class Variable:
	def __init__(self, counter:int):
		self.counter = counter
		self.type = ''
		self.line = -1
		self.arg = False

	def new_variable(self, type: str, arg: bool = False):
		if type != self.type:
			self.type =	type
			self.arg = arg


class Generation:
	def __init__(self):
		self.__counter = 0
		self.code: List[str] = []
		self._var_counter: Dict[str, Variable] = {}
		self.__size = 0
		self.__max_size = 0
		self._insert = -1

	def get_index_by_name(self, var:str):
		if var in self._var_counter:
			return self._var_counter[var]
		return -1

	def add_var(self, var:str, type: str):
		self._var_counter[var] = Variable(self.__counter, type, self.logical_line)
		self.__counter += 1
		return self.__counter - 1

	def clear_vars(self):
		self.code.insert(self._insert, f'.limit stack {self.__max_size}')
		self.code.insert(self._insert + 1, f'.limit locals {len(self._var_counter)}')
		self._var_counter.clear()
		self.__counter = 0
		self.__max_size = 0
		self.__size = 0

	def add(self, line: str, is_func = False):
		if is_func:
			self._insert = len(self.code) + 1
		self.code.append(line)

	def get_var(self, name:str):
		if name not in self._var_counter:
			self._var_counter[name] = Variable(self.__counter)
			self.__counter += 1
		var = self._var_counter[name]
		return var.counter, var.type

	def set_variable(self, name: str, type: str, is_arg: bool = False):
		var: Variable = None
		if name not in self._var_counter:
			var = Variable(self.__counter)
			self._var_counter[name] = var
			self.__counter += 1
		else:
			var = self._var_counter[name]
		var.new_variable(type.lower()[0], is_arg)

	def __str__(self):
		res: str = ''
		for codeline in self.code:
			res += f'{codeline}\n'
		return res

	def get_size(self) -> int:
		return self.__size

	def add_to_size(self, i):
		self.__size += i
		if self.__size > self.__max_size:
			self.__max_size = self.__size

	def get_type(self, name):
		return self._var_counter[name].type

	def get_last_line(self) -> str:
		return self.code[-1]

	def get_index_current_line(self) -> int:
		return len(self.code)

	def insert(self, index:int, code:str):
		self.code.insert(index, code)