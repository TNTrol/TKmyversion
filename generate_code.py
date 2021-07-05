from typing import List, Dict


class Variable:
	def __init__(self, counter:int):
		self.counter = counter
		self.type = ''
		self.line = -1
		self.arg = False

	def new_variable(self, line: int, type: str, arg:bool = False):
		if type != self.type:
			self.line = line
			self.type =	type
			self.arg = arg


class Generation:
	def __init__(self):
		self.counter_line = 0
		self.logical_line = -1
		self.counter = 0
		self.code: List[str] = []
		self.var_counter: Dict[str, Variable] = {}
		self.start = True
		self.__size = 0
		self.__max_size = 0
		self.insert = -1

	def get_index_by_name(self, var:str):
		if var in self.var_counter:
			return self.var_counter[var]
		return -1

	def add_var(self, var:str, type: str):
		self.var_counter[var] = Variable(self.counter,type, self.logical_line)
		self.counter += 1
		return self.counter - 1

	def clear_vars(self):
		self.code.insert(self.insert, f'.limit stack {self.__max_size}')
		self.code.insert(self.insert + 1, f'.limit locals {len(self.var_counter)}')
		self.var_counter.clear()
		self.counter = 0

	def add(self, line: str, is_func = False):
		if is_func:
			self.insert = len(self.code) + 1
		self.code.append(line)

	def get_var(self, name:str):
		if name not in self.var_counter:
			self.var_counter[name] = Variable(self.counter)
			self.counter += 1
		return self.var_counter[name].counter

	def set_variable(self, name:str, type: str, is_arg: bool = False):
		var = self.var_counter[name]
		l = 0 if is_arg else self.logical_line
		var.new_variable(l, type.upper()[0], is_arg)

	def __str__(self):
		res: str = ''
		for codeline in self.code:
			res += f'{codeline}\n'
		return res

	def get_size(self):
		return self.__size

	def add_to_size(self, i):
		self.__size += i
		if self.__size > self.__max_size:
			self.__max_size = self.__size

	def get_type(self, name):
		return self.var_counter[name].type

	def get_last_line(self):
		return self.code[-1]