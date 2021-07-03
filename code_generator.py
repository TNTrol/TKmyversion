from typing import List


class CodeLine:
    def __init__(self, code: str):
     self.code = code

    def __str__(self):
        return self.code


class CodeGenerator:
    def __init__(self):
        self.code_lines: List[CodeLine] = []

    def start(self):
        self.code_lines.append(CodeLine("declare i32 @printf(i8*, ...) nounwind"))
        self.code_lines.append(CodeLine("@formatInt = private constant [3 x i8] c\"%d\\0A\""))


    def add(self, code: str):
        self.code_lines.append(CodeLine(code))

