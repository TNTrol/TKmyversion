from lark import Lark, Token, InlineTransformer
from ast_node import *

'''
type_ident = (("int" | "float" ))    TO-DO   |  "bool" | "char"))
float_num -> <Дробное число>
int_num -> <Целое число>
num -> (("float_num" | "int_num"))
ident -> <Идентификатор>
group -> num | ident | '(' add ')'
mult -> group (('*' | '/' ) group)*
add -> mult (('+' | '-') mult)*
compare -> add (('<' | "<=" | '>' | ">=" | "==" | "!=") add)*
expr -> compare
init -> type_ident ident | type_ident ident = expr
assign -> ident "=" expr

if -> 'if' '(' compare ')' block ('else' block )?
while -> 'while' '(' compare ')' block

stmt -> assign | if | while  
stmts -> stmt*

block -> '{' stmts '}'

'''


class ASTBuilder(InlineTransformer):
    def __getattr__(self, item):
        if isinstance(item, str) and item.upper() == item:
            return lambda x: x

        if item in ('bin_op',):
            def get_bin_op_node(*args):
                op = BinOp(args[1].value)
                return BinOpNode(op, args[0], args[2],
                                 **{'token': args[1], 'line': args[1].column, 'column': args[1].column})

            return get_bin_op_node
        else:
            def get_node(*args):
                props = {}
                if len(args) == 1 and isinstance(args[0], Token):
                    props['token'] = args[0]
                    props['line'] = args[0].line
                    props['column'] = args[0].column
                    args = [args[0].value]
                cls = eval(''.join(x.capitalize() for x in item.split('_')) + 'Node')
                return cls(*args, **props)

            return get_node


def parse(prog: str, debug=False) -> StmtListNode:
    parser = Lark.open("./syntax.lark", start='start', lexer='standard', propagate_positions=True)  # , lexer="standard")
    prog = parser.parse(prog)
    if debug:
        print(prog.pretty())
        print(prog)
    prog = ASTBuilder().transform(prog)
    return prog
