import parser_base
import semantic
import os

from tests import working_test
from generate_code import Generation

if __name__ == '__main__':
    prog = open('tests/aaaaa.C', 'r').read()

    # prog = parser_base.parse("void d(int a[1]){}", True)
    # print(*parser_base.parse(prog).tree, sep=os.linesep)
    # parser_base.parse(prog)

    # if working_test(False):
    #     print("All tests have been passed")
    # else:
    #     print("It isn't ok")

    tree = parser_base.parse(prog)
    print('ast_tree:')
    print(*tree.tree, sep=os.linesep)
    tree.program=True
    print('semantic_check:')
    try:
        scope = semantic.get_default_scope()
        tree.semantic_check(scope)
        print(*tree.tree, sep=os.linesep)
        gen = Generation()
        tree.to_bytecode(gen)
        print(gen)
        help = open('helper.class', 'r')
        text = help.read()
        help.close()
        out = open('main.j', 'w')
        out.write(text + '\n' + str(gen))
        out.close()
    except semantic.SemanticException as e:
        print('Ошибка: {}'.format(e.message))
