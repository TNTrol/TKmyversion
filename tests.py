import sys

import parser_base
import os


def run_all(debug=False)->bool:
    return  working_test(debug) and working_test(debug)


def working_test(debug=False) -> bool:
    string = open('tests/working_test.c', 'r').read()
    # print(parser_base.parse(string).tree())

    try:
        if debug:
            print("working testing:")
            print(*parser_base.parse(string, debug).tree, sep=os.linesep)
        else:
            parser_base.parse(string).tree
        return True
    except:
        print(sys.exc_info())
        print(sys.last_traceback)
        return False


def dont_working_tests(debug=False)->bool:
    print("don't working testing:")
    for i in range(16):
        file = open(f'tests/not_working_test_{i}.c').read()
        try:
            print(f'{i} ', sep="")
            parser_base.parse(file)
            return False
        except:
            pass

    return True