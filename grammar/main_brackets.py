from common import *
from parser import ParserErrorCorrect


def get_sentence():
    return '{}}(()'

def build_sentence(s):
    s = ' '.join([char + '/' + char + '<' + char + '>' for char in s])
    return s

@timing
def main():
    s = build_sentence(get_sentence())
    ParserErrorCorrect.run(
        grammar_path='grammars/rules_brackets.cfg',
        s=s,
        debug=False,
    )


if __name__ == '__main__':
    main()