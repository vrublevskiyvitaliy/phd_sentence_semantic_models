from common import *
from parser import ParserErrorCorrect


@timing
def parse(s):
    ParserErrorCorrect.run(
        grammar_path='grammars/rules_mini.cfg',
        s=s,
        debug=False,
        start_rule='S'
    )

def main():
    s = 'Look/Look<VB> at/at<IN> of/of<IN> water-gazers/water-gazers<NNS> the/the<DT> crowds/crowds<NNS> there/there<RB> ./.<.>'
    parse(s)


if __name__ == '__main__':
    main()