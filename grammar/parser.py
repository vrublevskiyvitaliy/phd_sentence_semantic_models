#!/usr/bin/python
# coding=utf-8
# -*- encoding: utf-8 -*-

from earley.parser import Parser
from grammar.grammar import *
from earley.sentence import Sentence
from earley.parse_trees import ParseTrees
from earley.config import *
from earley.chart import *


class ParserErrorCorrect(Parser):

    def __init__(self, grammar, sentence, debug=False, start_rule='ROOT'):
        '''Initialize parser with grammar and sentence'''
        super(ParserErrorCorrect, self).__init__(grammar, sentence, debug, start_rule)

    @staticmethod
    def run(grammar_path, s, debug=False, start_rule='ROOT', grammar=None):
        if grammar is None:
            grammar = GrammarWithCorrection.build_error_correction_grammar(grammar_path)
        else:
            grammar = GrammarWithCorrection.build_error_correction_grammar_from_grammar(grammar)
        sentence = Sentence.from_string(s)
        earley = ParserErrorCorrect(grammar, sentence, debug, start_rule)

        earley.parse()  # output sentence validity
        if earley.is_valid_sentence():
            trees = ParseTrees(earley)
            if debug:
                print '==> Sentence is valid.'
                print 'Valid parse trees:'
                print trees
            return trees.min_node['weight']
        else:
            if debug:
                print '==> Sentence is invalid.'
            return -1

    def prescan(self, chart, position):
        '''
           Scan current word in sentence, and add appropriate
           grammar categories to current chart
           Generally just add word with all combination of tags.
           Do nothing else.
        :param chart: Chart
        :param position: int
        :return: None
        '''
        word = self.sentence[position - 1]
        if word:
            rules = [RuleWithWeight(tag, [word.word]) for tag in word.tags]
            for rule in rules:
                chart.add_row(ChartRow(rule, 1, position - 1))

    def predict(self, chart, position, words_left):
        '''
            Predict next parse by looking up grammar rules for pending categories in current chart.
            Try to move dot to next position by 'opening' new rule.
        :param chart: Chart
        :param position: int
        :param words_left: int Amount of words left to parse. Can be used in trimming trees.
        :return:
        '''
        while chart.predict_row_index < len(chart):
            row = chart.rows[chart.predict_row_index]
            next_cat = row.next_category()
            rules = self.grammar[next_cat]
            if rules:
                for rule in rules:
                    if rule.is_chart_used(position):
                        continue
                    dot_position = 0
                    if rule.is_epsilon_rule():
                        dot_position = 1
                    new = ChartRow(rule, dot_position, position, weight=rule.weight)
                    chart.add_row(new)
                    rule.add_chart(position)
            chart.predict_row_index += 1

    def complete(self, chart, position, words_left):
        '''
            Complete a rule that was done parsing, and promote previously pending rules
        :param chart: Chart
        :param position: int
        :param words_left: int Amount of words left to parse. Can be used in trimming trees.
        :return:
        '''
        for row in chart.rows:
            if row.is_complete():
                completed = row.rule.lhs
                for r in self.charts[row.start].rows:
                    if completed == r.next_category():

                        weight = row.weight + r.weight
                        new = ChartRow(
                            rule=r.rule, # Правило, в якому ми щойно закінчили розкривати нетермінал та пересунули точку
                            dot=r.dot + 1, # Змістили точку
                            start=r.start, # Кількість розібраних до цього токенів не змінилася в цьому правилі
                            previous=r, # Вказуємо на ChartRow в попередніх бакетах який ми продовжуємо
                            completing=row, # Вказуємо на ChartRow який ми щойно закрили
                            weight=weight # Вага
                        )
                        # if Parser.TRIM_BY_LENGTH and new.get_left_len() > words_left:
                        #     continue
                        if weight < 4:
                            chart.add_row(new)