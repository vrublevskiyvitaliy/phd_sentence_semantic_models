#!/usr/bin/python
# coding=utf-8
# -*- encoding: utf-8 -*-

from operator import add

class TreeNode:
    INDENT_STEP = 3

    def __init__(self, body, children=[]):
        '''Initialize a tree with body and children'''
        self.body = body
        self.children = children

    def __len__(self):
        '''A length of a tree is its leaves count'''
        if self.is_leaf():
            return 1
        else:
            return reduce(add, [len(child) for child in self.children])

    def __repr__(self):
        return self.repr_ierarhical_notation()
        return self.repr_bracket_notation()

    def repr_bracket_notation(self):
        '''Returns string representation of a tree in bracket notation'''

        st = "[.{0} ".format(self.body)
        if not self.is_leaf():
            st += ' '.join([str(child) for child in self.children])
        st += ' ]'
        return st

    def repr_ierarhical_notation(self, level=0):
        '''Returns string representation of a tree in ierarhical notation'''
        st = ' ' *  level * TreeNode.INDENT_STEP
        st += "{0}".format(self.body)
        st += "\n"
        if not self.is_leaf():
            for child in self.children:
                #st += ' ' * (indent + TreeNode.INDENT_STEP)
                st += child.repr_ierarhical_notation(level + 1)
        return st

    def is_leaf(self):
        '''A leaf is a childless node'''
        return len(self.children) == 0


class ParseTrees:
    def __init__(self, parser):
        '''Initialize a syntax tree parsing process'''
        self.parser = parser
        self.charts = parser.charts
        self.length = len(parser)

        self.nodes = []
        self.min_node = None

        for root in parser.complete_parses:
            node = {
                'tree': self.build_nodes_ierarchical(root),
                'weight': root.weight
            }
            if self.min_node is None or self.min_node['weight'] > node['weight']:
                self.min_node = node

        self.nodes.append(self.min_node)

    def __len__(self):
        '''Trees count'''
        return len(self.nodes)

    def __repr__(self):
        '''String representation of a list of trees with indexes'''
        return '<Parse Trees>\n{0}</Parse Trees>' \
            .format('\n'.join("Parse tree #{0} weight = {1}:\n{2}\n\n" \
                                  .format(i + 1, self.nodes[i]['weight'], str(self.nodes[i]['tree']))
                              for i in range(len(self))))

    def build_nodes(self, root):
        '''Recursively create subtree for given parse chart row'''
        nodes = []

        # find subtrees of current symbol
        if root.completing:
            down = self.build_nodes(root.completing)
        else:
            down = [TreeNode(root.prev_category())]

        # prepend subtrees of previous symbols
        prev = root.previous
        left = []
        while prev and prev.dot > 0:
           left[:0] = [self.build_nodes(prev)]
           prev = prev.previous

        left.append(down)

        return [TreeNode(root.rule.lhs, children) for children in left]

    def build_nodes_ierarchical(self, root, terminal_to_traverse_back=None):
        '''
            Recursively create subtree for given parse chart row
        :param root: ChartRow
        :param terminal_to_traverse_back:
        :return:
        '''
        ''''''
        nodes = []

        # find subtrees of current symbol
        if root.completing:
            down = self.build_nodes_ierarchical(root.completing)
        else:
            down = TreeNode(root.prev_category())

        # prepend subtrees of previous symbols
        prev = root.previous
        left = []
        while prev and prev.dot > 0 and terminal_to_traverse_back is None:
           left[:0] = [self.build_nodes_ierarchical(prev, 1).children[0]]
           prev = prev.previous

        left.append(down)

        return TreeNode(root.rule.lhs, left)

