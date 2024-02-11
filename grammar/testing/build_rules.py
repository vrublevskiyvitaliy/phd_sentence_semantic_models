from earley.grammar import *


BUILD_ALL = 'all'
BUILD_MINI = 'mini'

MINI_TREES = [x for x in xrange(0, 100)]

MODE = BUILD_MINI

RULES_TRANSITIVE = False

def get_trees():
    with open("brown-tree.txt", "r") as myfile:
        lines = myfile.readlines()
        return lines

def extract_rules(tree, grammar, source):
    #if len(tree['v']) > 1:
    #        print tree['v'][1] + '/' + tree['v'][1] + '<' + tree['v'][0] + '>'
    if tree['children']:
        seq = []
        for el in tree['children']:
            seq.append(el['v'][0])
        head = tree['v'][0]
        grammar.add_rule(Rule(head, seq, source))
        for el in tree['children']:
            extract_rules(el, grammar, source)


def extract_pos_tags(tree, pos_tags):
    if len(tree['v']) > 1:
        pos_tags.append((tree['v'][1], tree['v'][0]))
    if tree['children']:
        for el in tree['children']:
            extract_pos_tags(el, pos_tags)


def extract_rules_transitive(tree, grammar, source):
    if tree['children']:
        seq = []
        head = tree['v'][0]
        for el in tree['children']:
            current_rule = extract_rules_transitive(el, grammar, source)
            if current_rule is None:
                seq.append(el['v'][0])
                continue
            if len(current_rule.rhs) > 1:
                grammar.add_rule(current_rule)
                seq.append(el['v'][0])
            else:
                seq.append(current_rule.rhs[0])
        if len(seq) > 0:
            return Rule(head, seq, source)
    return None


def get_parse_tree(tree):
    l = len(tree)
    root = {'v' : [], 'children' : []}
    stack = [root]
    current_element = root
    current_level = 0
    token = ''
    for i in xrange(l):
        if tree[i] not in ['(', ')', ' ']:
            token += tree[i]
        else:
            if tree[i] == '(':
                v = []
                if token:
                    v.append(token)
                el = {'v': v, 'children': []}
                current_element['children'].append(el)
                current_element = el
                current_level += 1
                while len(stack) <= current_level:
                    stack.append([])
                stack[current_level] = current_element
            elif tree[i] == ')':
                if token:
                    current_element['v'].append(token)
                current_level -= 1
                current_element = stack[current_level]
                if current_level == 0:
                    break
            elif tree[i] == ' ':
                if len(token):
                    current_element['v'].append(token)

            token = ''
    root = root['children'][0]
    return root


def parse_tree(tree, grammar):
    root = get_parse_tree(tree)
    if RULES_TRANSITIVE:
        rule = extract_rules_transitive(root, grammar, tree)
        grammar.add_rule(rule)
    else:
        extract_rules(root, grammar, tree)


def get_rules_file():
    if RULES_TRANSITIVE:
        return 'rules_transitive.cfg'
    if MODE == BUILD_MINI:
        return 'rules_mini.cfg'
    else:
        return 'rules.cfg'


def get_main_grammar():
    grammar = Grammar()
    trees = get_trees()
    for tree in trees:
        parse_tree(tree, grammar)

    return grammar


def main():
    grammar = get_main_grammar()
    grammar.trim_rules_by_context(1)
    grammar.save_to_file(get_rules_file())
    return


def main_mini():
    trees = get_trees()
    grammar = Grammar()

    for i in MINI_TREES:
        print trees[i]
        parse_tree(trees[i], grammar)

    grammar.save_to_file(get_rules_file())


def get_grammar_by_trees(trees):
    grammar = Grammar()
    for tree in trees:
        parse_tree(tree, grammar)
    return grammar

if __name__ == '__main__':
    if MODE == BUILD_MINI:
        main_mini()
    else:
        main()