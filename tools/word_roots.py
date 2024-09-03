import os
import sys

local_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(local_dir + "/derinet/tools/data-api/derinet2")
import derinet.lexicon as dlex

class DeriNet(object):
    def __init__(self, path=local_dir+"/derinet/data/releases/cs/derinet-2-0.tsv"):
        self.lexicon = dlex.Lexicon()
        self.lexicon.load(path)

    def get_root(self, word):
        lexemes = self.lexicon.get_lexemes(word)
        if len(lexemes) > 0:
            root = lexemes[0].get_tree_root()
            return root.lemma
        else:
            return word
    