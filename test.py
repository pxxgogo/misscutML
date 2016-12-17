import numpy.random as nr
new_corpus = filter(lambda x: sum(map(lambda y: 1 if (y >= 'a' and y <= 'z') or (y >= 'A' and y <= 'X') else 0, x)) < 3, corpus)
# new_corpus = filter(lambda x: nr.random() <= 0.1, new_corpus)