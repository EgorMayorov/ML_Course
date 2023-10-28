def check(s, filename):
    f = open(filename, 'w')
    lst = s.lower().split(' ')
    dct = dict()
    for i in lst:
        count = lst.count(i)
        dct.update({i: count})
    for i in sorted(dct.keys()):
        f.write(i)
        f.write(' ')
        f.write(str(dct[i]))
        f.write('\n')
    f.close()
