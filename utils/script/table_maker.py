path = '../test/result_table_test.txt'
def row_num(n):
    n_c = ''
    for i in range(n):
        n_c += 'c'
    return n_c
if __name__ == '__main__':
    n_c=row_num(6)
    print(n_c)
    table_start='\\begin{tabular}{'+n_c+'}\n\\toprule'
    table_end = '\end{tabular}'
    open(path, 'w').write(table_start + '\n')
    open(path, 'a').write(table_end)