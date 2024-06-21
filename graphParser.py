import csv

from graph import Graph

def parseGraph(fin, dialect='excel', encoding='utf-8', delimiter=' ', skiprows=5):
    if isinstance(fin, str):
        fin = open(fin, "r", encoding=encoding)

    reader = csv.reader(fin, dialect=dialect, delimiter=delimiter)
    v = None
    edges = []
    
    for i, row in enumerate(reader):
        if i <= skiprows:
            continue
        if i == skiprows+1:
            # print(f"|V|={row[2]}, |E|={row[3]}")
            v = int(row[2])
            continue
        
        edges.append((int(row[1])-1, int(row[2])-1))
        # print(",".join(row))
    
    return Graph(v, edges)

if __name__ == '__main__':
    g = parseGraph("1-FullIns_3.col", skiprows=5)
    g.get_dot().render('Machine.gv.pdf', view=True)
