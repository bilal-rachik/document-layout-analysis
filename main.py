from table_extraction.parsers.lattice import Lattice
from table_extraction.plotting import PlotMethods

if __name__ == '__main__':
    parser = Lattice()
    tables = parser.extract_tables(r'C:\DEV\Table_Detection\data\fact230001-1.png')
    print(tables[1].parsing_report)
    pl = PlotMethods()
    table_1 = pl(tables[0])
    table_1.show()

