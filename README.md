# document-layout-analysis

Document-layout-analysis est un petit projet Python qui permet à quiconque d'extraire facilement des tableaux et de la segmentation du texte à partir d'image  !

* Voici comment extraire des tableaux à partir d'image

<pre>
>>> from table_extraction.parsers.lattice import Lattice
>>> parser = Lattice()
>>> tables = parser.extract_tables(r'C:\DEV\Table_Detection\data\fact230001-1.png')
>>> print(tables)
\[&lt;Table shape=(8, 4)&gt;,&lt;Table shape=(13, 4)&gt\]
>>> print(tables[1].parsing_report)
{
    'accuracy': 99.91,
     'whitespace': 71.15,
      'order': 2,*
       'page': None
 }
>>> tables[0].to_csv('test.csv') # to_json, to_excel, to_html, to_sqlite
>>> tables[0].df # get a pandas DataFrame!
</pre>

* vous pouvez regarder aussi ce notebook [Block segmentation and text extraction](https://github.com/bilal-rachik/document-layout-analysis/blob/main/Block%20segmentation%20and%20text%20extraction.ipynb)
l'objectif est de détecter et isoler les zones de texte dans une image. Le but de la segmentation par blocs est d'extraire des régions de texte par rapport à l'ordre de lecture. (Un ensemble ordonné de régions d'image contenant du texte)

