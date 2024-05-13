# -*- coding: utf-8 -*-

import os
import pandas
import glob
from IPython.display import display

list_file = [_.replace('\\', '/') for _ in sorted(glob.glob('*/*/*.*g'))]
tab = pandas.DataFrame({'filename': list_file})

for index in tab.index:
    filename = tab.loc[index,'filename']
    classid = os.path.basename(os.path.dirname(filename))
    setid = os.path.dirname(os.path.dirname(filename))
    tab.loc[index,'class'] = classid
    tab.loc[index,'set'] = setid
    
display(pandas.crosstab(tab['class'], tab['set']))
