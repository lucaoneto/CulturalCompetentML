import sys
sys.path.insert(1, '../')
from Utils.utils import FileClass

f = FileClass('./test.csv')
f.writecm([[0,1], [2,3]])
f.writerow([5,5,5,5])
rows = f.readrows()
print(f'Rows are {rows}')
cms = f.readcms()
print(f'Confusion Matrices are {cms}')