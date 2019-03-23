import pandas as pd

file = r'./PeMS/station1/1.txt'
table = pd.read_table(file, usecols=[0,1])
table.to_json('./flow.json')
print("hello")