import ast
import pprint
import re
with open("data/town01/town01_infos_train.pkl", "r") as f:
    data = f.read()

data_clean = re.sub(r'array\((\[.*?\])\)', r'\1', data, flags=re.DOTALL)

dic = ast.literal_eval(data_clean)

with open('data/town01/town01_infos_train.pkl', 'w') as f:
    pprint.pprint(dic, stream=f, width=120, sort_dicts=False)

