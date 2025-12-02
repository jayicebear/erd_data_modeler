import re, chardet
import json 

with open('/home/ljm/web_modeler/SQL_data/CMSUSR1.sql', 'rb') as f:
    encoding = chardet.detect(f.read())['encoding']

with open('/home/ljm/web_modeler/SQL_data/CMSUSR1.sql', 'r', encoding=encoding, errors='ignore') as f:
    content = f.read()

# 테이블 설명
table_descriptions = {
    m.group(1): m.group(2)
    for m in re.finditer(r"COMMENT\s+ON\s+TABLE\s+([A-Z0-9_]+\.[A-Z0-9_]+)\s+IS\s+'([^']+)'", content, re.I)
}

# 컬럼 설명
column_descriptions = {}
for m in re.finditer(r"COMMENT\s+ON\s+COLUMN\s+([A-Z0-9_]+\.[A-Z0-9_]+)\.([A-Z0-9_\"]+)\s+IS\s+'([^']+)'", content, re.I):
    t, c, d = m.group(1), m.group(2).strip('"'), m.group(3)
    if t not in column_descriptions:
        column_descriptions[t] = {}
    column_descriptions[t][c] = d
    
    
with open('./SQL_data/table_description.json', 'w', encoding='utf-8') as f:
    json.dump(table_descriptions, f, ensure_ascii=False, indent=2)
    
with open('./SQL_data/table_column.json', 'w', encoding='utf-8') as f:
    json.dump(column_descriptions, f, ensure_ascii=False, indent=2)
    
print("테이블 설명:", len(table_descriptions))
print("\n컬럼 설명:", len(column_descriptions))
