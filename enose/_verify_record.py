# -*- coding: utf-8 -*-
from openpyxl import load_workbook
p = r"E:\apple\enose\data\record.xlsx"
wb = load_workbook(p)
print("sheets:", wb.sheetnames)
ws = wb["记录表"]
print("headers:", [c.value for c in ws[1]])
print("example row:", [c.value for c in ws[2]][:5], "...")
print("dvs:", [(dv.type, str(dv.formula1)[:40]) for dv in ws.data_validations.dataValidation])
ws2 = wb["填写说明"]
print("notes rows:", ws2.max_row)