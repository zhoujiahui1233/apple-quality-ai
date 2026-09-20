from pathlib import Path
p=Path(r'E:\apple\_patch_modelcmp_docs.py')
s=p.read_text(encoding='utf-8')
s=s.replace('def tbl(anchor, headers, rows, fs=10, caption=None):\n    if caption:\n        mk(anchor, caption, bold=True, size=fs, indent=False, font="黑体")\n    tb = anchor._parent.add_table(rows=1, cols=len(headers)); tb.style = "Table Grid"',
            'def tbl(doc, anchor, headers, rows, fs=10, caption=None):\n    if caption:\n        mk(anchor, caption, bold=True, size=fs, indent=False, font="黑体")\n    tb = doc.add_table(rows=1, cols=len(headers)); tb.style = "Table Grid"')
s=s.replace('tbl(a, ["系统/仪器指标"', 'tbl(doc, a, ["系统/仪器指标"')
p.write_text(s,encoding='utf-8')
print('patched')
