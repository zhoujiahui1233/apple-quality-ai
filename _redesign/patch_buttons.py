# -*- coding: utf-8 -*-
import io, os

BASE = os.path.dirname(os.path.abspath(__file__))
TPL = os.path.join(BASE, '..', 'templates')

def patch(fname, subs):
    p = os.path.join(TPL, fname)
    s = io.open(p, encoding='utf-8').read()
    n = 0
    for a, b in subs:
        c = s.count(a)
        if c:
            s = s.replace(a, b)
            n += c
        else:
            print('  [NOT FOUND]', fname, '::', a[:70])
    io.open(p, 'w', encoding='utf-8', newline='\n').write(s)
    print(fname, '->', n, 'replacements')

patch('history.html', [
    ('<a href="/history?review=1" class="btn btn-sm" style="border:1px solid var(--warning);color:var(--warning);background:rgba(255,193,7,0.06);">',
     '<a href="/history?review=1" class="btn-w btn-sm2">'),
    ('<a href="/api/export/report" class="btn btn-sm" style="border:1px solid var(--primary);color:var(--primary);background:rgba(0,212,255,0.08);">',
     '<a href="/api/export/report" class="btn-p btn-sm2">'),
    ('<button class="btn btn-sm" style="border:1px solid var(--accent);color:var(--accent);background:rgba(0,255,136,0.06);" onclick="batchUpdate(\'grade\')">',
     '<button class="btn-a btn-sm2" onclick="batchUpdate(\'grade\')">'),
    ('<button class="btn btn-sm" style="border:1px solid var(--warning);color:var(--warning);background:rgba(255,193,7,0.06);" onclick="batchUpdate(\'category\')">',
     '<button class="btn-w btn-sm2" onclick="batchUpdate(\'category\')">'),
    ('<button class="btn btn-sm" style="border:1px solid var(--primary);color:var(--primary);background:rgba(0,212,255,0.06);" onclick="batchMarkReviewed()">',
     '<button class="btn-p btn-sm2" onclick="batchMarkReviewed()">'),
    ('<button class="btn btn-sm btn-danger" onclick="batchDelete()">',
     '<button class="btn-d btn-sm2" onclick="batchDelete()">'),
    ('<button class="btn btn-sm" style="border:1px solid var(--danger);color:#ff6b81;background:rgba(255,71,87,0.06);" onclick="deleteAll()">',
     '<button class="btn-d btn-sm2" onclick="deleteAll()">'),
])

patch('stats.html', [
    ('<a href="/api/export/report" class="btn btn-sm" style="border:1px solid var(--primary);color:var(--primary);background:rgba(0,212,255,0.08);">',
     '<a href="/api/export/report" class="btn-p btn-sm2">'),
    ('<button class="btn btn-sm" style="border:1px solid var(--primary);color:var(--primary);background:rgba(0,212,255,0.08);" onclick="mgCount()">',
     '<button class="btn-p btn-sm2" onclick="mgCount()">'),
    ('<button class="btn btn-sm" style="border:1px solid var(--accent);color:var(--accent);background:rgba(0,255,136,0.06);" onclick="mgUpdate(\'grade\')">',
     '<button class="btn-a btn-sm2" onclick="mgUpdate(\'grade\')">'),
    ('<button class="btn btn-sm" style="border:1px solid var(--warning);color:var(--warning);background:rgba(255,193,7,0.06);" onclick="mgUpdate(\'category\')">',
     '<button class="btn-w btn-sm2" onclick="mgUpdate(\'category\')">'),
    ('<button class="btn btn-sm btn-danger" onclick="mgDelete()">',
     '<button class="btn-d btn-sm2" onclick="mgDelete()">'),
])
