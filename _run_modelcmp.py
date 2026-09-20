# -*- coding: utf-8 -*-
"""同图四模型对比 + 系统最终裁决（120张）"""
import sys, random, time, csv, shutil
sys.path.insert(0, r'E:\apple')
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import numpy as np, cv2

GOOD=Path(r'E:\apple\dataset_collected\apple\合格')
BAD=Path(r'E:\apple\dataset_collected\apple\不合格')
TMP=Path(r'E:\apple\_e2e_tmp'); TMP.mkdir(exist_ok=True)

def valid_jpg(d):
    out=[]; seen=set()
    for f in sorted(d.iterdir()):
        if f.suffix.lower()!='.jpg' or f.stem in seen: continue
        seen.add(f.stem)
        if f.stat().st_size<1000: continue
        data=np.fromfile(str(f),dtype=np.uint8)
        if cv2.imdecode(data, cv2.IMREAD_COLOR) is None: continue
        out.append(f)
    return out

g=valid_jpg(GOOD); b=valid_jpg(BAD)
random.seed(20260904); random.shuffle(g); random.shuffle(b)
g=g[:60]; b=b[:60]

from services.detection_service import DetectionService
svc=DetectionService()
def prem(grd): return 'Premium' in str(grd)
rows=[]; t0=time.time(); i=0
for tag,files in [('good',g),('bad',b)]:
    for f in files:
        i+=1
        tmp=TMP/('m%03d.jpg'%i)
        shutil.copy2(f,tmp)
        core=svc._apple_detect_core(str(tmp), 'cmp4_%03d'%i)
        tmp.unlink(missing_ok=True)
        truth = 1 if tag=='bad' else 0   # 1=不合格
        if 'error' in core:
            rows.append([f.name,tag,'ERR','','','','','']); continue
        svm=prem(core.get('svm_grade')); rf=prem(core.get('rf_grade'))
        dl=prem(core.get('dl_grade')); yolo=prem(core.get('yolo_grade'))
        fin=prem(core.get('grade'))
        rows.append([f.name, tag, 'SVM' if svm else 'RF_none', 'x','x','x','x','x'])  # placeholder fix below
        rows[-1]=[f.name, tag, int(svm), int(rf), int(dl), int(yolo), int(fin), str(core.get('grade'))[:20]]
        if i%20==0: print('  %d/%d  %.0fs'%(i,120,time.time()-t0))

shutil.rmtree(TMP, ignore_errors=True)
# 汇总（premium=0 合格）
keys=['svm','rf','dl','yolo','final']
import collections
res={k:{'good':0,'bad':0,'tot':0,'n_g':0,'n_b':0} for k in keys}
for r in rows:
    name,tag,svm,rf,dl,yolo,fin,grade=r
    if tag=='ERR': continue
    truth = 1 if tag=='bad' else 0
    preds={'svm':svm,'rf':rf,'dl':dl,'yolo':yolo,'final':fin}  # 1=不合格
    for k in keys:
        # 预测“不合格”=1；正确=预测==truth
        ok = (preds[k]==truth)
        res[k]['tot']+=ok
        if tag=='good': res[k]['n_g']+=1; res[k]['good']+=ok
        else: res[k]['n_b']+=1; res[k]['bad']+=ok

print('\n==== 同图四模型对比(60合格+60不合格) ====')
print('%-8s %-10s %-10s %-10s'%('模型','合格池','不合格池','总体'))
for k in keys:
    s=res[k]
    print('%-8s %d/%d (%.1f%%)  %d/%d (%.1f%%)  %d/120 (%.1f%%)'%(
        k, s['good'],s['n_g'],100*s['good']/s['n_g'], s['bad'],s['n_b'],100*s['bad']/s['n_b'], s['tot'], 100*s['tot']/120))
# 两两一致率(120)
agree={}
for a in keys:
    for b in keys:
        if a>=b: continue
        same=0; n=0
        for r in rows:
            if r[1]=='ERR': continue
            va={'svm':r[2],'rf':r[3],'dl':r[4],'yolo':r[5],'final':r[6]}[a]
            vb={'svm':r[2],'rf':r[3],'dl':r[4],'yolo':r[5],'final':r[6]}[b]
            n+=1; same+=(va==vb)
        agree[(a,b)]=100*same/n
print('\n==== 模型间判定一致率(%) ====')
for (a,b),v in agree.items():
    print('%-7s vs %-7s : %.1f'%(a,b,v))
# 保存明细
outp=Path(r'E:\apple\论文素材\同图四模型对比_明细.csv')
with open(outp,'w',newline='',encoding='utf-8-sig') as f:
    w=csv.writer(f); w.writerow(['file','truth','svm','rf','dl','yolo','final','final_grade'])
    for r in rows: w.writerow(r)
print('明细已存:', outp)
