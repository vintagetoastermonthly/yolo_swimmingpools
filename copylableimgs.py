outdir='bigpoolsimg2'
checkdir='bigpoolsimg'
imdir='../chips_0075m'
lbdir='output/labels'

threshold=0.7

import os, numpy as np, shutil

counter=0
for f in os.listdir(lbdir):
	if f.endswith('.txt'):
		srcfile=os.path.join(imdir,f.replace('.txt','.jpg'))
		dstfile=os.path.join(outdir,f.replace('.txt','.jpg'))
		chkfile=os.path.join(checkdir,f.replace('.txt','.jpg'))
		counter+=1
		if counter % 1000 == 0: print ( counter, end=' ', flush=True )
		if not os.path.isfile(dstfile) and not os.path.isfile(chkfile):
			infile=os.path.join(lbdir,f)
			isbigpool=False
			with open(infile) as src:
				for line in src:
					#print ( infile, line )
					(cls, x1, x2, x3, x4, conf)=line.split(' ')
					#print ( f'cls:{cls}, x1:{x1}, x2:{x2}, x3:{x3}, x4:{x4}, conf:{conf}')
					if cls == '0' and float(conf) > threshold:
						isbigpool=True
			if isbigpool==True:
				shutil.copy(srcfile,dstfile)
print ( counter )
