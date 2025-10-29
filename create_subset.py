import os
import shutil


labeldir='output/labels'
imgdir='sample7k'
outdir='swimpool2'

counter=0
for f in os.listdir(labeldir):
	if f.endswith('.txt'):
		imgfile=os.path.join(imgdir,f.replace('.txt','.jpg'))
		outfile=os.path.join(outdir,f.replace('.txt','.jpg'))
		if os.path.isfile(imgfile):
			print ( counter, imgfile )
			counter+=1
			shutil.copy(imgfile,outfile)
