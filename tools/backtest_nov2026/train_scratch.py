import sys, shutil; sys.path.insert(0,'/home/claude/aai6640-team5')
from pathlib import Path
from gaslab import data as D, service
D.DATA = Path('/home/claude/scratch/data')
m = service.train(seeds=3, out=Path('/home/claude/scratch/model')); print(m)
