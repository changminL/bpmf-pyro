import sys
import random

if len(sys.argv) != 3:
	print("Usage : python make_data.py #ofuser #ofitem")
	exit()
n_user = int(sys.argv[1])
n_item = int(sys.argv[2])

o = open("ratings-made.dat", 'w')

for line in open("ratings.dat",'r').readlines():
		line2 = line.strip().split('::')
		if ( int(line2[0]) <= n_user) and (int (line2[1]) <= n_item):
				o.write(line)

o.close()
