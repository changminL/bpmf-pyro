import sys
import random

if len(sys.argv) != 4:
	print("Usage : python make_data.py #ofuser #ofitem #ofrating")
	exit()
n_user = int(sys.argv[1])
n_item = int(sys.argv[2])
n_rating = int(sys.argv[3])

o = open("ratings-made.dat", 'w')

for i in range(n_rating):
		user_id = random.randint(1,n_user)
		item_id = random.randint(1,n_item)
		rating = random.randint(1,5)
		o.write("::".join([str(user_id),str( item_id), str(rating)]) + "::1234\n")

o.close()
