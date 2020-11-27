"""
	Computes samples of FrRe.psi for random cylinders.
	The output is written to the file o as numpy array.
	
  Usage: python3 generate_data_random_5.py -n 10 -d 32 -p 4 -o "data.csv"
  
  Args: -n, number of samples
        -p, number of subprocesses
        -d, mesh density
				-o, output file
				
	Example: 
		python3 generate_data_random_5.py -n 1000 -d 32 -p 4 -o "xy_random_5.csv"
"""
import sys, getopt, time
from multiprocessing import Pool
from FrRe import *
from tqdm import tqdm

def compute_psi(m, md):
	"""
	The particular sweep definition for this job.
	Computes samples of FrRe.psi for random cylinders.
	
	Input: Number of samples m.
	       Mesh density md.
	
	Output: Returns m samples for mesh density md
	        in a numpy array structured as
	        x0 x1 x2 x3 x4 psi
	        ...
	        with m lines.
	"""
	# geometric parameters
	x_min = 0.0 # left base
	x_max = 1.0 # right base
	r_min = 0.1 # minimal radius
	r_max = 0.5 # maximal radius
	N = 5 # discrete radius points

	# load (p=1) part of boundary
	def ga(x, on_boundary):
		return x[0] < DOLFIN_EPS

	# response region parameters
	lambda_min = 0.0  # left point of lambda range
	lambda_max = 60.0 # right point of lambda range

	# compute m samples
	x = np.linspace(x_min, x_max, N)
	xy = np.zeros((m, N+1)) # to store data point (x1, ..., xN, psi)
	for i, (mesh, radius) in tqdm(enumerate(RandomCylinder2d(x=x, max_n=m, min_radius=r_min, max_radius=r_max, export_radius=True, mesh_density=md)), total=md):
		psival = psi(mesh, ga, lambda_min, lambda_max)
		xy[i, 0:5] = radius
		xy[i, 5] = psival
		pass
	return xy

def main(argv):
	"""
	Computes FrRe.psi for n (p*(n//p) to be precise) samples at mesh density d.
	The work is split into p subprocesses, each does n//p.
	"""
	# command line arguments
	usage = 'generate_data_random_5.py -n SAMPLES -d MESH_DENSITY -p PROCESSES -o "outputfile"'
	n = 0 # samples
	d = 0 # mesh density
	p = 0 # processes
	xyfile = '' # outputfile
	try:
		opts, args = getopt.getopt(argv, "hn:d:p:o:")
	except getopt.GetoptError:
		print(usage)
		sys.exit(2)
	if len(opts) == 0:
		print(usage)
		sys.exit(2)	
	for opt, arg in opts:
		if opt == '-h':
			print(usage)
			sys.exit()
		elif opt == '-n':
			n = int(arg)
		elif opt == '-d':
			d = int(arg)
		elif opt == '-p':
			p = int(arg)
		elif opt == '-o':
			xyfile = arg

	# multiprocessing
	start = time.time()
	pool = Pool(processes=p)
	results = [pool.apply_async(compute_psi, args=(n//p, d)) for _ in range(p)]
	results = [r.get() for r in results]
	output = np.concatenate(results, axis=0)
	pool.close()
	
	# write output
	np.savetxt(xyfile, output, delimiter=' ', fmt='%.6f')
	
	# write to log file
	end = time.time()
	print('Time taken in hours:', (end-start)/3600)

if __name__ == '__main__':
	main(sys.argv[1:])
