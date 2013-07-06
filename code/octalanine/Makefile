structures/mutant.minimized.pdb: minimize.py structures/mutant.pdb
	cd structures && python minimize.py mutant.pdb

octalanine.minimized.pdb: minimize.py structures/octalanine.pdb
	cd structures && python minimize.py octalanine.pdb

trajectories/mutant_long_traj.h5: run_long.py structures/mutant.minimized.pdb
	python run_long.py structures/mutant.minimized.pdb trajectories/mutant_long_traj.h5

trajectories/wildtype_long_traj.h5: run_long.py structures/octalanine.minimized.pdb
	python run_long.py structures/octalanine.minimized.pdb trajectories/wildtype_long_traj.h5
