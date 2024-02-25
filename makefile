sync_ckpts:
	rsync -arv --include='*/' --include='ckpts/***' --include='progress.csv' --exclude='*' se2161@greene.hpc.nyu.edu:/scratch/se2161/pcgrl-jax/saves/ ./saves/