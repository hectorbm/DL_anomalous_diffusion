import os

# Global worker params
worker_id = 1
num_workers = 1
worker_mode = False
env_vars_error = True
try:
    worker_id = int(os.environ['WORKER_ID'])
    num_workers = int(os.environ['NUM_WORKERS'])
    env_vars_error = False
except KeyError:
    print('Please set OS environment variables for worker mode')
