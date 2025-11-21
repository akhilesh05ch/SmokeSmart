import multiprocessing
import os

# Bind to the port provided by Render
bind = f"0.0.0.0:{os.getenv('PORT', '5000')}"

# Use only 1 worker to save memory (Render free tier has only 512MB)
workers = 1

# Worker class - sync is more memory efficient than async
worker_class = "sync"

# Threads per worker
threads = 2

# Timeout - increase to 120 seconds for model loading
timeout = 120

# Max requests per worker before restart (to prevent memory leaks)
max_requests = 100
max_requests_jitter = 10

# Preload app to share memory between workers
preload_app = False  # Set to False to delay model loading

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Memory optimization
worker_tmp_dir = "/dev/shm"  # Use RAM disk for temporary files
