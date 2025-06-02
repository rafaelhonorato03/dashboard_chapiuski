import os

bind = "0.0.0.0:" + str(os.environ.get("PORT", 8050))
workers = 4
threads = 2
timeout = 120
worker_class = "gthread" 