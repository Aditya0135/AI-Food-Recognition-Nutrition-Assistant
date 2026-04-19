import logging
import os
import sys


logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

logs_dir = "logs"
log_filepath = os.path.join(logs_dir, "running_logs.log")
os.makedirs(logs_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)

for name in ["dsl_registry", "triton", "cutedsl"]:
    ext_logger = logging.getLogger(name)
    ext_logger.disabled = True
    ext_logger.propagate = False
    ext_logger.handlers.clear()

logger = logging.getLogger("AI_Food_Recognition_Nutrition_Assistant")
