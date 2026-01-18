import logging
import os
from pytz import timezone 
from datetime import datetime

def create_logger(path, api_cls):
    # Ensure path is not None and provide a default
    if path is None:
        path = "logs"
    
    # Convert to string if it's not already
    path = str(path)
    
    os.makedirs(path, exist_ok=True)
    time_str = datetime.now(timezone("Asia/Kolkata")).strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.log'.format(time_str, api_cls)
    final_log_file = os.path.join(path, log_file)

    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger().addHandler(console)

    return logger