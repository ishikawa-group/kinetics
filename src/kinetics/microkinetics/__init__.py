import warnings
import logging

warnings.filterwarnings("ignore")

# Suppress M3GNet INFO messages
logging.getLogger("m3gnet").setLevel(logging.WARNING)