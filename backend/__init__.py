from . import backend_py

try:
  from . import backend_cpp

except (ModuleNotFoundError, ImportError) as e:
  import logging
  _logger = logging.getLogger(__name__)
  _logger.warning(f"The c++ backend could not be imported, falling back to python!\nReason:\n\n{e}\n\n")

  # Might be handy to kill the program if this step fails ...
  # print(f"Reason - {type(e)}:", e)
  # import sys
  # sys.exit(1)

  # Import python backend but change its name:
  # - in the example both files have the same signature (e.g. function names)
  # - the code stays intact even if the compilation step in CFFI fails
  # - only the runtime changes (hopefully)
  from . import backend_py as backend_cpp

# Imports related to the data and visualization
from . import visualization
from . import helpers

# Only import desired stuff with 'from backend import *'
__all__ = ["backend_py",
           "backend_cpp",
           "visualization", 
           "helpers"]
