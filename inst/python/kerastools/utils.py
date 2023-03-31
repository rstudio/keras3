
import subprocess
from functools import partial

class _patched_subprocess_Popen:
  def __enter__(self):
    self._orig_Popen = subprocess.Popen
    subprocess.Popen = partial(subprocess.Popen, stdin = subprocess.DEVNULL)

  def __exit__(self, *args):
     subprocess.Popen = self._orig_Popen
