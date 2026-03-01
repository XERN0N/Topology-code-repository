import time
import sys

class TicToc():
    """Implements tic toc timing with labels"""
    def __init__(self,pt=True):
        """Set pt to true to enable printing of the timing results"""
        self.startTimes={}
        self.print_timing=pt

    def tic(self,label=''):
        """Stores start time at label"""
        self.startTimes[label] = time.time()

    def toc(self,label=''):
        """Calculates end-start time for label and prints the result if print_timing is True"""
        try:
            t0=self.startTimes[label]
        except (KeyError):
            t0=self.startTimes['']
        dt=time.time()-t0
        if self.print_timing:
            print(label,"Elapsed time is %.4f s."%dt)
        sys.stdout.flush
        return dt
