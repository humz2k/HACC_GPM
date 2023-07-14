#import sys
#sys.path.append("/home/hqureshi/half-precision-hacc/hacc-halfprecision/pycosmotools")

import pycosmo

class myAnalysis(pycosmo.PyCosmoAnalysis):

    def __init__(self,*args):
        super().__init__(*args)
        print("myAnalysis:")
        print("   ng =",self.ng)
        print("   rl =",self.rl)
        print("   world_rank =",self.world_rank)
        print("   world_size =",self.world_size)
        print("   local_coords =",self.local_coords)
        print("   local_grid_size =",self.local_grid_size)

    def on_initial_conditions(self):
        print("ICs:")
        print(self.particles)
        print(self.power_spectrum)

    def on_analysis_step(self):
        print("Analysis1")
        print(self.particles)

    def on_power_spectrum_step(self):
        print("POWER")
        print(self.power_spectrum)

    def on_final(self):
        print("FINAL")
        print(self.particles)
        print(self.power_spectrum)
        