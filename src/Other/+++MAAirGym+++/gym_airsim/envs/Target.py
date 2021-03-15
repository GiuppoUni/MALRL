import logging
import numpy as np

import utils

class Target:
    def __init__(self,id,lat,lon,z=0):
        self.id = id
        self.name = "Target_" + str(id)
        self.lat = lat 
        self.lon = lon
        self.z = z

        self.x_val , self.y_val , self.z_val  = utils.lonlatToAirSim(lon,lat,z,
                                # utils.env_cfg.o_x,utils.env_cfg.o_y,utils.env_cfg.o_z)
                                    0,0,0)

        self.isLocalized = False
        self.isClassified = False

    def setLocalized(self,bool):
        self.isLocalized = bool
    def setClassified(self,bool):
        self.isClassified = bool


class TargetManager:
    def __init__(self, n_targets):
        self.targets = {}
        self.num_targets = n_targets
        self.ts_allocated = {}
        self.ts_unallocated = {}
        self.resetTargets()

    def getGPSCoo(self,i):
        if i == 0:
            return 12.459601163864138, 41.902277040963696
        if i == 1:
            return  12.462015151977539, 41.903239263785025
        if i == 2:
            return 12.459413409233095, 41.90383415777753
        else:
            return np.random.uniform(12.45,12.47,1) , np.random.uniform(41.90,4.95,1) 
    
    def resetTargets(self):
        self.targets = {}
        for i in range(self.num_targets):
            _coo = self.getGPSCoo(i)
            _t = Target(i,lon = _coo[0], lat = _coo[1])
            self.targets[i] = _t 
        self.ts_unallocated = self.targets
        self.ts_allocated = {}

    def reset_targets_status(self):
        # for t in self.targets:
        #     t.isLocalized = False
        #     t.isClassified = False
        self.ts_unallocated = self.targets
        self.ts_allocated = {}
        

    def allocate_target(self,drone_name,target_id):
        self.ts_allocated[target_id] = {
            "ts_obj":self.targets[target_id],
            "drone_name":drone_name
            }

    def unallocate_target(self,target_id):
        self.ts_allocated.pop(target_id)



    # TODO generate the mesh for current target in unreal editor 
    def ue_generate_mesh(self,client):
        # # List of returned meshes are received via this function
        # meshes=client.simGetMeshPositionVertexBuffers()


        # index=0

        # for m in meshes:
        #     print(m)
        #     # Finds solar panel mesh in the StreetMapProject environment
        #     if 'solar' in m.name:
        #         # add/move mesh inside block
        #         break
        pass