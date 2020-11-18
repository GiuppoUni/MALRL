import matplotlib.pyplot as plt

from tkinter import filedialog
from tkinter import *
import cv2, os
import numpy as np
import utils
from airsim.utils import to_eularian_angles
from airsim.types import Quaternionr
env_cfg_filepath = 'environments'
env_folder = "./"
filename = utils.map_filename
env_cfg = utils.read_cfg(config_filename=env_folder + 'config.cfg')


filename_split = os.path.split(filename)
folder = filename_split[0]
floor_cfg_filepath = folder+'/config.cfg'
player_x_env = 5
player_y_env = 5



# root = Tk()
# # To let user select
# # filename =filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("PNG files","*.PNG"),("png files","*.png"),("All files","*.*"))   )
# root.destroy()

# coords =[]
# fig = plt.figure(figsize=(10,10))
# ax = fig.add_subplot(111)
# floor_image = cv2.imread(filename)
# floor_image = cv2.cvtColor(floor_image, cv2.COLOR_BGR2RGB)



# plt.imshow(floor_image)

# def onclick(event):
#     ix, iy = event.xdata, event.ydata

#     o_x=            	304.4
#     o_y=             	431
#     alpha=          	13.40219092
#     x_unreal = (ix - o_x)/ alpha
#     y_unreal = (iy - o_y) / alpha

#     x_env = 100 * x_unreal+player_x_env
#     y_env = 100 * y_unreal+player_y_env

#     coords.append((ix, iy))

#     global text 
#     if len(coords) > 1:
#         text.remove()
#     text_str = 'Image coordinates \nx: ' + str(np.round(ix,2)) + '\ny:' + str(np.round(iy,2))
#     text_str += '\n\nUnreal coordinates \nx: ' + str(np.round(x_unreal, 2)) + '\ny:' + str(np.round(y_unreal, 2))
#     text_str += '\n\nEnvironment coordinates \nx: ' + str(np.round(x_env, 2)) + '\ny:' + str(np.round(y_env, 2))
#     print(text_str)
#     print('-'*50)
#     text = ax.text(25, 275, text_str, style='italic',
#                            bbox={'facecolor': 'white', 'alpha': 0.5})
#     plt.axis('off')
#     plt.show()

#     return coords
# cid = fig.canvas.mpl_connect('button_press_event', onclick)
# plt.show()



def initialize_nav_fig(env_cfg):
    if not os.path.exists(env_folder + 'results'):
        os.makedirs(env_folder + 'results')

    # Mapping floor to 0 height
    f_z = env_cfg.floor_z / 100.0
    c_z = (env_cfg.ceiling_z - env_cfg.floor_z) / 100.0
    p_z = (env_cfg.player_start_z - env_cfg.floor_z) / 100.0

    plt.ion()
    fig_z = plt.figure()
    ax_z = fig_z.add_subplot(111)
    line_z, = ax_z.plot(0, 0)
    ax_z.set_ylim(0, c_z)
    plt.title("Altitude variation")

    # start_posit = self.client.simGetVehiclePose()

    fig_nav = plt.figure()
    ax_nav = fig_nav.add_subplot(111)
    img = plt.imread(env_folder + utils.map_filename)
    ax_nav.imshow(img)
    plt.axis('off')
    plt.title("Navigational map")
    plt.plot(env_cfg.o_x, env_cfg.o_y, 'b*', linewidth=20)
    nav, = ax_nav.plot(env_cfg.o_x, env_cfg.o_y)
    
    return p_z, f_z, fig_z, ax_z, line_z, fig_nav, ax_nav, nav



class NavMapper():
    def __init__(self,client):
        self.client = client
        self.p_z, self.f_z, self.fig_z, self.ax_z, self.line_z, self.fig_nav, self.ax_nav, self.nav = initialize_nav_fig(env_cfg)
        self.nav_text = self.ax_nav.text(0, 0, '')
        
        self.posit = dict()
        self.distance = dict()
        for name in client.drones_names: self.distance[name] = 0
        self.old_posit = dict()
        self.altitude = dict()
        for name in client.drones_names: self.altitude[name] = [] 
        
        self.nav_x = []
        self.nav_y = []
        




    def update_nav_fig(self):
        # if simGetCollisionInfo.has_collided == True:
        if False:
            print('Drone collided')
            print("Total distance traveled: ", np.round(self.distance[name_agent], 2))
            active = False
            self.client.moveByVelocityAsync(vx=0, vy=0, vz=0, duration=1, vehicle_name=name_agent).join()

            if nav_x:  # Nav_x is empty if the drone collides in first iteration
                ax_nav.plot(nav_x.pop(), nav_y.pop(), 'r*', linewidth=20)

        else:
            for name_agent in self.client.drones_names:
                self.posit[name_agent] = self.client.simGetVehiclePose(vehicle_name=name_agent)
                if name_agent not in self.old_posit:
                    # TODO optimize it
                    self.old_posit[name_agent] =  self.posit[name_agent]  

                self.distance[name_agent] = self.distance[name_agent] + np.linalg.norm(np.array(
                    [self.old_posit[name_agent].position.x_val - self.posit[name_agent].position.x_val,
                        self.old_posit[name_agent].position.y_val - self.posit[name_agent].position.y_val]))
                # self.altitude[name_agent].append(-self.posit[name_agent].position.z_val+p_z)
                self.altitude[name_agent].append(-self.posit[name_agent].position.z_val - self.f_z)

                quat = Quaternionr(self.posit[name_agent].orientation.x_val, self.posit[name_agent].orientation.y_val,
                        self.posit[name_agent].orientation.z_val, self.posit[name_agent].orientation.w_val)
                yaw = to_eularian_angles(quat)[2]

                x_val = self.posit[name_agent].position.x_val
                y_val = self.posit[name_agent].position.y_val
                z_val = self.posit[name_agent].position.z_val

                self.nav_x.append(env_cfg.alpha * x_val + env_cfg.o_x)
                self.nav_y.append(env_cfg.alpha * y_val + env_cfg.o_y)
                self.nav.set_data(self.nav_x, self.nav_y)
                self.nav_text.remove()
                self.nav_text = self.ax_nav.text(25, 55, 'Distance: ' + str(np.round(self.distance[name_agent], 2)),
                                        style='italic',
                                        bbox={'facecolor': 'white', 'alpha': 0.5})

                self.line_z.set_data(np.arange(len(self.altitude[name_agent])), self.altitude[name_agent])
                self.ax_z.set_xlim(0, len(self.altitude[name_agent]))
                self.fig_z.canvas.draw()
                self.fig_z.canvas.flush_events()

                
                self.old_posit[name_agent] = self.posit[name_agent]

                # Verbose and log making
                s_log = 'Position = ({:<3.2f},{:<3.2f}, {:<3.2f}) Orientation={:<1.3f}'.format(
                    x_val, y_val, z_val, yaw
                )

                print(s_log)


if __name__ == "__main__":

    p_z, self.f_z, fig_z, ax_z, line_z, fig_nav, ax_nav, nav = initialize_nav_fig(env_cfg)
