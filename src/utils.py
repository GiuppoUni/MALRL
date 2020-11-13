import os 
import numpy as np
import cv2
import airsim
import time

def enable_control(client, drones):
    for vehicle in list(drones):
        client.enableApiControl(True, vehicle)
        client.armDisarm(True, vehicle)

def takeoff_all_drones(client, drone_names: list) -> None:
    """
       Make all vehicles takeoff, one at a time and return the
       pointer for the last vehicle takeoff to ensure we wait for
       all drones
    """
    vehicle_pointers = []
    for drone_name in drone_names:
        vehicle_pointers.append( client.takeoffAsync(vehicle_name=drone_name) )
    # All of this happens asynchronously. Hold the program until the last vehicle
    # finishes taking off.
    return vehicle_pointers[-1]

def get_all_drone_positions(client, drones: list) -> np.array:
    for i, drone_name in enumerate(list(drones)):
        state_data = client.getMultirotorState(vehicle_name=drone_name)
        print(state_data)
        drones[drone_name].pos_vec3 = position_to_list(state_data.kinematics_estimated.position)
        drones[drone_name].gps_pos_vec3 = gps_position_to_list(state_data.gps_location)


def transform_input(responses):
    img1d = np.array(responses[0].image_data_float, dtype=np.float)
    img1d = 255/np.maximum(np.ones(img1d.size), img1d)
    img2d = np.reshape(img1d, (responses[0].height, responses[0].width))

    from PIL import Image
    image = Image.fromarray(img2d)
    im_final = np.array(image.resize((84, 84)).convert('L')) 

    return im_final

def interpret_action(action):
    scaling_factor = 0.25
    if action == 0:
        quad_offset = (0, 0, 0)
    elif action == 1:
        quad_offset = (scaling_factor, 0, 0)
    elif action == 2:
        quad_offset = (0, scaling_factor, 0)
    elif action == 3:
        quad_offset = (0, 0, scaling_factor)
    elif action == 4:
        quad_offset = (-scaling_factor, 0, 0)    
    elif action == 5:
        quad_offset = (0, -scaling_factor, 0)
    elif action == 6:
        quad_offset = (0, 0, -scaling_factor)
    
    return quad_offset

def compute_reward(quad_state, quad_vel, collision_info):
    thresh_dist = 7
    beta = 1

    z = -10
    pts = [np.array([-.55265, -31.9786, -19.0225]), np.array([48.59735, -63.3286, -60.07256]), np.array([193.5974, -55.0786, -46.32256]), np.array([369.2474, 35.32137, -62.5725]), np.array([541.3474, 143.6714, -32.07256])]

    quad_pt = np.array(list((quad_state.x_val, quad_state.y_val, quad_state.z_val)))

    if collision_info.has_collided:
        reward = -100
    else:    
        dist = 10000000
        for i in range(0, len(pts)-1):
            dist = min(dist, np.linalg.norm(np.cross((quad_pt - pts[i]), (quad_pt - pts[i+1])))/np.linalg.norm(pts[i]-pts[i+1]))

        #print(dist)
        if dist > thresh_dist:
            reward = -10
        else:
            reward_dist = (math.exp(-beta*dist) - 0.5) 
            reward_speed = (np.linalg.norm([quad_vel.x_val, quad_vel.y_val, quad_vel.z_val]) - 0.5)
            reward = reward_dist + reward_speed

    return reward

def isDone(reward):
    done = 0
    if  reward <= -10:
        done = 1
    return done



def snap_all_cams(client, vehicle_names):
    responses = dict()
    for v in vehicle_names:
        responses[v] = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.Scene, False, False),
        airsim.ImageRequest("1", airsim.ImageType.DepthVis),  #depth visualization image
        airsim.ImageRequest("2", airsim.ImageType.DepthPlanner,True ),
        airsim.ImageRequest("3", airsim.ImageType.DepthPerspective),
        airsim.ImageRequest("4", airsim.ImageType.DisparityNormalized),
        # airsim.ImageRequest("5", airsim.ImageType.Segmentation),
        # airsim.ImageRequest("6", airsim.ImageType.SurfaceNormals ),
        # airsim.ImageRequest("7", airsim.ImageType.Infrared )
        ])  #scene vision image in uncompressed RGB array
        print(type(responses[v]))
        responses[v].append(client.simGetImage("0", airsim.ImageType.Segmentation))
        responses[v].append(client.simGetImage("0", airsim.ImageType.SurfaceNormals))
        responses[v].append(client.simGetImage("0", airsim.ImageType.Infrared))

        print(v+': Retrieved images in num: %d' % len(responses[v]))
    return responses



def snap_cam(client,vehicles_names,cameraType):
    for v in vehicles_names:
        response = client.simGetImage("0", cameraType)
        tmp_dir = "single_shots"
        filename = os.path.join(tmp_dir, v +"_cam"+str(cameraType))

        try:
            os.makedirs(tmp_dir)
        except OSError:
            if not os.path.isdir(tmp_dir):
                raise
    
        # print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
        img_rgb = cv2.imdecode(airsim.string_to_uint8_array(response), cv2.IMREAD_UNCHANGED)
        cv2.imwrite(os.path.normpath(filename + '.png'), img_rgb) # write to png
        
def save_all_cams(responses,dir="."):
    tmp_dir = os.path.join(dir, "airsim_images"+time.strftime("%Y%m%d-%H%M%S"))
    print ("Saving images to %s" % tmp_dir)
    try:
        os.makedirs(tmp_dir)
    except OSError:
        if not os.path.isdir(tmp_dir):
            raise

    for drone_name in responses:
        for idx,response in enumerate(responses[drone_name]):
            filename = os.path.join(tmp_dir, drone_name +"_cam"+str(idx))

            if type(response) is bytes:
                print("Type",type(response))
                img_rgb = cv2.imdecode(airsim.string_to_uint8_array(response), cv2.IMREAD_UNCHANGED)
                cv2.imwrite(os.path.normpath(filename + '.png'), img_rgb) # write to png
            elif response.pixels_as_float:
                print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
                airsim.write_pfm(os.path.normpath(filename + '.pfm'), airsim.get_pfm_array(response))
            elif response.compress: #png format
                print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
                airsim.write_file(os.path.normpath(filename + '.png'), response.image_data_uint8)
            else: #uncompressed array
                print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
                img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) #get numpy array
                img_rgb = img1d.reshape(response.height, response.width, 3) #reshape array to 3 channel image array H X W X 3
                cv2.imwrite(os.path.normpath(filename + '.png'), img_rgb) # write to png


def snap_and_save_all_cams(client,vehicle_names):
    pass
    