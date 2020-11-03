class Fleet(num_drones=0):
    def __init__(self):
        self.num_drones = num_drones
        self.drones = dict()

    def add_drone(self, drone_name):
        self.num_drones += 1
        drone = Drone(drone_type=drone_type, name=drone_name)
        self.drones[drone_name] = drone