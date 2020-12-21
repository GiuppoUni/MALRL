from gym.envs.registration import register
import pandas



df = pandas.read_csv("fixed_goals.csv", index_col='name')
# print(df)
fixed_goals = df.to_numpy()

df = pandas.read_csv("init_pos.csv", index_col='name')
# print(df)
fixed_init_pos_list = df.to_numpy()

register(
     id='uav-maze-v0',
     entry_point='gym_maze.envs:MazeEnv',
     max_episode_steps=1000,
kwargs={
    "maze_file":"maze2d_004.npy",
    "enable_render":False,
    "do_track_trajectories":True,"num_goals":None, "measure_distance" : True,
    "verbose" : False,"n_trajs":1,"random_pos" : False,"seed_num" : 12,
    "fixed_goals" : fixed_goals,
    "fixed_init_pos": fixed_init_pos_list[0]
}
)



# register(
#     id='maze-v0',
#     entry_point='gym_maze.envs:MazeEnvSample5x5',
#     max_episode_steps=2000,
# )
# register(
#     id='maze-sample-5x5-v0',
#     entry_point='gym_maze.envs:MazeEnvSample5x5',
#     max_episode_steps=2000,
# )

# register(
#     id='maze-random-5x5-v0',
#     entry_point='gym_maze.envs:MazeEnvRandom5x5',
#     max_episode_steps=2000,
#     nondeterministic=True,
# )

# register(
#     id='maze-sample-10x10-v0',
#     entry_point='gym_maze.envs:MazeEnvSample10x10',
#     max_episode_steps=10000,
# )

# register(
#     id='maze-random-10x10-v0',
#     entry_point='gym_maze.envs:MazeEnvRandom10x10',
#     max_episode_steps=10000,
#     nondeterministic=True,
# )

# register(
#     id='maze-sample-3x3-v0',
#     entry_point='gym_maze.envs:MazeEnvSample3x3',
#     max_episode_steps=1000,
# )

# register(
#     id='maze-random-3x3-v0',
#     entry_point='gym_maze.envs:MazeEnvRandom3x3',
#     max_episode_steps=1000,
#     nondeterministic=True,
# )


# register(
#     id='maze-sample-100x100-v0',
#     entry_point='gym_maze.envs:MazeEnvSample100x100',
#     max_episode_steps=1000000,
# )

# register(
#     id='maze-random-100x100-v0',
#     entry_point='gym_maze.envs:MazeEnvRandom100x100',
#     max_episode_steps=1000000,
#     nondeterministic=True,
# )

# register(
#     id='maze-random-10x10-plus-v0',
#     entry_point='gym_maze.envs:MazeEnvRandom10x10Plus',
#     max_episode_steps=1000000,
#     nondeterministic=True,
# )

# register(
#     id='maze-random-20x20-plus-v0',
#     entry_point='gym_maze.envs:MazeEnvRandom20x20Plus',
#     max_episode_steps=1000000,
#     nondeterministic=True,
# )

# register(
#     id='maze-random-30x30-plus-v0',
#     entry_point='gym_maze.envs:MazeEnvRandom30x30Plus',
#     max_episode_steps=1000000,
#     nondeterministic=True,
# )
