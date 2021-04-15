from gym.envs.registration import register
import pandas




register(
     id='MALRLEnv-v0',
     entry_point='gym_maze.envs:MazeEnv',
     max_episode_steps=1000,
kwargs={
    "maze_file":"maze2d_004.npy",
      "maze_size":(640, 640), 
    "enable_render":True,
    "verbose" : False,"n_trajs":9,"random_start_pos":True,
    "random_goal_pos":True,
    "seed_num" : 12,
   "num_goals":1, 
   "seed_num" : 1,
   "visited_cells" : []
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
