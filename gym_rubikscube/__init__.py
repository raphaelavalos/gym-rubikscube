from gym.envs.registration import register

register(id='RubiksCube3x3-v1',
         entry_point='gym_rubikscube.envs:Rubiks3x3Env'
         )

register(id='RubiksCube3x3Goal-v1',
         entry_point='gym_rubikscube.envs:Rubiks3x3GoalEnv'
         )
