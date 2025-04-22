import numpy as np
from formula_env import FormulaEnv

def test_formula_env_with_rendering():
    # Initialize the environment
    env = FormulaEnv(render_mode="human")  # Enable rendering
    
    # Reset the environment
    observation = env.reset()
    assert isinstance(observation, dict), "Observation should be a dictionary."
    assert "position" in observation, "Observation should contain 'position'."
    assert "velocity" in observation, "Observation should contain 'velocity'."
    assert "yaw" in observation, "Observation should contain 'yaw'."
    assert "steer_angle" in observation, "Observation should contain 'steer_angle'."
    assert "left_cones" in observation, "Observation should contain 'left_cones'."
    assert "right_cones" in observation, "Observation should contain 'right_cones'."
    assert "start_cones" in observation, "Observation should contain 'start_cones'."
    
    # Take a few random steps and render
    for _ in range(25):
        action = {
            "steering_velocity": np.array([np.random.uniform(-np.pi/2, np.pi/2)]),
            "acceleration": np.array([10]),
        }
        observation, reward, done, _, _ = env.step(action)
                 
        # Check observation
        assert isinstance(observation, dict), "Observation should be a dictionary."
        assert "position" in observation, "Observation should contain 'position'."
        assert "velocity" in observation, "Observation should contain 'velocity'."
        assert "yaw" in observation, "Observation should contain 'yaw'."
        assert "steer_angle" in observation, "Observation should contain 'steer_angle'."
        assert "left_cones" in observation, "Observation should contain 'left_cones'."
        assert "right_cones" in observation, "Observation should contain 'right_cones'."
        assert "start_cones" in observation, "Observation should contain 'start_cones'."
        
        # Check reward
        assert isinstance(reward, float), "Reward should be a float."
        
        # Check done flag
        assert isinstance(done, bool), "Done should be a boolean."
        
        # Render the environment
        
        if done:
            print("Episode finished. Resetting environment.")

    
    # Close the environment
    env.close()
    print("All tests passed with rendering!")

if __name__ == "__main__":
    test_formula_env_with_rendering()