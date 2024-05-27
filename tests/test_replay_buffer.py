from breakout_rl_dqn.replay_buffer import ReplayBuffer

def test_replay_buffer():
    replay_buffer = ReplayBuffer(capacity=100)
    replay_buffer.add_experience(1, 2, 3, 4, False)
    assert len(replay_buffer.buffer) == 1

    state, action, reward, next_state, done = replay_buffer.sample_batch(1)
    assert (state[0], action[0], reward[0], next_state[0], done[0]) == (1, 2, 3, 4, False)
