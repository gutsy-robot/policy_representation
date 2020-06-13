import h5py
import numpy as np
import os
import argparse


# This assumes the player_id=0 is the shooter and player_id=1 is the bait


def get_opts():
    parser = argparse.ArgumentParser(description='TSF_RL: Space Fortress')
    parser.add_argument('--trajectory_dir', type=str,
                        default="../../new/Team-Space-Fortress/hat/self_play/trajectories/",
                        help='direcctory containing h5 files')

    opts = parser.parse_args()
    return opts


opts = get_opts()
traj_path = opts.trajectory_dir
print(traj_path)
for file in os.listdir(traj_path):
    if file.endswith('.h5'):

        traj_file = h5py.File(traj_path + file, 'r')
        bait_id = file.split('Bait_')[1].split('_')[0]
        shooter_id = file.split('Bait_')[1].split('_', 1)[1].split('.h5')[0]

        bait_states = np.concatenate((traj_file['states'][:, :, :10], traj_file['states'][:, :, 16:19],
                                         traj_file['states'][:, :, 22:25]), axis=2)
        shooter_states = np.concatenate((traj_file['states'][:, :, :4], traj_file['states'][:, :, 10:16],
                                      traj_file['states'][:, :, 19:22],
                                      traj_file['states'][:, :, 25:]), axis=2)

        bait_actions = traj_file['actions'][:, :, :3]
        shooter_actions = traj_file['actions'][:, :, 3:]
        metric = traj_file['metrics']
        try:
            os.mkdir(traj_path + shooter_id + '_' + bait_id)
        except:
            print("exists")

        np.save(traj_path + shooter_id + '_' + bait_id + '/shooter_states.npy',
                shooter_states)
        np.save(traj_path + shooter_id + '_' + bait_id + '/bait_states.npy', bait_states)

        np.save(traj_path + shooter_id + '_' + bait_id + '/shooter_actions.npy', shooter_actions)
        np.save(traj_path + shooter_id + '_' + bait_id + '/bait_actions.npy', bait_actions)
        np.save(traj_path + shooter_id + '_' + bait_id + '/metrics.npy', metric)
