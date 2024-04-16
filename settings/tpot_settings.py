import os


# PC2 ===================
res_plan = {'duration': '3h5m',
            'rset': '1',
            'ncpus': '16',
            'mem': '96g',
            'vmem': '96g'}


# Times ==================
timeout_mins = 180       # minutes per pipeline
# periodically check for results
sleep_period = 60 * 5   # seconds
# consider tolerance for timeout without response
# time_without_response > res_plan['duration'] !!!!
time_without_response = int(timeout_mins + (3 * sleep_period/60))    # minutes


# TPOT ====================
random_state = 62
parallel_eval_count = 10
cache_dir = os.environ['PC2PFS'] + "/hpc-prf-neunet/riltner/cache/"
checkpoints_dir = os.environ['PC2PFS'] + "/hpc-prf-neunet/riltner/checkpoints/"


# Dataset =================
features_filter = ['profile_id', 'u_q', 'i_d', 'ambient', 'coolant', 'i_q', 'motor_speed', 'torque', 'u_d']

# Target order: 'pm', 'stator_yoke', 'stator_tooth', 'stator_winding'
# Keep this order when setting targets! Otherwise, individual cv scores per target might have a wrong order.
targets_filter = ['stator_winding']