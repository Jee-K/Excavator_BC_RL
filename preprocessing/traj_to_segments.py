import numpy as np
from urdf_skeleton_custom import CustomURDFSkeleton

def get_theta_rh_from_control(joints : np.ndarray) -> tuple[np.float64, np.float64, np.float64]:

  theta = joints[0]

  urdf_path = "./excavatorURDF/excavator_lowpoly_locked_splitbucket.urdf"
  skeleton = CustomURDFSkeleton(urdf_path)

  # Define arm joints
  arm_joints = ["lower_arm", "upperToLow", "scoop1"]

  # Define keypoints (full body): (name, link, offset)
  # TODO we'd like this to be sourced... a different way. Even if they're baked permanently into the urdfskeleton class, that might still be better
  keypoints = [
    ("frame_front_mid", "compact_excavator_frame_body", np.array([0.800000, 0.000000, -0.000000])),
    ("frame_rear_mid", "compact_excavator_frame_body", np.array([-0.800000, -0.000000, 0.000000])),
    ("turret_center", "compact_excavator_turret_cabin_roller", np.array([-0.250000, -0.000000, -0.050000])),
    ("arm_base", "part01_pin_1", np.array([0.040000, 0.100000, -0.320000])),
    ("boom_tip", "upper_boom", np.array([-0.654189, 2.211457, -0.040228])),
    ("stick_tip", "lower_boom", np.array([-0.025818, -1.040921, -0.095355])),
    ("bucket_floor", "bucketry", np.array([0.027431, 0.268836, 0.095707])),
    ("bucket_tip", "bucketry", np.array([-0.357516, 0.783560, 0.096045])),
  ]

  skeleton.define_skeleton(arm_joints, keypoints)
  keypoints_3d = skeleton.forward_kinematics(joints[1:])

  bucket_tip = keypoints_3d[-1]

  # import plotly.graph_objects as go

  # fig = go.Figure()
  # KEYPOINT_ORDER = [
  #   'frame_front_mid',
  #   'frame_rear_mid',
  #   'turret_center',
  #   'arm_base',
  #   'boom_tip',
  #   'stick_tip',
  #   'bucket_floor',
  #   'bucket_tip',
  # ]
  # kp = keypoints_3d

  # fig.add_trace(go.Scatter3d(
  #   x=kp[:, 0], y=kp[:, 1], z=kp[:, 2],
  #   mode='markers+text',
  #   marker=dict(size=8, color='red', line=dict(width=1, color='black')),
  #   text=KEYPOINT_ORDER,
  #   textposition='top center',
  #   textfont=dict(size=8),
  #   name='Keypoints',
  #   hovertemplate='<b>%{text}</b><br>X:%{x:.3f}<br>Y:%{y:.3f}<br>Z:%{z:.3f}<extra></extra>',
  # ))
  # fig.show()

  # print(bucket_tip, np.sqrt(np.square(bucket_tip[0]) + np.square(bucket_tip[1])), bucket_tip[2])

  return (np.float64(theta), np.sqrt(np.square(bucket_tip[0]) + np.square(bucket_tip[1])), -bucket_tip[2])

# print(get_theta_rh_from_control([0, .4, .3, 1.0])) # flat, slightly below excavator
# print(get_theta_rh_from_control([0, .4, .3, 0.9])) # flat, slightly more below excavator
# print(get_theta_rh_from_control([0, .0, .3, .5])) # pushing out like an extended hangman, fairly high
# print(get_theta_rh_from_control([23, -.9, -.45, .5])) # crunched in to the max


# given Q = {q_t}, Sk = <urdf, stl, etc>
# return first Di = {<t, theta, r, h>}, Du = {<t, theta, r, h>}
# then develop Se = <P_i less 50, Di_i, Du_i, Di_i+1, Q in [i less lead frames, i+1 less close frames]>
# which are <starting position, goal dig, goal dump, goal reset (next dig), motion in that area>

def build_segments(trajectories: np.ndarray, lead_frames_before_dig : int = 50, close_frames_before_reset : int = 0, bucket_flat_angle : float = 1.1) -> tuple:
  """
  With trajectories in Nx[base, boom, arm, scoop], lead frames and close frames being nonnegative ints, and bucket_flat_angle being a float-like representing radians
  """

  # TODO: beware new trajectories shape

  candidates = []

  for idx, step in enumerate(trajectories[1:-1]):

    # step consists of [turret, main boom, lower to upper, scoop]
    # [..., down, up, in] are the positive directions
    last_step_angle = trajectories[idx - 1][1] - step[idx - 1][2] + step[idx - 1][3]
    step_angle = step[1] - step[2] + step[3]
    next_step_angle = trajectories[idx + 1][1] - step[idx + 1][2] + step[idx + 1][3]


    # dig heuristic
    if step_angle >= bucket_flat_angle and (last_step_angle < step_angle >= next_step_angle):
      candidates.append((idx, "Dig"))

    # dump heuristic
    elif step_angle <= bucket_flat_angle and (last_step_angle > step_angle <= next_step_angle):
      candidates.append((idx, "Dump"))

  # add the last position as a false-dig which will serve as a return for the last segment, if no good dig exists
  # strictly speaking, this is a shortcut to avoid a loss of training data which may otherwise be useful even when
  # a well formed end goal does not exist
  candidates.append((len(trajectories), "Dig"))

  # form segments
  segments : list[tuple[int, int, int]] = []

  idx = 0
  while idx < len(candidates) and candidates[idx][1] != 'Dig':
    idx += 1

  last_dig_idx = idx
  idx += 1

  while idx < len(candidates):
    while idx < len(candidates) and candidates[idx][1] != 'Dump':
      idx += 1

    if idx >= len(candidates):
      break

    dump_idx = idx

    while idx < len(candidates) and candidates[idx][1] != 'Dig':
      idx += 1

    if idx >= len(candidates):
      break

    segments.append((last_dig_idx, dump_idx, idx))
    last_dig_idx = idx

  labeled_segments = []

  for dig_idx, dump_idx, reset_idx in segments:

    start_idx = min(0, dig_idx - lead_frames_before_dig)
    seq_end_idx = int(np.clip(reset_idx + 1 - close_frames_before_reset, 0, len(trajectories)))


    labeled_segments.append((get_theta_rh_from_control(trajectories[dig_idx]), get_theta_rh_from_control(trajectories[dump_idx]), get_theta_rh_from_control(trajectories[reset_idx]), trajectories[start_idx:seq_end_idx]))
  
  return labeled_segments

  

# TODO: implement a dataset aggregation
# turret rotation offset?
# relative angle rather than true angle?













