
repo_path = '~/__EVIP__/repo/evip'
branch_name = 'devel-next'

# Number of commits to process in git log
max_commits = 10

# How long a commit is displayed, how long it is stable (no animation)
sec_per_commit = 8
sec_per_stable = 7.5

# Size of the drawing canvas
canvas_width = 960
canvas_height = 540

# Minimum/maximum angle of circle section in which the tree is drawn
min_angle = 90
max_angle = 270

# Source tree is animated (1) or not (0)
dyn_angle = 1

# Radius of circle in which the source tree is drawn
radius = 200

# Leaves of changed files with greater radius (1) or not (0)
dyn_rad = 1

# Dot size, highlighted and base
dot_base_r = 2
dot_high_r = 10	

# X,Y starting position and vertical spacing of the commit labels
commit_x = 10
commit_y = 30
commit_dy = 30

# Width, height and X position (relative to commits) of selection frame
commit_frame_dx = -25
commit_frame_w = 300
commit_frame_h = commit_dy

# X starting position and vertical spacing of the commit labels
file_x = 650
file_dy = 30

# X,Y starting position and vertical spacing of the commit labels
detail_x = commit_x
detail_y = (canvas_height - 60)
detail_dy = commit_dy

# Connector: start and first turn X (relative to files), slope for up/down, circle radius, drawing time
conn_dx = 10
conn_turn_dx = 20
conn_slope = 0.3
conn_r = dot_high_r + 2 
conn_sec = 0.3

# Base and highlighted color of the source tree
tree_base_color = '#304050'
tree_high_color = 'cyan'


