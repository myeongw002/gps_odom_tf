import random

def sample_text_file(input_file, output_file):
    # Read all lines from the input file
    with open(input_file, 'r') as file:
        lines = file.readlines()
    
    # Determine the number of lines to sample (1/3 of the total lines)
    num_samples = len(lines) // 4

    # Randomly sample one-third of the lines
    sampled_lines = random.sample(lines, num_samples)
    
    # Write the sampled lines to the output file
    with open(output_file, 'w') as file:
        file.writelines(sampled_lines)

# Usage
input_file_path = "/home/team-miracle/ROS/catkin_ws/src/gps_odom_tf/path/path.txt"
output_file_path = "/home/team-miracle/ROS/catkin_ws/src/gps_odom_tf/path/path_sample.txt"
sample_text_file(input_file_path, output_file_path)
