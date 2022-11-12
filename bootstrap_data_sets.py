import csv
import multiprocessing
import os
import time

from tqdm.contrib.concurrent import thread_map

from boostrap_helper import BootstrapHelper
from full_body_pose_embedder import FullBodyPoseEmbedder
from pose_classification import PoseClassifier

# Entry Point

# train all models per level
difficulty_level = ['beginner', 'intermediate', 'advanced']

# test only specific model level (i.e., intermediate)
# difficulty_level = ['advanced']

csv_output_data_set = "guru_asana_pose_output_csv"
output_data_set = 'guru_asana_data_sets_out'
input_data_set = 'guru_asana_data_sets_in'


def train_normally():
    for level in difficulty_level:
        if not os.path.exists(input_data_set):
            raise FileNotFoundError("File not found. Add data sets to [guru_asana_data_sets_in\\] folder")
        train_data(level)


def train_in_parallel():
    workers = multiprocessing.cpu_count() * 4
    thread_map(train_data, difficulty_level, max_workers=workers)


def train_data(level):
    csv_out_folder = os.path.join(csv_output_data_set, level)
    dataset_folder = os.path.join(input_data_set, level)

    # Create directory for outputs
    for directory in [csv_output_data_set, output_data_set]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Initialize helper
    bootstrap_helper = BootstrapHelper(
        difficulty_level=level,
        data_set_folder=dataset_folder,
        per_level_out_folder=os.path.join(output_data_set, level),
        csvs_out_folder=csv_out_folder,
    )

    # Check how many pose classes and images for them are available.
    bootstrap_helper.print_images_in_statistics()

    # buffer time to log bootstrap
    time.sleep(0.1)

    # Bootstrap all images.
    # Set limit to some small number for debug.
    bootstrap_helper.bootstrap(per_pose_class_limit=None)

    # Check how many images were bootstrapped.
    bootstrap_helper.print_images_out_statistics()

    # After initial bootstrapping images without detected poses were still saved in
    # the folder (but not in the CSVs) for debug purpose. Let's remove them.
    print("Removing undetected poses")
    bootstrap_helper.align_images_and_csvs(print_removed_items=True, difficulty_level=level)
    bootstrap_helper.print_images_out_statistics()

    # Align CSVs with filtered images.
    print("Aligning CSVs with filtered images")
    bootstrap_helper.align_images_and_csvs(print_removed_items=True, difficulty_level=level)
    bootstrap_helper.print_images_out_statistics()

    # Initialize pose landmarks into embedding for transforms and outliers.
    pose_embedder = FullBodyPoseEmbedder()

    # Classifies give pose against database of poses.
    pose_classifier = PoseClassifier(
        pose_samples_folder=csv_out_folder,
        pose_embedder=pose_embedder,
        top_n_by_max_distance=30,
        top_n_by_mean_distance=10)

    # keep old reference
    outliers = pose_classifier.find_pose_sample_outliers()
    if len(outliers) > 0:
        print('Number of outliers: ', len(outliers))

    # pose_classifier.display_pose_sample(bootstrap_helper)
    # Analyze outliers.
    bootstrap_helper.analyze_outliers(outliers)

    # Remove all outliers (if you don't want to manually pick).
    bootstrap_helper.remove_outliers(outliers)

    # Align CSVs with images after removing outliers.
    print("Aligning CSVs with images after removing outliers")
    bootstrap_helper.align_images_and_csvs(print_removed_items=True, difficulty_level=level)
    bootstrap_helper.print_images_out_statistics()

    # Display old defective outliers
    bootstrap_helper.analyze_outliers(outliers, os.path.join(output_data_set, level))

    # then dump each level difficulty to finalize csv
    dump_joint_coordinates(csv_out_folder, level)


def dump_joint_coordinates(csv_out_folder, level):
    # Each file in the folder represents one pose class.
    trained_poses_folder = os.path.join('trained_poses_data_sets')

    if not os.path.exists(trained_poses_folder):
        os.makedirs(trained_poses_folder)

    file_names = [name for name in os.listdir(csv_out_folder) if name.endswith('csv')]
    with open(os.path.join(trained_poses_folder, level + '.csv'), 'w', newline='') as csv_out:
        csv_out_writer = csv.writer(csv_out, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        for file_name in file_names:
            # Use file name as pose class name.
            class_name = file_name[:-(len('csv') + 1)]
            # One file line: `sample_00001,x1,y1,x2,y2,....`.
            with open(os.path.join(csv_out_folder, file_name)) as csv_in:
                csv_in_reader = csv.reader(csv_in, delimiter=',')
                for row in csv_in_reader:
                    row.insert(1, class_name)
                    csv_out_writer.writerow(row)


# driver
train_normally()
