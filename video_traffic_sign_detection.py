import ast

import numpy as np
from roboflow import Roboflow
import supervision as sv
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import os



# Function to process video and save detection results
def process_video(input_video_path, output_video_path, log_file):
    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Total frames in the video: {total_frames}")

    # Define the codec and create a VideoWriter object to save output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    results_log = []
    frame_num = 0

    # Process each frame
    while frame_num < total_frames:
        ret, frame = cap.read()

        if not ret:
            print(f"Error reading frame {frame_num}, skipping.")
            frame_num += 1
            continue

        print(f"Processing frame {frame_num} of {total_frames}")

        # Save the frame temporarily to perform inference on it
        temp_image_path = "temp_frame.jpg"
        cv2.imwrite(temp_image_path, frame)

        # Run Roboflow inference on the frame
        result = model.predict(temp_image_path, confidence=50, overlap=20).json()

        # Check if there are any predictions
        if not result["predictions"]:
            print(f"No detections in frame {frame_num}")
            # Write the unannotated frame to the video if no detections
            out.write(frame)
            plt.imshow(frame)
            frame_num += 1
            continue

        # Extract bounding boxes and class IDs if detections are present
        boxes = np.array([[
            item['x'] - item['width'] / 2,  # x1
            item['y'] - item['height'] / 2,  # y1
            item['x'] + item['width'] / 2,  # x2
            item['y'] + item['height'] / 2   # y2
        ] for item in result["predictions"]])

        class_ids = np.array([item["class_id"] for item in result["predictions"]])
        labels = [item["class"] for item in result["predictions"]]

        # Create detections object
        detections = sv.Detections(xyxy=boxes, class_id=class_ids)

        # Annotators
        label_annotator = sv.LabelAnnotator()
        bounding_box_annotator = sv.BoxAnnotator()

        # Annotate the frame
        annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

        # Save the annotated frame to the output video
        out.write(annotated_frame)
        plt.imshow(annotated_frame)

        # Log the detection results
        for i, box in enumerate(boxes):
            conf = result["predictions"][i]['confidence']
            label = result["predictions"][i]['class']
            results_log.append({
                "frame": frame_num,
                "class": label,
                "confidence": conf,
                "bbox": box.tolist()
            })

        frame_num += 1

    cap.release()
    out.release()

    # Save the detection logs
    df = pd.DataFrame(results_log)
    df.to_csv(log_file, index=False)
    print(f"Processing complete. Log saved to {log_file}")

# Functions to visualize the results
def plot_confidence_scores(df_original, df_corrupted):

    df_original['smoothed_conf'] = df_original['confidence'].rolling(window=50).mean()
    df_corrupted['smoothed_conf'] = df_corrupted['confidence'].rolling(window=50).mean()


    plt.figure(figsize=(10, 6))
    plt.plot(df_original['smoothed_conf'], label='Original Video (Smoothed)', marker='o', markersize=3, color='green')
    plt.plot(df_corrupted['smoothed_conf'], label='Corrupted Video (Smoothed)', marker='x', markersize=3, color='red')
    plt.xlabel('Frame Number')
    plt.ylabel('Confidence Score')
    plt.title('YOLO Confidence Scores Comparison (Smoothed)')
    plt.legend()
    plt.savefig('output_plots/ts_confidence_scores_comparison.png')
    print("Confidence score comparison plot saved as 'output_plots/ts_confidence_scores_comparison.png'")


def plot_detections_histogram(df_original, df_corrupted):

    df_original['detections'] = df_original.groupby('frame')['confidence'].count()
    df_corrupted['detections'] = df_corrupted.groupby('frame')['confidence'].count()


    plt.figure(figsize=(10, 6))
    plt.hist(df_original['detections'], bins=15, alpha=0.5, label='Original Video', color='green')
    plt.hist(df_corrupted['detections'], bins=15, alpha=0.5, label='Corrupted Video', color='red')
    plt.xlabel('Number of Detections per Frame')
    plt.ylabel('Frequency')
    plt.title('Histogram of Detections per Frame')
    plt.legend()
    plt.savefig('output_plots/ts_detections_histogram.png')
    print("Detections histogram saved as 'output_plots/ts_detections_histogram.png'")

# Function to calculate bounding box area
def calculate_bbox_area(df):
    df['bbox'] = df['bbox'].apply(lambda x: ast.literal_eval(x))
    df['bbox_area'] = (df['bbox'].apply(lambda x: x[2] - x[0])) * (df['bbox'].apply(lambda x: x[3] - x[1]))


def plot_number_of_detections(df_original, df_corrupted):

    df_original['detections'] = df_original.groupby('frame')['confidence'].count()
    df_corrupted['detections'] = df_corrupted.groupby('frame')['confidence'].count()
    plt.figure(figsize=(10, 6))
    plt.plot(df_original['frame'], df_original['detections'], label='Original Video Detections', marker='o', markersize=3, color='green')
    plt.plot(df_corrupted['frame'], df_corrupted['detections'], label='Corrupted Video Detections', marker='x', markersize=3, color='red')
    plt.xlabel('Frame Number')
    plt.ylabel('Number of Detections')
    plt.title('Number of Detections per Frame')
    plt.legend()
    plt.savefig('output_plots/ts_number_of_detections.png')
    print("Number of detections per frame plot saved as 'output_plots/ts_number_of_detections.png'")


def plot_confidence_histogram(df_original, df_corrupted):

    plt.figure(figsize=(10, 6))
    plt.hist(df_original['confidence'], bins=30, alpha=0.5, label='Original Video', color='green')
    plt.hist(df_corrupted['confidence'], bins=30, alpha=0.5, label='Corrupted Video', color='red')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title('Histogram of Confidence Scores')
    plt.savefig('output_plots/ts_confidence_histogram.png')
    print("Confidence score histogram saved as 'output_plots/ts_confidence_histogram.png'")


def plot_bbox_area_comparison(df_original, df_corrupted):

    df_original['bbox_area'] = (df_original['bbox'].apply(lambda x: x[2] - x[0])) * (df_original['bbox'].apply(lambda x: x[3] - x[1]))
    df_corrupted['bbox_area'] = (df_corrupted['bbox'].apply(lambda x: x[2] - x[0])) * (df_corrupted['bbox'].apply(lambda x: x[3] - x[1]))
    plt.figure(figsize=(10, 6))
    plt.scatter(df_original['frame'], df_original['bbox_area'], label='Original Video', alpha=0.5, color='green')
    plt.scatter(df_corrupted['frame'], df_corrupted['bbox_area'], label='Corrupted Video', alpha=0.5, color='red')
    plt.xlabel('Frame Number')
    plt.ylabel('Bounding Box Area')
    plt.title('Bounding Box Area Comparison')
    plt.legend()
    plt.savefig('output_plots/ts_bbox_area_comparison.png')
    print("Bounding box area comparison plot saved as 'output_plots/ts_bbox_area_comparison.png'")


def plot_class_distribution_comparison(df_original, df_corrupted):

    class_counts_original = df_original['class'].value_counts()
    class_counts_corrupted = df_corrupted['class'].value_counts()
    plt.figure(figsize=(10, 6))
    class_counts_original.plot(kind='bar', color='green', alpha=0.5, label='Original Video', width=0.4, position=0)
    class_counts_corrupted.plot(kind='bar', color='red', alpha=0.5, label='Corrupted Video', width=0.4, position=1)

    plt.xlabel('Object Class')
    plt.ylabel('Number of Detections')
    plt.title('Class Distribution Comparison')
    plt.legend()
    plt.savefig('output_plots/ts_class_distribution_comparison.png')
    print("Class distribution comparison plot saved as 'output_plots/ts_class_distribution_comparison.png'")


def smooth_and_plot_confidence(df_original, df_corrupted, window_size=10,
                               output_path='output_plots/ts_confidence_scores_comparison.png'):

    def smooth_data(data, window_size):
        return data.rolling(window=window_size, min_periods=1).mean()

    df_original_grouped = df_original.groupby('frame')['confidence'].mean().reset_index()
    df_corrupted_grouped = df_corrupted.groupby('frame')['confidence'].mean().reset_index()

    df_original_grouped['smoothed_conf'] = smooth_data(df_original_grouped['confidence'], window_size)
    df_corrupted_grouped['smoothed_conf'] = smooth_data(df_corrupted_grouped['confidence'], window_size)

    plt.figure(figsize=(10, 6))
    plt.plot(df_original_grouped['frame'], df_original_grouped['smoothed_conf'],
             label='Original Video', marker='o', markersize=4, alpha=0.7, linewidth=1.5, color='green')
    plt.plot(df_corrupted_grouped['frame'], df_corrupted_grouped['smoothed_conf'],
             label='Corrupted Video', marker='x', markersize=4, alpha=0.7, linewidth=1.5, color='red')

    plt.xlabel('Frame Number')
    plt.ylabel('Confidence Score')
    plt.title('YOLO Confidence Scores Comparison (Smoothed)')
    plt.legend()
    plt.grid(True)

    plt.savefig(output_path)
    print(f"Confidence score comparison plot saved as '{output_path}'")

def plot_number_of_detections_per_frame(df_original, df_corrupted,
                                        output_path='output_plots/ts_number_of_detections_per_frame.png'):
    """
    This function plots the number of detections per frame for both original and corrupted videos.

    Parameters:
    df_original (pd.DataFrame): DataFrame containing original video results
    df_corrupted (pd.DataFrame): DataFrame containing corrupted video results
    output_path (str): Path to save the plot image
    """

    all_frames = pd.DataFrame({'frame': range(max(df_original['frame'].max(), df_corrupted['frame'].max()) + 1)})

    df_original_detections = df_original.groupby('frame')['confidence'].count().reset_index(name='detections')
    df_corrupted_detections = df_corrupted.groupby('frame')['confidence'].count().reset_index(name='detections')

    df_original_detections = all_frames.merge(df_original_detections, on='frame', how='left').fillna(0)
    df_corrupted_detections = all_frames.merge(df_corrupted_detections, on='frame', how='left').fillna(0)

    plt.figure(figsize=(10, 6))
    plt.plot(df_original_detections['frame'], df_original_detections['detections'],
             label='Original Video Detections', marker='o', markersize=3, color="green")
    plt.plot(df_corrupted_detections['frame'], df_corrupted_detections['detections'],
             label='Corrupted Video Detections', marker='x', markersize=3, color="red")
    plt.xlabel('Frame Number')
    plt.ylabel('Number of Detections')
    plt.title('Number of Detections per Frame')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    print(f"Number of detections per frame plot saved as '{output_path}'")



# def calculate_weighted_confidence(df_original, df_corrupted):
#     # Step 1: Get the total number of detections per frame in the original video
#     df_original_detections = df_original.groupby('frame')['confidence'].count().reset_index()
#     df_original_detections.columns = ['frame', 'total_detections']
#
#     # Step 2: Get the average confidence score for each frame in the corrupted video
#     df_corrupted_avg_conf = df_corrupted.groupby('frame')['confidence'].mean().reset_index()
#     df_corrupted_avg_conf.columns = ['frame', 'avg_confidence']
#
#     # Step 3: Merge the two dataframes on the 'frame' column
#     df_combined = pd.merge(df_original_detections, df_corrupted_avg_conf, on='frame')
#
#     # Step 4: Calculate the weighted confidence score by dividing average confidence by total detections
#     df_combined['weighted_confidence'] = df_combined['avg_confidence'] / df_combined['total_detections']
#
#     # Output the resulting dataframe
#     print(df_combined[['frame', 'total_detections', 'avg_confidence', 'weighted_confidence']])
#
#     return df_combined
#
# def plot_weighted_confidence(df_combined):
#     plt.figure(figsize=(10, 6))
#     plt.plot(df_combined['frame'], df_combined['weighted_confidence'], label='Weighted Confidence Score', marker='o', color='blue')
#     plt.xlabel('Frame Number')
#     plt.ylabel('Weighted Confidence Score')
#     plt.title('Weighted Confidence Score per Frame')
#     plt.legend()
#     plt.grid(True)
#     plt.savefig('output_plots/weighted_confidence_per_frame.png')
#     plt.show()

def calculate_weighted_confidence(df_original, df_corrupted):
    """
    Calculate weighted confidence for each frame based on original detections and corrupted confidence.
    """
    # Group by frame to calculate the total detections per frame in original and corrupted videos
    df_original_grouped = df_original.groupby('frame')['confidence'].count().reset_index(name='original_detections')
    df_corrupted_grouped = df_corrupted.groupby('frame')['confidence'].mean().reset_index(
        name='avg_corrupted_confidence')
    df_corrupted_detections = df_corrupted.groupby('frame')['confidence'].count().reset_index(
        name='corrupted_detections')

    # Merge dataframes to align original detections with corrupted confidence
    merged_df = pd.merge(df_original_grouped, df_corrupted_grouped, on='frame')
    merged_df = pd.merge(merged_df, df_corrupted_detections, on='frame')

    # Calculate weighted confidence for each frame
    merged_df['weighted_confidence'] = (merged_df['avg_corrupted_confidence'] / merged_df['original_detections']) * \
                                       merged_df['corrupted_detections']

    return merged_df

def plot_weighted_confidence(merged_df, output_path_ts='output_plots/ts_weighted_confidence_plot.png'):
    """
    Plot weighted confidence over the frames.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(merged_df['frame'], merged_df['weighted_confidence'], label='Weighted Confidence', marker='o',
             color='blue')
    plt.xlabel('Frame Number')
    plt.ylabel('Weighted Confidence')
    plt.title('YOLO Weighted Confidence Over Frames')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path_ts)
    plt.show()
    print(f"Weighted confidence plot saved as '{output_path_ts}'")

# Main execution
if __name__ == "__main__":

    # Initialize the Roboflow model
    rf = Roboflow(api_key="BxgGN2VV9dHmazRhr3eu")
    project = rf.workspace().project("roboflow-tenyks-fix-your-dataset")
    model = project.version(2).model

    # original_video_path = '/Users/abhinav/Documents/MS CS/Sem 5 - Fall 2024/src/YOLO_Video/data/original_videos/02c091d.mp4'
    # output_video_path = '/Users/abhinav/Documents/MS CS/Sem 5 - Fall 2024/src/YOLO_Video/output/ts_output_original_video_02c091d.mp4'
    # log_file_path = '/Users/abhinav/Documents/MS CS/Sem 5 - Fall 2024/src/YOLO_Video/output_logs/ts_logs_original_02c091d.csv'
    #
    # corrupted_video_path = '/Users/abhinav/Documents/MS CS/Sem 5 - Fall 2024/src/YOLO_Video/data/attacked_videos/02c091d.mp4'
    # output_corrupted_video_path = '/Users/abhinav/Documents/MS CS/Sem 5 - Fall 2024/src/YOLO_Video/output/ts_output_corrupted_video_02c091d.mp4'
    # corrupted_log_file_path = '/Users/abhinav/Documents/MS CS/Sem 5 - Fall 2024/src/YOLO_Video/output_logs/ts_logs_corrupted_02c091d.csv'

    original_video_path = '/Users/abhinav/Documents/MS CS/Sem 5 - Fall 2024/src/YOLO_Video/data/original_videos/0a544a0.mp4'
    output_video_path = '/Users/abhinav/Documents/MS CS/Sem 5 - Fall 2024/src/YOLO_Video/output/ts_output_original_video_0a544a0.mp4'
    log_file_path = '/Users/abhinav/Documents/MS CS/Sem 5 - Fall 2024/src/YOLO_Video/output_logs/ts_logs_original_0a544a0.csv'

    corrupted_video_path = '/Users/abhinav/Documents/MS CS/Sem 5 - Fall 2024/src/YOLO_Video/data/attacked_videos/0a544a0.mp4'
    output_corrupted_video_path = '/Users/abhinav/Documents/MS CS/Sem 5 - Fall 2024/src/YOLO_Video/output/ts_output_corrupted_video_0a544a0.mp4'
    corrupted_log_file_path = '/Users/abhinav/Documents/MS CS/Sem 5 - Fall 2024/src/YOLO_Video/output_logs/ts_logs_corrupted_0a544a0.csv'


    # Process original video
    process_video(original_video_path, output_video_path, log_file_path)

    # Process corrupted video
    process_video(corrupted_video_path, output_corrupted_video_path, corrupted_log_file_path)

    # Read the logged data
    df_original = pd.read_csv(log_file_path)
    df_corrupted = pd.read_csv(corrupted_log_file_path)
    #
    # # Plot comparison
    # plot_confidence_scores(df_original, df_corrupted)

    # Example usage with your dataframes
    df_weighted_conf = calculate_weighted_confidence(df_original, df_corrupted)
    # Save the result to a CSV file if needed
    df_weighted_conf.to_csv('output_logs/ts_weighted_confidence_scores.csv', index=False)

    plot_weighted_confidence(df_weighted_conf)

    # plot_confidence_scores(df_original, df_corrupted)
    plot_detections_histogram(df_original, df_corrupted)
    #
    # calculate_bbox_area(df_original)
    # calculate_bbox_area(df_corrupted)
    #
    # plot_number_of_detections(df_original, df_corrupted)
    plot_confidence_histogram(df_original, df_corrupted)
    # plot_bbox_area_comparison(df_original, df_corrupted)
    plot_class_distribution_comparison(df_original, df_corrupted)

    smooth_and_plot_confidence(df_original, df_corrupted, window_size=10,
                               output_path='output_plots/ts_confidence_scores_comparison.png')
    plot_number_of_detections_per_frame(df_original, df_corrupted,
                                        output_path='output_plots/ts_number_of_detections_per_frame.png')

    print("All plots generated and saved successfully.")
