import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches


# Define thresholds for detection ratio and confidence ratio
DETECTION_RATIO_LOWER_THRESHOLD = 0.99
DETECTION_RATIO_UPPER_THRESHOLD = 1.01
CONFIDENCE_RATIO_LOWER_THRESHOLD = 0.99
CONFIDENCE_RATIO_UPPER_THRESHOLD = 1.01
WEIGHTED_CONFIDENCE_THRESHOLD_MULTIPLIER = 1.5  # Assume frames with confidence >= 1.5 * mean are non-attacked



def process_video(input_video_path, output_video_path, log_file):
    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Total frames in the video: {total_frames}")  # Verify total frames

    # Define the codec and create a VideoWriter object to save output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    results_log = []
    frame_num = 0

    # Process each frame
    while frame_num < total_frames:
        ret, frame = cap.read()

        if not ret:
            print(f"Error reading frame {frame_num}, skipping.")  # Handle corrupted frame
            frame_num += 1
            continue

        print(f"Processing frame {frame_num} of {total_frames}")

        # Run YOLO inference on the frame
        results = model(frame)
        annotated_frame = results[0].plot()
        out.write(annotated_frame)

        # Log detection results
        for box in results[0].boxes:
            conf = box.conf.item()
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            class_id = box.cls.item()
            label = model.names[int(class_id)]

            results_log.append({
                "frame": frame_num,
                "class": label,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })

        frame_num += 1

    cap.release()
    out.release()

    # Save detection logs
    df = pd.DataFrame(results_log)
    df.to_csv(log_file, index=False)
    print(f"Processing complete. Log saved to {log_file}")




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
    plt.savefig('output_plots/detections_histogram.png')
    print("Detections histogram saved as 'output_plots/detections_histogram.png'")






def plot_confidence_histogram(df_original, df_corrupted):
    plt.figure(figsize=(10, 6))
    plt.hist(df_original['confidence'], bins=30, alpha=0.5, label='Original Video', color='green')
    plt.hist(df_corrupted['confidence'], bins=30, alpha=0.5, label='Corrupted Video', color='red')
    plt.xlabel('Confidence Score')
    plt.ylabel('Frequency')
    plt.title('Histogram of Confidence Scores')
    plt.savefig('output_plots/confidence_histogram.png')
    print("Confidence score histogram saved as 'output_plots/confidence_histogram.png'")



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
    plt.savefig('output_plots/class_distribution_comparison.png')
    print("Class distribution comparison plot saved as 'output_plots/class_distribution_comparison.png'")


def smooth_and_plot_confidence(df_original, df_corrupted, window_size=10,
                               output_path='output_plots/confidence_scores_comparison.png'):
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
                                        output_path='output_plots/number_of_detections_per_frame.png'):
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





# Function to plot weighted confidence with horizontal regions
def plot_weighted_confidence(merged_df, output_path='output_plots/weighted_confidence_plot.png'):
    plt.figure(figsize=(10, 6))

    threshold = merged_df['weighted_confidence'].mean() * WEIGHTED_CONFIDENCE_THRESHOLD_MULTIPLIER

    # Plot the weighted confidence with horizontal shading for attacked regions
    plt.plot(merged_df['frame'], merged_df['weighted_confidence'], label='Weighted Confidence', marker='o',
             color='blue')
    plt.axhline(y=threshold, color='gray', linestyle='--')

    # Shade regions horizontally based on threshold
    for i in range(len(merged_df) - 1):
        start_frame = merged_df['frame'].iloc[i]
        end_frame = merged_df['frame'].iloc[i + 1]
        if merged_df['weighted_confidence'].iloc[i] < threshold:
            plt.axvspan(start_frame, end_frame, color='lightcoral', alpha=0.3)  # Attacked Region
        else:
            plt.axvspan(start_frame, end_frame, color='lightgreen', alpha=0.3)  # Non-Attacked Region

    # Custom legend handles
    attacked_patch = mpatches.Patch(color='lightcoral', label='Attacked Region')
    non_attacked_patch = mpatches.Patch(color='lightgreen', label='Non-Attacked Region')
    plt.legend(handles=[plt.Line2D([0], [0], color='blue', label='Weighted Confidence'),
                        attacked_patch, non_attacked_patch])

    plt.xlabel('Frame Number')
    plt.ylabel('Weighted Confidence')
    plt.title('YOLO Weighted Confidence Over Frames')
    plt.grid(True)
    plt.savefig(output_path)
    plt.show()


# Function to plot detection ratio with horizontal regions for both lower and upper thresholds
def plot_detection_ratio(detection_rates, output_path='output_plots/detection_rate_ratio.png'):
    plt.figure(figsize=(10, 6))

    # Plot the detection rate ratio
    plt.plot(detection_rates['frame'], detection_rates['detection_ratio'],
             label='Detection Rate Ratio (Corrupted/Original)', marker='o', linestyle='--', color='blue')
    plt.axhline(y=DETECTION_RATIO_LOWER_THRESHOLD, color='gray', linestyle='--')
    plt.axhline(y=DETECTION_RATIO_UPPER_THRESHOLD, color='gray', linestyle='--')

    # Shade attacked regions outside the thresholds
    for i in range(len(detection_rates) - 1):
        start_frame = detection_rates['frame'].iloc[i]
        end_frame = detection_rates['frame'].iloc[i + 1]
        ratio_value = detection_rates['detection_ratio'].iloc[i]

        if ratio_value < DETECTION_RATIO_LOWER_THRESHOLD or ratio_value > DETECTION_RATIO_UPPER_THRESHOLD:
            plt.axvspan(start_frame, end_frame, color='lightcoral', alpha=0.3)  # Attacked Region
        else:
            plt.axvspan(start_frame, end_frame, color='lightgreen', alpha=0.3)  # Non-Attacked Region

    # Custom legend handles
    attacked_patch = mpatches.Patch(color='lightcoral', label='Attacked Region')
    non_attacked_patch = mpatches.Patch(color='lightgreen', label='Non-Attacked Region')
    plt.legend(handles=[plt.Line2D([0], [0], color='blue', linestyle='--', label='Detection Rate Ratio'),
                        attacked_patch, non_attacked_patch])

    plt.xlabel('Frame Number')
    plt.ylabel('Detection Rate Ratio')
    plt.title('Detection Rate Ratio (Corrupted/Original) per Frame')
    plt.grid(True)
    plt.savefig(output_path)
    plt.show()


# Function to plot confidence score ratio with horizontal regions for both lower and upper thresholds
def smooth_and_plot_confidence_with_ratio(df_original, df_corrupted, window_size=10,
                                          output_path='output_plots/confidence_scores_ratio_comparison.png'):
    def smooth_data(data, window_size):
        return data.rolling(window=window_size, min_periods=1).mean()

    # Group by frame and calculate mean confidence for each frame
    df_original_grouped = df_original.groupby('frame')['confidence'].mean().reset_index()
    df_corrupted_grouped = df_corrupted.groupby('frame')['confidence'].mean().reset_index()

    # Smooth the confidence scores
    df_original_grouped['smoothed_conf'] = smooth_data(df_original_grouped['confidence'], window_size)
    df_corrupted_grouped['smoothed_conf'] = smooth_data(df_corrupted_grouped['confidence'], window_size)

    # Merge on frame, keeping only frames present in both datasets (inner join)
    merged_df = pd.merge(df_original_grouped[['frame', 'smoothed_conf']],
                         df_corrupted_grouped[['frame', 'smoothed_conf']],
                         on='frame', suffixes=('_original', '_corrupted'))

    # Calculate confidence score ratio
    merged_df['confidence_ratio'] = merged_df['smoothed_conf_corrupted'] / merged_df['smoothed_conf_original']
    merged_df['confidence_ratio'].replace([float('inf'), -float('inf')], pd.NA, inplace=True)
    merged_df['confidence_ratio'].fillna(0, inplace=True)

    plt.figure(figsize=(10, 6))

    # Plot the confidence score ratio
    plt.plot(merged_df['frame'], merged_df['confidence_ratio'], label='Confidence Score Ratio (Corrupted/Original)',
             linestyle='--', color='blue')
    plt.axhline(y=CONFIDENCE_RATIO_LOWER_THRESHOLD, color='gray', linestyle='--')
    plt.axhline(y=CONFIDENCE_RATIO_UPPER_THRESHOLD, color='gray', linestyle='--')

    # Shade attacked regions outside the thresholds
    for i in range(len(merged_df) - 1):
        start_frame = merged_df['frame'].iloc[i]
        end_frame = merged_df['frame'].iloc[i + 1]
        ratio_value = merged_df['confidence_ratio'].iloc[i]

        if ratio_value < CONFIDENCE_RATIO_LOWER_THRESHOLD or ratio_value > CONFIDENCE_RATIO_UPPER_THRESHOLD:
            plt.axvspan(start_frame, end_frame, color='lightcoral', alpha=0.3)  # Attacked Region
        else:
            plt.axvspan(start_frame, end_frame, color='lightgreen', alpha=0.3)  # Non-Attacked Region

    # Custom legend handles
    attacked_patch = mpatches.Patch(color='lightcoral', label='Attacked Region')
    non_attacked_patch = mpatches.Patch(color='lightgreen', label='Non-Attacked Region')
    plt.legend(handles=[plt.Line2D([0], [0], color='blue', linestyle='--', label='Confidence Score Ratio'),
                        attacked_patch, non_attacked_patch])

    plt.xlabel('Frame Number')
    plt.ylabel('Confidence Score Ratio')
    plt.title('YOLO Confidence Score Ratio (Smoothed)')
    plt.grid(True)
    plt.savefig(output_path)
    plt.show()


if __name__ == "__main__":
    model = YOLO('yolov8n.pt')
    print("Model Loaded Successfuly")

    # original_video_path = 'data/6.mp4'
    # output_video_path = 'output/output_original_video_6.mp4'
    # log_file_path = 'output_logs/logs_original_6.csv'
    #
    # corrupted_video_path = 'data/6_attacked.mp4'
    # output_corrupted_video_path = 'output/output_corrupted_video_6.mp4'
    # corrupted_log_file_path = 'output_logs/logs_corrupted_6.csv'

    # EDIT THESE PATHS
    # original_video_path = 'data/02abbfa.mp4'
    # output_video_path = 'output/output_original_video_02abbfa.mp4'
    # log_file_path = 'output_logs/logs_original_02ab bfa.csv'
    #
    # corrupted_video_path = 'data/02abbfa_attacked.mp4'
    # output_corrupted_video_path = 'output/output_corrupted_video_02abbfa.mp4'
    # corrupted_log_file_path = 'output_logs/logs_corrupted_02abbfa.csv'

    # Path to original video
    # original_video_path = '/Users/abhinav/Documents/MS CS/Sem 5 - Fall 2024/src/YOLO_Video/data/6.mp4'
    # output_video_path = '/Users/abhinav/Documents/MS CS/Sem 5 - Fall 2024/src/YOLO_Video/output/output_original_video_6.mp4'
    # log_file_path = '/Users/abhinav/Documents/MS CS/Sem 5 - Fall 2024/src/YOLO_Video/output_logs/logs_original_6.csv'
    #
    # corrupted_video_path = '/Users/abhinav/Documents/MS CS/Sem 5 - Fall 2024/src/YOLO_Video/data/6_attacked.mp4'
    # output_corrupted_video_path = '/Users/abhinav/Documents/MS CS/Sem 5 - Fall 2024/src/YOLO_Video/output/output_corrupted_video_6.mp4'
    # corrupted_log_file_path = '/Users/abhinav/Documents/MS CS/Sem 5 - Fall 2024/src/YOLO_Video/output_logs/logs_corrupted_6.csv'

    original_video_path = '/Users/abhinav/Documents/MS CS/Sem 5 - Fall 2024/src/YOLO_Video/data/original_videos/02abbfa.mp4'
    output_video_path = '/Users/abhinav/Documents/MS CS/Sem 5 - Fall 2024/src/YOLO_Video/output/output_original_video_02abbfa.mp4'
    log_file_path = '/Users/abhinav/Documents/MS CS/Sem 5 - Fall 2024/src/YOLO_Video/output_logs/logs_original_02abbfa.csv'

    corrupted_video_path = '/Users/abhinav/Documents/MS CS/Sem 5 - Fall 2024/src/YOLO_Video/data/attacked_videos/02abbfa.mp4'
    output_corrupted_video_path = '/Users/abhinav/Documents/MS CS/Sem 5 - Fall 2024/src/YOLO_Video/output/output_corrupted_video_02abbfa.mp4'
    corrupted_log_file_path = '/Users/abhinav/Documents/MS CS/Sem 5 - Fall 2024/src/YOLO_Video/output_logs/logs_corrupted_02abbfa.csv'

    # original_video_path = '/Users/abhinav/Documents/MS CS/Sem 5 - Fall 2024/src/YOLO_Video/data/original_videos/02c091d.mp4'
    # output_video_path = '/Users/abhinav/Documents/MS CS/Sem 5 - Fall 2024/src/YOLO_Video/output/output_original_video_02c091d.mp4'
    # log_file_path = '/Users/abhinav/Documents/MS CS/Sem 5 - Fall 2024/src/YOLO_Video/output_logs/logs_original_02c091d.csv'
    #
    # corrupted_video_path = '/Users/abhinav/Documents/MS CS/Sem 5 - Fall 2024/src/YOLO_Video/data/attacked_videos/02c091d.mp4'
    # output_corrupted_video_path = '/Users/abhinav/Documents/MS CS/Sem 5 - Fall 2024/src/YOLO_Video/output/output_corrupted_video_02c091d.mp4'
    # corrupted_log_file_path = '/Users/abhinav/Documents/MS CS/Sem 5 - Fall 2024/src/YOLO_Video/output_logs/logs_corrupted_02c091d.csv'

    cap = cv2.VideoCapture(original_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in the video: {total_frames}")

    cap = cv2.VideoCapture(corrupted_video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total frames in the video: {total_frames}")



    # Process original video
    process_video(original_video_path, output_video_path, log_file_path)
    print("Video Processing Complete for Original Video")

    process_video(corrupted_video_path, output_corrupted_video_path, corrupted_log_file_path)
    print("Video Processing Complete for Corrupted Video")

    df_original = pd.read_csv(log_file_path)
    df_corrupted = pd.read_csv(corrupted_log_file_path)

    # Example usage with your dataframes
    df_weighted_conf = calculate_weighted_confidence(df_original, df_corrupted)
    # Save the result to a CSV file if needed
    df_weighted_conf.to_csv('output_logs/weighted_confidence_scores.csv', index=False)

    plot_weighted_confidence(df_weighted_conf)
    plot_detections_histogram(df_original, df_corrupted)
    plot_confidence_histogram(df_original, df_corrupted)

    smooth_and_plot_confidence(df_original, df_corrupted, window_size=10,
                               output_path='output_plots/confidence_scores_comparison.png')
    smooth_and_plot_confidence_with_ratio(df_original, df_corrupted, window_size=10)




    plot_number_of_detections_per_frame(df_original, df_corrupted,
                                        output_path='output_plots/number_of_detections_per_frame.png')

    # Calculate detection rates per frame for original and corrupted videos
    all_frames = pd.DataFrame({'frame': range(max(df_original['frame'].max(), df_corrupted['frame'].max()) + 1)})
    df_original_detections = df_original.groupby('frame')['confidence'].count().reset_index(name='original_detections')
    df_corrupted_detections = df_corrupted.groupby('frame')['confidence'].count().reset_index(name='corrupted_detections')

    # Merge with all_frames to ensure all frames are accounted for
    detection_rates = all_frames.merge(df_original_detections, on='frame', how='left').fillna(0)
    detection_rates = detection_rates.merge(df_corrupted_detections, on='frame', how='left').fillna(0)

    # Calculate detection ratio (corrupted / original) and handle division by zero
    detection_rates['detection_ratio'] = detection_rates['corrupted_detections'] / detection_rates[
        'original_detections']
    detection_rates['detection_ratio'] = detection_rates['detection_ratio'].replace([float('inf'), -float('inf')], 0)
    detection_rates['detection_ratio'] = detection_rates['detection_ratio'].fillna(0)
    plot_detection_ratio(detection_rates)

    print("All plots generated and saved successfully.")
