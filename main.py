from utils import read_video, save_video
from trackers import Tracker
def main():
    #Reading Video
    video_frames = read_video("/Users/sarthaktyagi/Desktop/internshipProject/15sec_input_720p.mp4")

    #Initialize Tracker
    tracker = Tracker("/Users/sarthaktyagi/Desktop/internshipProject/best.pt")
    tracks = tracker.get_object_tracks(video_frames, read_from_stub=False, stub_path='tracker_stubs/player_detection.pkl')
    #Draw Object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    #Save
    save_video(output_video_frames, 'output_videos/output.mp4')


if __name__ == "__main__":
    main()
