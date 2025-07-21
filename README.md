#re identification in a single feed 

This project performs football match analysis by detecting and tracking players, referees, and the ball from a video input.

---

##  Features

- Detects players, referees, and ball using a custom-trained  model.
- Tracks objects across frames using ByteTrack.
- Draws:
  -  Ellipses around players
  -  Ellipses around referees
  -  Triangles above the ball
- Maintains object identities across frames (Player IDs, etc.).
- Saves the annotated output video.

Clone the Repository
git clone https://github.com/Sarthak1311/player-reid-single-feed.git
cd player-reid-single-feed
2. Create and Activate a Virtual Environment (Recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install Dependencies
pip install -r requirements.txt
If requirements.txt is not available, install manually:
pip install ultralytics supervision opencv-python numpy
4, add the model (best.pt)
5. Run the Project
----> (run this command) python main.py
