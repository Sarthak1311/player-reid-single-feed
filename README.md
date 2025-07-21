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

