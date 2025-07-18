from feature_extractor import FeatureExtractor
import torch

class PlayerTrackerCNN:
    def __init__(self, similarity_threshold=0.7, max_disappeared=50):
        self.players = {}  # id: {embedding, bbox, cls, disappeared}
        self.similarity_threshold = similarity_threshold
        self.max_disappeared = max_disappeared
        self.extractor = FeatureExtractor()
        self.next_id = 0

    def match_players(self, frame, detections):
        matched_ids = []
        unmatched_ids = list(self.players.keys())

        new_embeddings = []
        for cls, box in detections:
            x1, y1, x2, y2 = map(int, box)
            cropped = frame[y1:y2, x1:x2]
            embedding = self.extractor.extract(cropped)
            new_embeddings.append((cls, box, embedding))

        for cls, box, emb in new_embeddings:
            best_match = None
            best_score = self.similarity_threshold

            for pid in unmatched_ids:
                prev_emb = self.players[pid]['embedding']
                score = torch.nn.functional.cosine_similarity(emb, prev_emb, dim=0).item()
                if score > best_score:
                    best_match = pid
                    best_score = score

            if best_match is not None:
                # Update matched player
                self.players[best_match]['embedding'] = emb
                self.players[best_match]['bbox'] = box
                self.players[best_match]['cls'] = cls
                self.players[best_match]['disappeared'] = 0
                matched_ids.append((best_match, (cls, box)))
                unmatched_ids.remove(best_match)
            else:
                # New player
                new_id = str(self.next_id)
                self.next_id += 1
                self.players[new_id] = {
                    "embedding": emb,
                    "bbox": box,
                    "cls": cls,
                    "disappeared": 0
                }
                matched_ids.append((new_id, (cls, box)))

        # Handle disappeared players
        for pid in unmatched_ids:
            self.players[pid]['disappeared'] += 1
            if self.players[pid]['disappeared'] > self.max_disappeared:
                del self.players[pid]

        return matched_ids
