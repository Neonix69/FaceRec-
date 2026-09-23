import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis


# ==============================
# CONFIGURATION
# ==============================

KNOWN_FACES_DIR = "known_faces"
TEST_IMAGES_DIR = "test_images"
RESULTS_DIR = "results"

# Lower = stricter matching
MATCH_THRESHOLD = 0.45


# ==============================
# INITIALIZE FACE MODEL
# ==============================

print("Loading face recognition model...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)

print("Model loaded successfully!")


# ==============================
# CREATE FOLDERS
# ==============================

os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(TEST_IMAGES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ==============================
# NORMALIZE EMBEDDING
# ==============================

def normalize_embedding(embedding):
    """
    Normalize a face embedding so that
    cosine similarity can be calculated.
    """

    norm = np.linalg.norm(embedding)

    if norm == 0:
        return embedding

    return embedding / norm


# ==============================
# LOAD KNOWN FACES
# ==============================

def load_known_faces():

    known_faces = {}

    print("\nLoading known faces...")

    for filename in os.listdir(KNOWN_FACES_DIR):

        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        ):
            continue

        image_path = os.path.join(
            KNOWN_FACES_DIR,
            filename
        )

        image = cv2.imread(image_path)

        if image is None:
            print(f"Could not read: {filename}")
            continue

        faces = app.get(image)

        if len(faces) == 0:
            print(f"No face found in: {filename}")
            continue

        if len(faces) > 1:
            print(
                f"Warning: multiple faces found in "
                f"{filename}. Using the largest face."
            )

        # Choose the largest face
        face = max(
            faces,
            key=lambda x:
            (x.bbox[2] - x.bbox[0])
            * (x.bbox[3] - x.bbox[1])
        )

        embedding = normalize_embedding(
            face.embedding
        )

        # Filename becomes person's name
        name = os.path.splitext(filename)[0]

        known_faces[name] = embedding

        print(f"Loaded: {name}")

    print(
        f"\nTotal known people: {len(known_faces)}"
    )

    return known_faces


# ==============================
# CALCULATE SIMILARITY
# ==============================

def cosine_similarity(embedding1, embedding2):

    return np.dot(
        embedding1,
        embedding2
    )


# ==============================
# RECOGNIZE A FACE
# ==============================

def recognize_face(face, known_faces):

    if not known_faces:
        return "Unknown", 0.0

    current_embedding = normalize_embedding(
        face.embedding
    )

    best_name = "Unknown"
    best_similarity = -1

    for name, known_embedding in known_faces.items():

        similarity = cosine_similarity(
            current_embedding,
            known_embedding
        )

        if similarity > best_similarity:

            best_similarity = similarity
            best_name = name

    if best_similarity >= MATCH_THRESHOLD:
        return best_name, best_similarity

    return "Unknown", best_similarity


# ==============================
# PROCESS IMAGE
# ==============================

def process_image(
    image_path,
    known_faces
):

    print(
        f"\nProcessing: "
        f"{os.path.basename(image_path)}"
    )

    image = cv2.imread(image_path)

    if image is None:
        print("Could not read image.")
        return

    faces = app.get(image)

    print(f"Faces detected: {len(faces)}")

    for face in faces:

        # Face bounding box
        x1, y1, x2, y2 = face.bbox.astype(int)

        # Recognize
        name, similarity = recognize_face(
            face,
            known_faces
        )

        # Convert similarity into a display score
        score = max(0, similarity) * 100

        print(
            f"  {name} "
            f"(similarity: {score:.1f}%)"
        )

        # Bounding box
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        # Label
        label = f"{name} {score:.1f}%"

        # Text background
        text_y = max(y1 - 10, 25)

        (text_width, text_height), _ = (
            cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2
            )
        )

        cv2.rectangle(
            image,
            (x1, text_y - text_height - 8),
            (x1 + text_width + 8, text_y + 4),
            (0, 255, 0),
            -1
        )

        # Text
        cv2.putText(
            image,
            label,
            (x1 + 4, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )

    # Save result
    filename = os.path.basename(image_path)

    output_path = os.path.join(
        RESULTS_DIR,
        f"result_{filename}"
    )

    cv2.imwrite(
        output_path,
        image
    )

    print(
        f"Saved result: {output_path}"
    )


# ==============================
# MAIN PROGRAM
# ==============================

def main():

    print("=" * 50)
    print("FACIAL RECOGNITION SYSTEM")
    print("=" * 50)

    # Load reference faces
    known_faces = load_known_faces()

    if not known_faces:

        print(
            "\nNo known faces found."
        )

        print(
            "Add face images to:"
        )

        print(
            f"  {KNOWN_FACES_DIR}"
        )

        return

    # Find test images
    test_images = [
        filename
        for filename in os.listdir(
            TEST_IMAGES_DIR
        )
        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".webp")
        )
    ]

    if not test_images:

        print(
            "\nNo test images found."
        )

        print(
            f"Add images to: {TEST_IMAGES_DIR}"
        )

        return

    # Process every image
    for filename in test_images:

        image_path = os.path.join(
            TEST_IMAGES_DIR,
            filename
        )

        process_image(
            image_path,
            known_faces
        )

    print("\n" + "=" * 50)
    print("PROCESSING COMPLETE")
    print("=" * 50)


# ==============================
# START PROGRAM
# ==============================

if __name__ == "__main__":
    main()
