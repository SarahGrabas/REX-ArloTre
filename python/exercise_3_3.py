import cv2
import cv2.aruco as aruco
import numpy as np
import matplotlib
import picamera2

matplotlib.use("Agg")  # Gem plots, også når programmet køres via SSH uden skærm.
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from datetime import datetime
from pathlib import Path
from time import sleep


# Kamerakalibrering fra opgave 3.1 ved opløsningen 1640 x 1232.
# Tilnærmelse: fx = fy, billedcentrum som hovedpunkt og ingen forvrængning.
focal_length = 1288.9       # I pixels
cx, cy = 1640 / 2, 1232 / 2 # Midten af billedet i pixels
camera_matrix = np.array(
    [[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]], dtype=np.float32,
)
dist_coeffs = np.zeros((5, 1), dtype=np.float32) # Tilnærmelse: Vi antager ingen forvrængning, da kameraet er kalibreret
marker_length = 0.146  # Alle markørers sorte kvadrat skal have sidelængden 14,6 cm

# ArUco-opsætning til OpenCV 4.6.0 (Versionen som ligger på Raspberry Pi'en)
dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
output_dir = Path(__file__).resolve().parent / "landmark_maps"


def build_landmark_map(ids, tvecs):
    """Returner [(ID, (Xc, Zc)), ...] for det aktuelle billede i enheden meter.

    Xc er positiv mod højre, Zc er positiv fremad. Yc udelades.
    Dette er et kort set ovenfra, hvis kameraet sidder vandret og peger fremad.
    """
    landmarks = []
    if ids is None or tvecs is None: # Kan der ikke ses nogle markører i billedet returneres en tom liste
        return landmarks

    for marker_id, tvec in zip(ids.flatten(), tvecs):
        Xc, Yc, Zc = tvec.reshape(3)
        if np.all(np.isfinite([Xc, Yc, Zc])) and Zc > 0:
            landmarks.append((int(marker_id), (float(Xc), float(Zc))))

    return landmarks


def plot_landmarks(landmarks, filename):
    """Gem et 2D-kort med kameraet i (0, 0) og ID ved hver markør."""
    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        ax.scatter(0, 0, marker=MarkerStyle("^"), s=100, color="black", label="Kamera")

        for marker_id, (Xc, Zc) in landmarks:
            ax.scatter(Xc, Zc, color="tab:blue")
            ax.annotate(
                f"ID {marker_id}", (Xc, Zc),
                xytext=(6, 6), textcoords="offset points",
            )

        # Udvid kortet, så også fjerne markører kommer med.
        half_width = max([1.0] + [abs(x) + 0.3 for _, (x, z) in landmarks])
        max_depth = max([1.0] + [z + 0.3 for _, (x, z) in landmarks])
        ax.set_xlim(-half_width, half_width)
        ax.set_ylim(-0.2, max_depth)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Xc: sideafstand [m] (positiv mod højre)")
        ax.set_ylabel("Zc: dybde [m] (positiv fremad)")
        ax.set_title(f"Landmarkkort - {len(landmarks)} synlige markører")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(filename, dpi=150)
    finally:
        plt.close(fig)


def main():
    output_dir.mkdir(parents=True, exist_ok=True)
    cam = picamera2.Picamera2()
    try:
        config = cam.create_video_configuration(
            {"size": (1640, 1232), "format": "RGB888"},
            queue=False,  # Hent et nyt billede efter brugerens Enter.
        )
        cam.configure(config)
        cam.start(show_preview=False)
        sleep(1)

        while input("Enter: tag billede og lav kort. q: afslut").strip().lower() != "q":
            frame = cam.capture_array("main")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = aruco.detectMarkers(gray, dictionary)

            # Nyt kort for hvert billede. Intet filter på target_id.
            tvecs = None
            if ids is not None and len(ids) > 0:
                _, tvecs, _ = aruco.estimatePoseSingleMarkers(
                    corners, marker_length, camera_matrix, dist_coeffs
                )

            landmarks = build_landmark_map(ids, tvecs)
            print("Landmarkkort: [(ID, (Xc, Zc)), ...], koordinater i meter")
            print(landmarks)
            if not landmarks:
                print("Ingen markører med gyldig position fundet i billedet.")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = output_dir / f"landmark_map_{timestamp}.png"
            plot_landmarks(landmarks, filename)
            print(f"Kort gemt: {filename}")

    except (KeyboardInterrupt, EOFError):
        print("\nKortlægning afsluttet.")
    finally:

        cam.close()


if __name__ == "__main__":
    main()
