import cv2
from fer import FER
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from collections import deque
import threading

# Emotion colors (BGR for opencv)
EMOTION_COLORS = {
    "angry": (0, 0, 255),
    "disgust": (0, 140, 255),
    "fear": (128, 0, 128),
    "happy": (0, 255, 255),
    "sad": (255, 0, 0),
    "surprise": (0, 255, 128),
    "neutral": (200, 200, 200)
}

# Store emotion history for graph
history_len = 50
emotion_history = {e: deque([0]*history_len, maxlen=history_len) 
                   for e in EMOTION_COLORS}
current_emotions = {e: 0.0 for e in EMOTION_COLORS}
lock = threading.Lock()

# Initialize detector
detector = FER(mtcnn=True)
cap = cv2.VideoCapture(0)

# ── Live matplotlib graph in separate thread ──
def run_graph():
    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor('#1e1e1e')
    ax.set_facecolor('#1e1e1e')
    plt.title("Live Emotion Graph", color='white')
    plt.tight_layout()

    def update(frame):
        ax.clear()
        ax.set_facecolor('#1e1e1e')
        ax.set_ylim(0, 1)
        ax.set_title("Live Emotion Graph", color='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#444')

        with lock:
            for emotion, history in emotion_history.items():
                color = [c/255 for c in EMOTION_COLORS[emotion][::-1]]
                ax.plot(list(history), label=emotion, color=color, linewidth=2)

        ax.legend(loc='upper right', fontsize=7,
                  facecolor='#2e2e2e', labelcolor='white')

    ani = animation.FuncAnimation(fig, update, interval=100)
    plt.show()

graph_thread = threading.Thread(target=run_graph, daemon=True)
graph_thread.start()

print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    result = detector.detect_emotions(frame)

    for face in result:
        x, y, w, h = face["box"]
        emotions = face["emotions"]
        dominant = max(emotions, key=emotions.get)
        confidence = emotions[dominant]
        color = EMOTION_COLORS.get(dominant, (255, 255, 255))

        with lock:
            for e, score in emotions.items():
                if e in emotion_history:
                    emotion_history[e].append(score)
                    current_emotions[e] = score

        # Face rectangle with dominant emotion color
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)

        # Dominant emotion label
        label = f"{dominant.upper()} {confidence*100:.0f}%"
        cv2.rectangle(frame, (x, y-35), (x+w, y), color, -1)
        cv2.putText(frame, label, (x+5, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # Mini emotion bars on the side
        bar_x = x + w + 10
        if bar_x + 160 < frame.shape[1]:
            for i, (emotion, score) in enumerate(sorted(emotions.items(),
                                                         key=lambda x: -x[1])):
                bar_y = y + i * 28
                bar_color = EMOTION_COLORS.get(emotion, (200, 200, 200))
                bar_len = int(score * 140)
                cv2.rectangle(frame, (bar_x, bar_y),
                              (bar_x + bar_len, bar_y + 18), bar_color, -1)
                cv2.putText(frame, f"{emotion} {score*100:.0f}%",
                            (bar_x, bar_y + 13),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # Dark overlay header
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 35), (30, 30, 30), -1)
    cv2.putText(frame, "EMOTION DETECTOR  |  Press Q to quit",
                (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)

    cv2.imshow("Emotion Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done!")