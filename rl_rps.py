import cv2
import mediapipe as mp
import time
import random
import tkinter as tk
from tkinter import simpledialog, messagebox
import pickle
import os
import numpy as np

# ---------------------- USER INPUT ----------------------
root = tk.Tk()
root.withdraw()

mode = simpledialog.askstring("Mode", "A = Auto Level-Up (beat AI to advance)\nC = Custom Level", initialvalue="A")
if not mode or mode.upper() not in ['A', 'C']:
    root.destroy()
    raise SystemExit
mode = mode.upper()

target_score = simpledialog.askinteger("Target", "Rounds to win (e.g., 5):", minvalue=3, initialvalue=5)
if target_score is None:
    root.destroy()
    raise SystemExit

if mode == 'C':
    level = simpledialog.askinteger("Level", "1=Beginner\n2=Easy Q\n3=Medium\n4=Trained\n5=Adaptive", minvalue=1, maxvalue=5)
    if level is None:
        root.destroy()
        raise SystemExit
else:
    level = 1

# ---------------------- SETUP ----------------------
choices = ['Rock', 'Paper', 'Scissors']
tip_ids = [4, 8, 12, 16, 20]
beats = {'Rock': 'Paper', 'Paper': 'Scissors', 'Scissors': 'Rock'}
q_table_file = 'q_table.pkl'

level_config = {
    1: {'history': 2, 'eps': 0.8,  'name': 'Beginner'},
    2: {'history': 3, 'eps': 0.5,  'name': 'Easy Q'},
    3: {'history': 4, 'eps': 0.3,  'name': 'Medium Q'},
    4: {'history': 5, 'eps': 0.1,  'name': 'Trained Q'},
    5: {'history': 6, 'eps': 0.05, 'name': 'Adaptive'}
}

min_rounds_to_advance = {
    1: 8,
    2: 10,
    3: 12,
    4: 15,
    5: 20
}

Q_table = {}
if os.path.exists(q_table_file):
    try:
        with open(q_table_file, 'rb') as f:
            Q_table = pickle.load(f)
        print(f"📥 Loaded Q-table with {len(Q_table)} states")
    except Exception as e:
        print("Failed to load Q-table:", e)
        Q_table = {}
else:
    print("🆕 Starting with empty Q-table")

def save_q_table():
    try:
        with open(q_table_file, 'wb') as f:
            pickle.dump(Q_table, f)
        print(f"📤 Saved Q-table with {len(Q_table)} states")
    except Exception as e:
        print("Q-table save error:", e)

# ---------------------- ENHANCED REAL-TIME GRAPH ----------------------
def render_mini_graph(win_data, qsize_data, avg_q_data, eps_data, rounds, size=(420, 240)):
    w, h = size
    img = np.zeros((h, w, 3), dtype=np.uint8) + 20
    pad = 32
    usable_w = w - 2 * pad
    usable_h = h - 2 * pad

    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (100, 100, 100), 1)

    # Left Y-axis: Win Rate (0-100%)
    for val in (0, 25, 50, 75, 100):
        y = pad + int((100 - val) * usable_h / 100.0)
        cv2.line(img, (pad, y), (w//2 - 5, y), (40, 40, 40), 1)
        cv2.putText(img, f"{val}%", (4, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1, cv2.LINE_AA)

    # Right Y-axis: Q-Table Size
    max_qsize = max(qsize_data) if qsize_data else 1
    if max_qsize == 0: max_qsize = 1
    step = max(1, int(max_qsize // 4))
    for val in range(0, int(max_qsize) + step, step):
        norm_val = min(1.0, val / max_qsize)
        y = pad + int((1 - norm_val) * usable_h)
        cv2.line(img, (w//2 + 5, y), (w - pad, y), (40, 40, 40), 1)
        cv2.putText(img, f"{val}", (w - pad + 2, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1, cv2.LINE_AA)

    if not rounds:
        return img

    max_x = max(10, rounds[-1])
    xs = [pad + int((r - 1) * usable_w / max(1, max_x - 1)) for r in rounds]

    # Plot Win Rate (green)
    win_pts = []
    for i in range(len(rounds)):
        x = xs[i]
        y = pad + int((100 - win_data[i]) * usable_h / 100.0)
        win_pts.append((x, y))
    if len(win_pts) >= 2:
        cv2.polylines(img, [np.array(win_pts, dtype=np.int32)], False, (0, 220, 0), 2, cv2.LINE_AA)

    # Plot Q-Table Size (orange)
    qsize_pts = []
    for i in range(len(rounds)):
        x = xs[i]
        norm_q = min(1.0, qsize_data[i] / max_qsize)
        y = pad + int((1 - norm_q) * usable_h)
        qsize_pts.append((x, y))
    if len(qsize_pts) >= 2:
        cv2.polylines(img, [np.array(qsize_pts, dtype=np.int32)], False, (255, 165, 0), 2, cv2.LINE_AA)

    # Title
    cv2.putText(img, "RL Learning Progress", (pad, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240,240,240), 1, cv2.LINE_AA)
    
    # Legend
    cv2.rectangle(img, (w - 130, 8), (w - 112, 20), (0, 220, 0), -1)
    cv2.putText(img, "Win%", (w - 106, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240,240,240), 1, cv2.LINE_AA)
    cv2.rectangle(img, (w - 130, 24), (w - 112, 36), (255, 165, 0), -1)
    cv2.putText(img, "Q-Size", (w - 106, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240,240,240), 1, cv2.LINE_AA)

    # Current values
    if win_data:
        cv2.putText(img, f"Win Rate: {win_data[-1]:.1f}%", (pad, h - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 0), 1, cv2.LINE_AA)
    if qsize_data:
        cv2.putText(img, f"Q-States: {int(qsize_data[-1])}", (pad, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 165, 0), 1, cv2.LINE_AA)
    if avg_q_data:
        cv2.putText(img, f"Avg Q: {avg_q_data[-1]:.2f}", (pad, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (100, 200, 255), 1, cv2.LINE_AA)

    return img

def overlay_image_alpha(dest, src, x, y, alpha=0.88):
    h, w = src.shape[:2]
    if y + h > dest.shape[0] or x + w > dest.shape[1]:
        return
    overlay_area = dest[y:y+h, x:x+w]
    blended = cv2.addWeighted(overlay_area, 1.0 - alpha, src, alpha, 0)
    dest[y:y+h, x:x+w] = blended

# ---------------------- MEDIAPIPE ----------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
if not cap.isOpened():
    messagebox.showerror("Camera Error", "Cannot open webcam.")
    root.destroy()
    raise SystemExit

# ---------------------- GAME STATE ----------------------
user_score = ai_score = tie_score = 0
total_rounds = 0
game_history = []
gesture_buffer = []
buffer_size = 3
waiting_for_cooldown = False
cooldown_start = None
current_user_gesture = "None"
ai_play = "None"
round_accepted = False
game_over = False

win_data, qsize_data, avg_q_data, eps_data, round_list = [], [], [], [], []

# ---------------------- TRUE RL HELPERS ----------------------
learning_rate = 0.7
discount = 0.9

def get_state(history, h_len):
    if len(history) < h_len:
        return tuple([('Start', 'Start')] * (h_len - len(history)) + history)
    return tuple(history[-h_len:])

def choose_ai_action(state, eps):
    if state not in Q_table:
        Q_table[state] = [0.0, 0.0, 0.0]
    if random.random() < eps:
        return random.choice(choices)
    return choices[np.argmax(Q_table[state])]

def get_reward(ai_move, user_move):
    if beats[user_move] == ai_move:
        return 1.0
    elif beats[ai_move] == user_move:
        return -1.0
    else:
        return 0.0

# ---------------------- ROBUST GESTURE CLASSIFIER ----------------------
def classify_hand(lms):
    fingers = []
    for i in range(1, 5):
        tip = lms[tip_ids[i]]
        pip = lms[tip_ids[i] - 2]
        fingers.append(1 if tip.y < pip.y - 0.02 else 0)
    
    total_up = sum(fingers)
    
    if total_up == 0:
        return 'Rock'
    elif total_up == 2 and fingers[0] == 1 and fingers[1] == 1:
        return 'Scissors'
    elif total_up >= 3:
        return 'Paper'
    else:
        return None

# ---------------------- MAIN LOOP ----------------------
win_title = "RL Rock Paper Scissors — Real Learning in Action!"
cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)

try:
    while True:
        success, frame = cap.read()
        if not success:
            print("Camera frame fail; exiting.")
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        current_gesture = None
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                if not waiting_for_cooldown:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, handLms, mp_hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_utils.DrawingSpec(color=(0,255,0), thickness=2),
                        mp.solutions.drawing_utils.DrawingSpec(color=(255,0,0), thickness=2)
                    )
                current_gesture = classify_hand(handLms.landmark)

        if current_gesture:
            gesture_buffer.append(current_gesture)
            if len(gesture_buffer) > buffer_size:
                gesture_buffer.pop(0)
            if len(gesture_buffer) == buffer_size and len(set(gesture_buffer)) == 1:
                current_user_gesture = current_gesture
            else:
                current_user_gesture = "..."
        else:
            gesture_buffer.clear()
            current_user_gesture = "None"

        if (not waiting_for_cooldown) and (current_user_gesture in choices) and (not round_accepted) and (not game_over):
            user_move = current_user_gesture
            h_len = level_config[level]['history']
            eps = level_config[level]['eps']

            state = get_state(game_history, h_len)
            ai_play = choose_ai_action(state, eps)

            winner = 'Tie'
            if beats[user_move] == ai_play:
                winner = 'AI'
            elif beats[ai_play] == user_move:
                winner = 'User'

            next_history = game_history + [(user_move, ai_play)]
            next_state = get_state(next_history, h_len)

            reward = get_reward(ai_play, user_move)
            if state not in Q_table:
                Q_table[state] = [0.0, 0.0, 0.0]
            if next_state not in Q_table:
                Q_table[next_state] = [0.0, 0.0, 0.0]

            ai_idx = choices.index(ai_play)
            current_q = Q_table[state][ai_idx]
            max_next_q = max(Q_table[next_state])
            new_q = current_q + learning_rate * (reward + discount * max_next_q - current_q)
            Q_table[state][ai_idx] = new_q

            if winner == 'AI':
                ai_score += 1
            elif winner == 'User':
                user_score += 1
            else:
                tie_score += 1

            total_rounds += 1
            game_history = next_history

            win_rate = (ai_score / total_rounds) * 100 if total_rounds > 0 else 0
            q_size = len(Q_table)
            avg_q = np.mean([np.max(q_vals) for q_vals in Q_table.values()]) if Q_table else 0.0

            win_data.append(win_rate)
            qsize_data.append(q_size)
            avg_q_data.append(avg_q)
            eps_data.append(eps)
            round_list.append(total_rounds)

            if total_rounds % 10 == 0:
                print(f"Round {total_rounds} | Win Rate: {win_rate:.1f}% | Q-States: {q_size} | Avg Q: {avg_q:.2f}")

            if mode == 'A':
                if winner == 'User' and user_score >= target_score:
                    required_rounds = min_rounds_to_advance.get(level, 8)
                    if total_rounds >= required_rounds:
                        if level < 5:
                            messagebox.showinfo("🏆 Level Complete!", 
                                f"You beat Level {level}!\n"
                                f"AI learned from {total_rounds} rounds.\n"
                                f"Advancing to Level {level + 1}!")
                            level += 1
                            user_score = ai_score = tie_score = 0
                            total_rounds = 0
                        else:
                            messagebox.showinfo("🎉 Champion!", "You beat the hardest AI! You're the RPS Master!")
                            game_over = True
                    else:
                        remaining = required_rounds - total_rounds
                        messagebox.showinfo("⏳ Keep Playing!", 
                            f"AI needs more data to learn!\n"
                            f"Play {remaining} more round(s) (total: {total_rounds}/{required_rounds})")
                elif winner == 'AI' and ai_score >= target_score:
                    messagebox.showinfo("💀 Game Over", "AI won! You failed to beat the current level.")
                    game_over = True
            else:
                if user_score >= target_score or ai_score >= target_score:
                    game_over = True

            waiting_for_cooldown = True
            cooldown_start = time.time()
            round_accepted = True

        elif waiting_for_cooldown:
            if time.time() - cooldown_start >= 2.0:
                waiting_for_cooldown = False
                round_accepted = False
                gesture_buffer.clear()

        # ---------------------- DISPLAY ----------------------
        status = "✓ Ready" if (not waiting_for_cooldown and current_user_gesture in choices) else ""
        cv2.putText(frame, f"MODE: {'AUTO' if mode=='A' else 'CUSTOM'} | LEVEL {level}: {level_config[level]['name']}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"You: {current_user_gesture} {status}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)

        if waiting_for_cooldown:
            cv2.putText(frame, f"AI CHOSE: {ai_play}", (10, 96),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 180, 0), 2, cv2.LINE_AA)

        cv2.putText(frame, f"Score — You: {user_score} | AI: {ai_score} | Tie: {tie_score}",
                    (10, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

        if total_rounds > 0:
            win_rate = (ai_score / total_rounds) * 100
            cv2.putText(frame, f"AI Win Rate: {win_rate:.1f}%", (10, 168),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2, cv2.LINE_AA)
            if mode == 'A':
                req = min_rounds_to_advance.get(level, 8)
                cv2.putText(frame, f"Rounds: {total_rounds}/{req}", (10, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 100), 1, cv2.LINE_AA)

        if current_user_gesture == "None":
            cv2.putText(frame, "❓ No Hand Detected", (w - 240, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 255), 2, cv2.LINE_AA)
        elif current_user_gesture == "...":
            cv2.putText(frame, "⏳ Stabilizing Gesture...", (w - 260, h - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2, cv2.LINE_AA)

        if game_over:
            if mode == 'A':
                msg = "🎉 YOU BEAT LEVEL 5!" if user_score >= target_score else "😈 AI WON THE LEVEL!"
            else:
                msg = "🎉 YOU WIN!" if user_score >= target_score else "😈 AI WINS!"
            cv2.putText(frame, msg, (w // 2 - 220, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                        (0, 255, 255) if 'YOU' in msg else (0, 100, 255), 3, cv2.LINE_AA)
            cv2.putText(frame, "Press ESC to exit", (w // 2 - 120, h // 2 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (200, 200, 200), 2, cv2.LINE_AA)

        graph_img = render_mini_graph(
            win_data if win_data else [0],
            qsize_data if qsize_data else [0],
            avg_q_data if avg_q_data else [0],
            eps_data if eps_data else [1.0],
            round_list if round_list else [0],
            size=(420, 240)
        )
        gx = w - graph_img.shape[1] - 10
        gy = 10
        overlay_image_alpha(frame, graph_img, gx, gy, alpha=0.88)

        cv2.imshow(win_title, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

finally:
    save_q_table()
    cap.release()
    cv2.destroyAllWindows()
    try:
        root.destroy()
    except:
        pass

    print("✅ Game closed cleanly.")
