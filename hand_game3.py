import cv2
import mediapipe as mp
import pyautogui

# 初始化 MediaPipe 手部模型
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# 打开摄像头
capture = cv2.VideoCapture(0)

while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        break

    # 翻转图像，以便镜像显示
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    # 将图像从 BGR 转换为 RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 使用 MediaPipe 进行手部检测
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # 绘制手部关键点
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 获取手腕位置（landmark 0）、中指根部位置（landmark 9）和食指指尖（landmark 8）
            wrist = hand_landmarks.landmark[0]
            middle_base = hand_landmarks.landmark[9]
            finger_tip = hand_landmarks.landmark[8]  # 食指指尖

            wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
            middle_x, middle_y = int(middle_base.x * w), int(middle_base.y * h)
            finger_x, finger_y = int(finger_tip.x * w), int(finger_tip.y * h)

            # 检测手指向上：如果食指的 y 坐标小于手腕 y 坐标，表示手指向上
            if finger_y < wrist_y - 30:  # 阈值根据需要调整
                pyautogui.press('up')
                cv2.putText(frame, "UP", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # 检测手指向下：如果食指的 y 坐标大于手腕 y 坐标，表示手指向下
            elif finger_y > wrist_y + 30:  # 阈值根据需要调整
                pyautogui.press('down')
                cv2.putText(frame, "DOWN", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            # 检测手掌顺时针旋转：如果中指基部的x坐标大于手腕x坐标，则认为是顺时针旋转
            if middle_x > wrist_x + 20:  # 阈值根据需要调整
                pyautogui.press('left')
                cv2.putText(frame, "LEFT (CW Rotation)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # 检测手掌逆时针旋转：如果中指基部的x坐标小于手腕x坐标，则认为是逆时针旋转
            elif middle_x < wrist_x - 20:  # 阈值根据需要调整
                pyautogui.press('right')
                cv2.putText(frame, "RIGHT (CCW Rotation)", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 显示结果
    cv2.imshow("Hand Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头和销毁窗口
capture.release()
cv2.destroyAllWindows()
