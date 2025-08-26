#=== GUI제작 툴 ===
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import messagebox
#=== 스트리밍 모듈 ===
import os, sys, time, cv2, torch, numpy as np, datetime, subprocess, uuid, qrcode
#=== 카메라 모듈 ===
from picamera2 import Picamera2
#=== 임베딩 툴 ===
from keras_facenet import FaceNet as KerasFaceNet
#=== firebase 연동 모듈 ===
import firebase_admin
from firebase_admin import credentials, storage, firestore
#=== 벡터값 비교 모듈 ===
from sklearn.metrics.pairwise import cosine_similarity
#=== I2C 통신 모듈 ===
from smbus2 import SMBus
import threading,time
#=== 시리얼 통신 ===
import serial
import time
import serial.tools.list_ports

# === Firebase SDK 경로 설정 ===
cred_path = "/home/mbp_mk2/test2-61728-firebase-adminsdk-fbsvc-c93ab6c1a3.json"
bucket_name = "test2-61728.firebasestorage.app"
#=== Firebase정보 초기화 ===
if not firebase_admin._apps:
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred, {'storageBucket': bucket_name})
bucket = storage.bucket()
db = firestore.client()

# === YOLO 모델 로드(pytorch 기반) ===
torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
embedder = KerasFaceNet()
sys.path.append('/home/mbp_mk2/venv/yolov5-face')
from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords

device = 'cpu'
#=== YOLO 위짓 불러오기 ===
weights_path = '/home/mbp_mk2/venv/yolov5-face/weights/yolov5n-0.5.pt'
ckpt = torch.load(weights_path, map_location=device, weights_only=False)
model = ckpt['model'].float().eval().fuse()

# === 전역변수 초기화 ===
fail_count = 0
face_detect_count = 0
compared = False
last_yolo_time = 0

# === 시리얼 포트 초기화 ===
def find_esp32_port():
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if "ttyACM" in port.device:   
            return port.device
    return None

# === 시리얼 초기화 ===
port = find_esp32_port()
ser = serial.Serial(port, 115200, timeout=1)

# === 스캔한 얼굴 이미지 업로드 함수 ===
def upload_scan_image(face_image: np.ndarray):
    # === scan_face 폴더의 파일 목록 확인 ===
    blobs = list(bucket.list_blobs(prefix="scan_face/"))
    indices = []
    # === 저장할 파일제목 생성 ===
    for blob in blobs:
        name = os.path.basename(blob.name)
        if name.startswith("test_") and name.endswith(".jpg"):
            try:
                index = int(name.replace("test_", "").replace(".jpg", ""))
                indices.append(index)
            except ValueError:
                continue

    next_index = max(indices) + 1 if indices else 1
    filename = f"scan_face/test_{next_index}.jpg"
    local_path = f"/tmp/test_{next_index}.jpg"
    
    # === 스토리지에 업로드 ===
    cv2.imwrite(local_path, face_image)
    blob = bucket.blob(filename)
    blob.upload_from_filename(local_path)
    print(f"[UPLOAD] 얼굴 이미지 업로드됨 → {filename}")
    
    #=== FaceNet 임베딩 생성 ===
    resized = cv2.resize(face_image, (160, 160))
    normalized = resized.astype("float32") / 255.0
    input_tensor = np.expand_dims(normalized, axis=0)
    embedding = embedder.embeddings(input_tensor)[0].tolist()

    # === 임베딩 결과를 scan 컬렉션에 저장 ===
    doc_id = f"scan_{next_index:02d}"
    doc_ref = db.collection("scan").document(doc_id)
    doc_ref.set({
        "timestamp": datetime.datetime.utcnow(),
        "embedding": embedding,
        "image_path": filename
    })

# === Firestore 인증 로그 기록 함수 ===
def log_time_to_firestore(user_id: str, similarity: float):
    now = datetime.datetime.utcnow()
    log_id = f"log_{now.strftime('%Y%m%d_%H%M%S')}"
    # === logs 컬랙션에 기록 ===
    db.collection("logs").document(log_id).set({
        "matched_user": user_id,
        "similarity": float(round(similarity, 4)),
        "timestamp": now
    })
    print(f"[LOG] Firestore 기록됨: {log_id}")
    

# === 스캔한 벡터를 사용자 벡터와 비교하는 함수 ===
def compare_to_user_vectors(scan_vector: np.ndarray):
    global fail_count
    users_ref = db.collection("users")
    docs = users_ref.stream()
    best_score = -1
    best_user = None

    for doc in docs:
        data = doc.to_dict()
        if "embedding" in data:
            user_vector = np.array(data["embedding"], dtype=np.float32)
            sim = cosine_similarity([scan_vector], [user_vector])[0][0]
            print(f"[COMPARE] {doc.id} 유사도: {sim:.4f}")
            if sim > best_score:
                best_score = sim
                best_user = doc.id

    #=== 일치도 0.8이상일시 통과 ==== 
    if best_score >= 0.8:
        print(f" 유사도 {best_score:.4f} → 인증 성공")
        send_unlock()
        log_time_to_firestore(best_user, best_score)
        fail_count = 0
        
        return True
    else:
        #=== 3연속 실패시 QR인증 시작 ===
        fail_count += 1
        print(f"유사도 {best_score:.4f} → 인증 실패")
        
        if fail_count >= 3:
            return "qrgate"
        return False

# ===== X1202 (배터리 모듈) 기본 설정 =====
I2C_ADDR = 0x36
bus = SMBus(1)  # I2C-1 버스 사용

#=== 베터리 잔량 측정 함수 ===
def read_battery_percent():
    try:
        msb = bus.read_byte_data(I2C_ADDR, 0x04)
        lsb = bus.read_byte_data(I2C_ADDR, 0x05)
        percent = ((msb << 8) | lsb) >> 8
        return percent
    except Exception as e:
        print("배터리 읽기 오류:", e)
        return None
# === 시리얼 출력 함수(unlock) ===
def send_unlock(retries=5):
    for attempt in range(1, retries+1):
        port = find_esp32_port()
        if port is None:
            print(f"[{attempt}] ESP32 포트 없음, 1초 대기 후 재시도")
            time.sleep(1)
            continue

        try:
            print(f"[{attempt}] 포트 열기: {port}")
            with serial.Serial(port, 115200, timeout=1, write_timeout=1) as ser:
                time.sleep(3)  # ESP32 리셋 안정화 대기 (2초→3초로 늘림)
                ser.write(b"Unlock\n")
                ser.flush()
                print("Unlock 전송 완료")
                return True
        except Exception as e:
            print(f"[{attempt}] 오류 발생: {e}, 1초 대기 후 재시도")
            time.sleep(1)

    print("Unlock 전송 실패")
    return False

# === 시리얼 출력 함수(Alarm) ===
def send_alarm(retries=5):
    for attempt in range(1, retries+1):
        port = find_esp32_port()
        if port is None:
            print(f"[{attempt}] ESP32 포트 없음, 1초 대기 후 재시도")
            time.sleep(1)
            continue

        try:
            print(f"[{attempt}] 포트 열기: {port}")
            with serial.Serial(port, 115200, timeout=1, write_timeout=1) as ser:
                time.sleep(3)  # ESP32 리셋 안정화 대기 (2초→3초로 늘림)
                ser.write(b"Alarm\n")
                ser.flush()
                print("Alarm 전송 완료")
                return True
        except Exception as e:
            print(f"[{attempt}] 오류 발생: {e}, 1초 대기 후 재시도")
            time.sleep(1)


    print("Alarm 전송 실패")
    return False


# === GUI 내부 클래스 ===
class FaceAppGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("도어락 얼굴 인식 GUI")
        # === 배터리 표시 라벨 ===
        self.battery_label = tk.Label(self.root, text="배터리: --%", font=("Arial", 16))
        self.battery_label.place(x=0, y=0)
        self.root.geometry("520x600")
        self.battery_label.lift() 
        
        # === 전체화면 표시 ===
        self.root.attributes('-fullscreen', True)
        self.root.bind("<Escape>", lambda event: self.root.attributes("-fullscreen", False)) #esc로 창모드 전환        
        self.registration_mode = False
        self.registration_detect_count = 0

        # === 스트리밍 표시 라벨 ===
        self.picam2 = Picamera2()
        self.picam2.configure(self.picam2.create_preview_configuration(main={"size": (480, 360)}))
        self.picam2.start()        
        self.image_label = tk.Label(self.root)
        self.image_label.place(x=0, y=0)
        
        # === 버튼 표시 라벨 ===
        button_frame = tk.Frame(self.root)
        button_frame.pack(side='right', padx=20, pady=10)


        # === 중간에 여백 생성 ===
        button_frame.grid_columnconfigure(0, weight=0)  # 좌측 열
        button_frame.grid_columnconfigure(1, weight=0)
        button_frame.grid_columnconfigure(2, weight=1)  # 여백 열 (가운데 비우기)
        button_frame.grid_columnconfigure(3, weight=0)
        button_frame.grid_columnconfigure(4, weight=0)

        face_register_img = ImageTk.PhotoImage(Image.open("/home/mbp_mk2/Pictures/icon3.png").resize((168, 230)))
        reset_img = ImageTk.PhotoImage(Image.open("/home/mbp_mk2/Pictures/icon4.png").resize((168, 230)))

        # === 버튼 위치설정 ===
        tk.Button(button_frame, image=face_register_img, command=self.start_qr_process).grid(
            row=0, column=5, padx=5, sticky='e'
            )

        tk.Button(button_frame, image=reset_img, command=self.reset_all_data).grid(
            row=1, column=5, padx=5, sticky='e'
            )
        self.face_register_img = face_register_img
        self.reset_img = reset_img
        
        self.qr_uuid = None
        self.streaming_enabled = True
        
        self.device = device
        self.model = model
        self.embedder = embedder
        self.update_stream()
        threading.Thread(target=self.update_battery_label, daemon=True).start()

    #=== 스트리밍 시작 함수 ===
    def update_stream(self):
        global last_yolo_time, face_detect_count, compared

        if not self.streaming_enabled:
            self.root.after(100, self.update_stream)
            return  

        frame = self.picam2.capture_array()
        #=== 스트리밍 해상도/색상 ===
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)

        frame_disp = frame.copy()
        current_time = time.time()

        #=== 1초 간격으로 객체 탐지 ===
        if current_time - last_yolo_time >= 1.0:
            last_yolo_time = current_time
            img = cv2.resize(frame, (640, 640)).transpose(2, 0, 1)
            img_tensor = torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0).to(device).float() / 255.0

            #=== 감지한 이미지 크기조정 ===
            with torch.no_grad():
                pred = model(img_tensor)[0]
                detections = non_max_suppression(pred, 0.5, 0.4)[0]

            #=== 등록 모드 실행시 동작  ===
            if detections is not None and len(detections):
                if self.registration_mode:
                    detections[:, :4] = scale_coords(img_tensor.shape[2:], detections[:, :4], frame.shape).round()
                    x1, y1, x2, y2 = map(int, detections[0][:4])
                    cv2.rectangle(frame_disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    self.registration_detect_count += 1
                    print(f"[등록 모드] 얼굴 감지됨: {self.registration_detect_count}/3")

                    #=== 3연속 감지시 저장 ===
                    if self.registration_detect_count >= 3:
                        face_crop = frame[y1:y2, x1:x2]

                        # === Storage 저장 (user_face/) ===
                        blobs = list(bucket.list_blobs(prefix="user_face/"))
                        indices = []
                        for blob in blobs:
                            name = os.path.basename(blob.name)
                            if name.startswith("test_") and name.endswith(".jpg"):
                                try:
                                    index = int(name.replace("test_", "").replace(".jpg", ""))
                                    indices.append(index)
                                except ValueError:
                                    continue
                        next_index = max(indices) + 1 if indices else 1
                        filename = f"user_face/test_{next_index}.jpg"
                        local_path = f"/tmp/test_{next_index}.jpg"
                        cv2.imwrite(local_path, face_crop)

                        blob = bucket.blob(filename)
                        blob.upload_from_filename(local_path)
                        image_url = blob.public_url

                        # ===  FaceNet 임베딩 ===
                        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                        face_resized = cv2.resize(face_rgb, (160, 160))
                        embedding = self.embedder.embeddings([face_resized])[0]
                        embedding_list = embedding.tolist()

                        # === users 컬랙션에 벡터값 저장 ===
                        db.collection("users").add({
                            "embedding": embedding_list,
                            "image_url": image_url,
                            "timestamp": datetime.datetime.utcnow()
                        })

                        print("[등록 완료] users 컬렉션 및 user_face 저장 완료")
                        messagebox.showinfo("등록","등록이 완료되었습니다")
                        # === 등록 모드 종료 ===
                        self.registration_mode = False
                        self.registration_detect_count = 0
                    self.root.after(15, self.update_stream)                   
                    return
                
                # === Storage 저장 (scan_face/) ===
                detections[:, :4] = scale_coords(img_tensor.shape[2:], detections[:, :4], frame.shape).round()
                x1, y1, x2, y2 = map(int, detections[0][:4])
                cv2.rectangle(frame_disp, (x1, y1), (x2, y2), (0, 255, 0), 2)
                face_detect_count += 1

                #=== 3연속 감지시 저장 ===
                if face_detect_count >= 3 and not compared:
                    crop = frame[y1:y2, x1:x2]
                    face_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    face_resized = cv2.resize(face_rgb, (160, 160))

                    # ===  FaceNet 임베딩 ===
                    upload_scan_image(crop)
                    embedding = embedder.embeddings([face_resized])[0]
                    # ===  users 벡터와 코사인 비교 ===
                    result = compare_to_user_vectors(embedding)

                    #=== 인증결과 출력 ===
                    if result is True:
                        print("[인증 성공]")
                        send_unlock()
                    elif result == "qrgate":
                        print("[QR 인증 실행]")
                        self.streaming_enabled = False
                        self.root.after(10, self.start_qrgate_process)
                    else:
                        print("[인증 실패]")

                    compared = True
                    face_detect_count = 0
                    self.root.after(2000, self.reset_compare_flag)
            else:
                face_detect_count = 0
                compared = False

        #=== 스트리밍 화면조정 ===
        frame_disp = cv2.cvtColor(frame_disp, cv2.COLOR_BGR2RGB)        
        img = Image.fromarray(cv2.resize(frame_disp, (600, 480))) 
        imgtk = ImageTk.PhotoImage(image=img)
        self.image_label.imgtk = imgtk
        self.image_label.configure(image=imgtk)
        #=== 베터리 표기 앞으로 ===
        self.battery_label.lift() 
        self.root.after(15, self.update_stream)
        
    #=== 배터리 모니터링 함수 === 
    def update_battery_label(self):
        while True:
            percent = read_battery_percent()
            if percent is not None:
                self.battery_label.config(text=f"배터리: {percent}%")
            time.sleep(2)

    #=== 변수 초기화 함수 ===
    def reset_compare_flag(self):
        global compared
        compared = False

    def register_device(self):
        subprocess.Popen([VENV_PYTHON, "/home/mbp_mk2/venv/register_device.py"])

    # === 스트리밍 종료 함수 ===
    def close_app(self):
        self.picam2.stop()
        self.streaming_enabled = False
        self.root.destroy()

    #=== QR 생성함수(등록용) ===
    def start_qr_process(self):
        self.streaming_enabled = False
        try:
            self.picam2.stop()
        except:
            pass

        self.qr_uuid = str(uuid.uuid4())
        self.qr_start_time = time.time() 
        db.collection("qr_auth").document(self.qr_uuid).set({
            "status": "pending",
            "created_at": datetime.datetime.utcnow()
        })
        print(f"[QR 등록] UUID: {self.qr_uuid}")

        self.show_qr_code(self.qr_uuid)
        self.check_qr_status()

    #=== QR 생성함수(잠금해제용) ===
    def start_qrgate_process(self):
        self.streaming_enabled = False
        try:
            self.picam2.stop()
        except:
            pass

        self.qr_uuid = str(uuid.uuid4())
        db.collection("qr_gate").document(self.qr_uuid).set({
            "status": "pending",
            "created_at": datetime.datetime.utcnow()
        })

        self.show_qr_code(self.qr_uuid)
        self.check_qrgate_status()

    #=== QR 출력함수 ===
    def show_qr_code(self, qr_data):
        qr_img = qrcode.make(qr_data).resize((480, 480)).convert("RGB")
        qr_img = ImageTk.PhotoImage(image=qr_img)
        self.image_label.configure(image=qr_img)
        self.image_label.imgtk = qr_img
        self.root.update_idletasks()
        self.root.update()

    #=== QR스캔 프로세스(등록) ===
    def check_qr_status(self, start_time=None):
        if not self.root.winfo_exists():
            return
        if start_time is None:
            start_time = time.time()

        try:# 인증 성공시
            doc = db.collection("qr_auth").document(self.qr_uuid).get()
            if doc.exists and doc.to_dict().get("status") == "authenticated":
                print("[QR 인증 완료 → 얼굴 등록 시작]")

                #=== 스트리밍 제개, 얼굴등록 모드 실행 ===      
                self.picam2.stop()        
                self.streaming_enabled = True
                self.registration_mode = True
                self.registration_detect_count = 0
                self.picam2.configure(self.picam2.create_preview_configuration(main={"size": (480, 360)}))
                self.picam2.start()
                self.update_stream()

                # === QR 이미지 제거 ===
                blank_img = ImageTk.PhotoImage(Image.new("RGB", (480, 360), color=(0, 0, 0)))
                self.image_label.configure(image=blank_img)
                self.image_label.imgtk = blank_img
                return
            #=== 10초 미감지시 scan화면으로 복귀 ===
            elif time.time() - start_time >= 10:
                print("[QR 인증 시간 초과 → 스트리밍 복귀]")
                self.picam2.configure(self.picam2.create_preview_configuration(main={"size": (480, 360)}))
                self.picam2.start()
                self.streaming_enabled = True
                return

            else:
                self.root.after(1000, lambda: self.check_qr_status(start_time))

        except Exception as e:
            print(f"[QR 상태 확인 오류] {e}")
            
    #=== QR스캔 프로세스(잠금해제) ===
    def check_qrgate_status(self, start_time=None):
        if not self.root.winfo_exists():
            return
        if start_time is None:
            start_time = time.time()

        try: # 인증 성공시
            doc = db.collection("qr_gate").document(self.qr_uuid).get()            
            if doc.exists and doc.to_dict().get("status") == "authenticated":
                send_unlock()
                print("[QR 인증 완료 → 스트리밍 재시작]")
                self.picam2.configure(self.picam2.create_preview_configuration(main={"size": (480, 360)}))
                self.picam2.start()
                self.streaming_enabled = True
                return

            #=== 10초 미감지시 메시지 전송 ===
            if time.time() - start_time <= 10:
                self.root.after(1000, lambda: self.check_qrgate_status(start_time))
                return

            print("[QR 인증 실패 → 푸시 알림 전송 시작]")
            send_alarm()
            image_url, timestamp = self.get_latest_scan_face_image()

            from google.oauth2 import service_account
            import google.auth.transport.requests
            import requests
            import json

            cred_path = "/home/mbp_mk2/test2-61728-firebase-adminsdk-fbsvc-c93ab6c1a3.json"
            project_id = "test2-61728"
            fcm_url = f"https://fcm.googleapis.com/v1/projects/{project_id}/messages:send"
            #=== FCM토큰 검색 ===
            docs = db.collection("fcm_tokens").stream()
            tokens = []
            for doc in docs:
                data = doc.to_dict()
                if "token" in data:
                    tokens.append(data["token"])

            if not tokens:
                print("[FCM] 전송할 토큰이 없습니다.")
            else:
                scoped_credentials = service_account.Credentials.from_service_account_file(
                    cred_path,
                    scopes=["https://www.googleapis.com/auth/firebase.messaging"]
                )
                auth_req = google.auth.transport.requests.Request()
                scoped_credentials.refresh(auth_req)
                access_token = scoped_credentials.token

                headers = {
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json; UTF-8"
                }

                for token in tokens:
                    message = {
                        "message": {
                            "token": token,
                            "notification": {
                                "title": "출입인증 실패",
                                "body": f"출입 기록: {timestamp}",
                                "image": image_url
                            },
                            "data": {
                                "imageUrl": image_url,
                                "timestamp": timestamp
                            }
                        }
                    }

                    response = requests.post(fcm_url, headers=headers, data=json.dumps(message))
                    print(f"\n전송 대상: {token}")
                    print("Status Code:", response.status_code)
                    print("Response:", response.text)

            # === scan 화면으로 복귀 ===
            print("[스트리밍 재시작]")
            self.picam2.configure(self.picam2.create_preview_configuration(main={"size": (480, 360)}))
            self.picam2.start()
            self.streaming_enabled = True

        except Exception as e:
            print(f"[QR 상태 확인 오류] {e}")


    #=== 최근 스캔한 이미지를 가져오는 함수 ===
    def get_latest_scan_face_image(self):
        blobs = list(bucket.list_blobs(prefix="scan_face/"))
        latest_blob = None
        latest_time = None

        print(f"[DEBUG] 총 {len(blobs)}개의 scan_face 이미지가 감지됨.")

        for blob in blobs:
            if blob.name.endswith(".jpg"):
                print(f"[DEBUG] 확인 중: {blob.name}, 업데이트 시각: {blob.updated}")
                if not latest_time or blob.updated > latest_time:
                    latest_blob = blob
                    latest_time = blob.updated

        if latest_blob:
            print(f"[DEBUG] 선택된 최신 이미지: {latest_blob.name}")
            url = latest_blob.generate_signed_url(datetime.timedelta(minutes=10))
            print(f"[DEBUG] 생성된 signed_url: {url}")
            return url, latest_time.strftime("%Y-%m-%d %H:%M:%S")
        else:
            print("[ERROR] scan_face 폴더에 유효한 이미지가 없습니다.")
            return "", ""
            
    #=== 데이터 초기화 함수 ===
    def reset_all_data(self):
        from tkinter import messagebox
        import threading

        # === 사용자 확인 다이얼로그 ===
        confirm = messagebox.askyesno("초기화 확인", "정말로 모든 데이터를 삭제하시겠습니까?\n이 작업은 되돌릴 수 없습니다.")
        if not confirm:
            print("[초기화 취소됨] 사용자가 '아니오'를 선택함.")
            return

        # === 삭제 로직은 별도 스레드에서 실행 ===
        def do_reset():
            print("[초기화] Firestore 및 Storage 데이터 삭제 시작...")

            # === Firestore 컬렉션 삭제 ===
            collections_to_delete = ["users", "logs", "fcm_tokens", "qr_gate", "qr_auth", "scan"]
            for col in collections_to_delete:
                docs = db.collection(col).stream()
                for doc in docs:
                    db.collection(col).document(doc.id).delete()
                    print(f"[Firestore] 삭제됨: {col}/{doc.id}")

            # === Storage 디렉토리 삭제 ===
            folders_to_clear = ["user_face/", "scan_face/"]
            for folder in folders_to_clear:
                blobs = list(bucket.list_blobs(prefix=folder))
                for blob in blobs:
                    blob.delete()
                    print(f"[Storage] 삭제됨: {blob.name}")

            print("[초기화 완료] 모든 데이터 및 이미지 삭제 완료.")
            messagebox.showinfo("초기화 완료", "모든 데이터가 삭제되었습니다.")

        threading.Thread(target=do_reset).start()
        
# === main 코드: GUI 실행 ===
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceAppGUI(root)
    root.mainloop()

