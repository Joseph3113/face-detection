import cv2
import numpy as np
import os
import time

TEMP_DIR = "template_wajah"

piksel_awal = (220, 120)
piksel_akhir = (440, 400)

attendance_log = {}

def capture_template():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Gunakan CAP_DSHOW untuk stabilitas di Windows

    if not cap.isOpened():
        print("Kamera gagal dibuka. Pastikan kamera tidak sedang digunakan oleh aplikasi lain.")
        return

    print("Tekan 's' untuk menyimpan foto atau tekan 'q' untuk membatalkan.")
    nama = input("Masukkan nama untuk template: ")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame dari kamera.")
            break

        display_frame = frame.copy()
        cv2.rectangle(display_frame, piksel_awal, piksel_akhir, (162, 208, 37), 2)
        cv2.imshow("Ambil Wajah", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            os.makedirs(os.path.join(TEMP_DIR, nama), exist_ok=True)
            template_path = os.path.join(TEMP_DIR, nama, "template.jpg")

            cropped_template = frame[piksel_awal[1]:piksel_akhir[1], piksel_awal[0]:piksel_akhir[0]]
            cv2.imwrite(template_path, cropped_template)
            print(f"Foto wajah template untuk {nama} disimpan di {template_path}")
            break

        if key == ord('q'):
            print("Pengambilan template dibatalkan.")
            break

    cap.release()
    cv2.destroyAllWindows()

def face_detection():
    templates = {}
    for person_name in os.listdir(TEMP_DIR):
        person_dir = os.path.join(TEMP_DIR, person_name)
        if os.path.isdir(person_dir):
            template_path = os.path.join(person_dir, "template.jpg")
            template = cv2.imread(template_path, 0)
            if template is not None:
                templates[person_name] = template

    if not templates:
        print("Tidak ada template wajah yang ditemukan.")
        return

    print("Mulai deteksi wajah. Tekan 'q' untuk keluar.")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Kamera gagal dibuka.")
        return

    detection_start_time = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal membaca frame dari kamera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.rectangle(frame, piksel_awal, piksel_akhir, (162, 208, 37), 2)
        region_of_interest = gray[piksel_awal[1]:piksel_akhir[1], piksel_awal[0]:piksel_akhir[0]]

        for person_name, template in templates.items():
            w, h = template.shape[::-1]
            result = cv2.matchTemplate(region_of_interest, template, cv2.TM_CCOEFF_NORMED)
            threshold = 0.7
            loc = np.where(result >= threshold)

            if len(loc[0]) > 0:
                if person_name not in detection_start_time:
                    detection_start_time[person_name] = time.time()
                
                elapsed_time = time.time() - detection_start_time[person_name]
                if elapsed_time >= 2:
                    if person_name not in attendance_log:
                        attendance_log[person_name] = time.strftime("%Y-%m-%d %H:%M:%S")
                        print(f"{person_name} tercatat hadir pada {attendance_log[person_name]}.")

                    cv2.putText(frame, f"Hadir: {person_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                for pt in zip(*loc[::-1]):
                    top_left = (pt[0] + piksel_awal[0], pt[1] + piksel_awal[1])
                    bottom_right = (top_left[0] + w, top_left[1] + h)
                    cv2.rectangle(frame, top_left, bottom_right, (162, 208, 37), 2)
                    cv2.putText(frame, person_name, (top_left[0], top_left[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (162, 208, 37), 2)
            else:
                detection_start_time.pop(person_name, None)

        cv2.imshow("Deteksi Wajah", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def view_attendance():
    if not attendance_log:
        print("Belum ada yang tercatat hadir.")
    else:
        print("\nDaftar Kehadiran:")
        for person_name, timestamp in attendance_log.items():
            print(f"- {person_name}: {timestamp}")

if __name__ == "__main__":
    os.makedirs(TEMP_DIR, exist_ok=True)

    while True:
        print("\nMenu:")
        print("1. Ambil Template Wajah")
        print("2. Mulai Deteksi Wajah")
        print("3. Lihat Kehadiran")
        print("4. Keluar")
        choice = input("Pilih menu (1/2/3/4): ")

        if choice == "1":
            capture_template()
        elif choice == "2":
            face_detection()
        elif choice == "3":
            view_attendance()
        elif choice == "4":
            print("Keluar dari program.")
            break
        else:
            print("Pilihan tidak valid. Coba lagi.")
