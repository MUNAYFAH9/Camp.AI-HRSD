import cv2

# تهيئة المصدر
cap = cv2.VideoCapture(0)

# تهيئة مكتبة تعرف الوجه
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    # قراءة الإطار الحالي من الكاميرا
    ret, frame = cap.read()

    # تحويل الإطار إلى اللون الرمادي
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # اكتشاف الوجوه في الإطار الرمادي
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # رسم مربع حول الوجوه المكتشفة
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # عرض الإطار المعدل
    cv2.imshow('Face Detection', frame)

    # انتظار الضغط على مفتاح ESC لإنهاء البرنامج
    if cv2.waitKey(1) == 27:
        break

# إغلاق الكاميرا وتدمير النوافذ المفتوحة
cap.release()
cv2.destroyAllWindows()