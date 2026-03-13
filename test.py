import tensorflow as tf
import numpy as np

# 자바 String.hashCode() 결과 예시 (미리 계산하거나 수집한 값)
# 예: "com.android.chrome".hashCode() -> 12345678
# 예: "com.google.android.youtube".hashCode() -> 87654321
X_raw = np.array([[12345678], [87654321], [45678901]], dtype=np.float32)
y_train = np.array([[120], [300], [60]], dtype=np.float32) # 초 단위 타겟값

# [중요] 정규화: 해시코드는 값이 너무 크므로 2^31로 나눠서 0~1 사이로 맞춤
X_train = X_raw / 2147483647.0

# 단순 회귀 모델 구성
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=8, activation='relu', input_shape=[1]),
    tf.keras.layers.Dense(units=1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=1000, verbose=0)

# TFLite로 변환
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# 파일 저장
with open('pms_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("pms_model.tflite 생성 완료!")
