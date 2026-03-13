import tensorflow as tf
import os

bucket_size = 10000

class PowerModelModule(tf.Module):
    def __init__(self):
        super().__init__()
        # 초기 Bias를 15.0으로 설정
        self.embedding = tf.Variable(tf.random.uniform([bucket_size, 8], -0.1, 0.1), name='embedding')
        self.w1 = tf.Variable(tf.random.normal([9, 1], stddev=0.1), name='w1')
        self.b1 = tf.Variable(tf.constant([15.0], dtype=tf.float32), name='b1')
        self.learning_rate = tf.constant(0.01)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1], dtype=tf.int32, name='app_id'),
        tf.TensorSpec(shape=[1], dtype=tf.float32, name='usage_time')
    ])
    def predict(self, app_id, usage_time):
        app_vec = tf.gather(self.embedding, app_id)
        app_vec = tf.reshape(app_vec, [1, 8])
        x = tf.concat([app_vec, tf.reshape(usage_time, [1, 1])], axis=1)
        prediction = tf.matmul(x, self.w1) + self.b1
        return {'output': prediction}

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[], dtype=tf.int32, name='app_id'),
        tf.TensorSpec(shape=[1], dtype=tf.float32, name='usage_time'),
        tf.TensorSpec(shape=[1], dtype=tf.float32, name='label')
    ])
    def train(self, app_id, usage_time, label):
        with tf.GradientTape() as tape:
            # 1. 앱별 임베딩 추출
            app_vec = tf.gather(self.embedding, app_id)
            app_vec = tf.reshape(app_vec, [1, 8])

            # 2. 특징 결합 (사용 시간 포함 9차원)
            # usage_time이 너무 크면 (예: ms 단위) 여기서 NaN 위험이 있으니 
            # Java에서 '초' 단위로 넘겨주는지 꼭 확인해!
            inputs = tf.concat([app_vec, tf.reshape(usage_time, [1, 1])], axis=1)

            # 3. 예측값 계산 (y = Wx + b)
            pred = tf.matmul(inputs, self.w1) + self.b1
            pred = tf.reshape(pred, [1])

            # 4. Loss 계산 (Huber Loss 스타일로 NaN 방지)
            diff = pred - label
            # 오차가 클 때는 제곱(square) 대신 절대값(abs)에 가깝게 처리하여 
            # Gradient가 폭발하는 것을 방지함
            loss = tf.where(
                tf.abs(diff) <= 1.0,
                0.5 * tf.square(diff),
                tf.abs(diff) - 0.5
            )

        # 5. 가중치 업데이트 대상 설정 (Bias인 b1은 제외하여 15초 고정)
        trainable_vars = [self.w1, self.embedding]
        grads = tape.gradient(loss, trainable_vars)

        # 6. Gradient Clipping (가장 중요!)
        # 기울기가 -1.0 ~ 1.0 범위를 벗어나지 않게 잘라내서 NaN 발생 원천 차단
        clipped_grads = [tf.clip_by_value(g, -1.0, 1.0) for g in grads]

        # 7. 업데이트 적용
        self.w1.assign_sub(self.learning_rate * clipped_grads[0])
        
        # 특정 앱의 임베딩만 업데이트
        new_app_vec = app_vec - self.learning_rate * clipped_grads[1]
        self.embedding.assign(tf.tensor_scatter_nd_update(
            self.embedding, [[app_id]], new_app_vec))

        return {'loss': tf.reduce_mean(loss), 'pred': pred}


    # --- 추가: 현재 가중치를 추출하는 서명 ---
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1], dtype=tf.float32, name='dummy_in')
    ])
    def export_weights(self, dummy_in):
        # dummy_in을 사용하지 않으면 컨버터가 서명에서 빼버릴 수 있음
        # 결과에 영향을 주지 않도록 0을 곱해서 더해주는 식의 '가짜 연산' 추가
        fake_op = dummy_in * 0.0
        
        return {
            'emb_out': tf.identity(self.embedding),
            'w1_out': tf.identity(self.w1),
            'b1_out': tf.identity(self.b1) + fake_op # 연산에 포함시켜 서명 유지
        }


    # --- 추가: 가중치를 주입하는 서명 ---
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[bucket_size, 8], dtype=tf.float32, name='emb_in'),
        tf.TensorSpec(shape=[9, 1], dtype=tf.float32, name='w1_in'),
        tf.TensorSpec(shape=[1], dtype=tf.float32, name='b1_in')
    ])
    def import_weights(self, emb_in, w1_in, b1_in):
        self.embedding.assign(emb_in)
        self.w1.assign(w1_in)
        self.b1.assign(b1_in)
        # 반드시 float32 타입의 리스트 [1.0]으로 반환
        return {'status': tf.constant([1.0], dtype=tf.float32)}


# 저장 및 변환
model_module = PowerModelModule()
SAVED_MODEL_DIR = "temp_power_model"
tf.saved_model.save(model_module, SAVED_MODEL_DIR, 
                    signatures={
                        'predict': model_module.predict, 
                        'train': model_module.train,
                        'export': model_module.export_weights,
                        'import': model_module.import_weights
                    })

converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_model = converter.convert()

with open('power_manager_model.tflite', 'wb') as f:
    f.write(tflite_model)
