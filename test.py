import tensorflow as tf

class PowerModelModule(tf.Module):
    def __init__(self):
        super().__init__()
        self.bucket_size = 10000
        # 초기값 범위를 넓혀 빠릿함 유지
        self.embedding = tf.Variable(
            tf.random.uniform([self.bucket_size, 8], -0.5, 0.5), name='embedding')
        self.w1 = tf.Variable(
            tf.random.normal([9, 1], stddev=0.1), name='w1')
        self.b1 = tf.Variable(tf.constant([15.0], dtype=tf.float32), name='b1')
        self.learning_rate = tf.constant(0.1)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[], dtype=tf.int32, name='app_id'),
        tf.TensorSpec(shape=[1], dtype=tf.float32, name='usage_time'),
        tf.TensorSpec(shape=[1], dtype=tf.float32, name='label')
    ])
    def train(self, app_id, usage_time, label):
        with tf.GradientTape() as tape:
            # 1. 앱 임베딩 추출
            app_vec = tf.gather(self.embedding, app_id)
            app_vec_expanded = tf.expand_dims(app_vec, 0)
            
            # 2. 입력 결합 및 예측 (스케일링 포함)
            usage_time_scaled = tf.expand_dims(usage_time, 0) * 0.1
            inputs = tf.concat([app_vec_expanded, usage_time_scaled], axis=1)
            pred = tf.matmul(inputs, self.w1) + self.b1
            pred_final = tf.reshape(pred, [1])

            # 3. MSE Loss (빠른 수렴)
            loss = tf.reduce_mean(tf.square(pred_final - label))

        # 4. Gradient 계산 및 클리핑
        grads = tape.gradient(loss, [self.w1, self.embedding])
        clipped_grads, _ = tf.clip_by_global_norm(grads, 50.0)

        # 5. [중요] Flex Ops 없이 가중치 업데이트
        # W1은 직접 assign_sub 사용 (표준 연산)
        self.w1.assign_sub(self.learning_rate * clipped_grads[0])
        
        # Embedding 업데이트: Flex 연산인 scatter_update를 피하기 위해 
        # 원-핫 인코딩 마스크를 사용하여 해당 행만 업데이트
        mask = tf.one_hot(app_id, depth=self.bucket_size) # [10000]
        mask = tf.expand_dims(mask, 1) # [10000, 1]
        
        # 업데이트할 변화량 계산
        grad_emb = clipped_grads[1]
        delta = self.learning_rate * grad_emb # [10000, 8]
        
        # 특정 앱 ID 행만 살아있는 델타 생성
        masked_delta = delta * mask
        
        # 뺄셈 후 결과값 클리핑 (NaN 방지)
        updated_emb = tf.clip_by_value(self.embedding - masked_delta, -50.0, 50.0)
        self.embedding.assign(updated_emb)

        return {
            'loss': tf.reshape(loss, [1, 1]),
            'pred': tf.reshape(pred_final, [1, 1])
        }

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[], dtype=tf.int32, name='app_id'),
        tf.TensorSpec(shape=[1], dtype=tf.float32, name='usage_time')
    ])
    def predict(self, app_id, usage_time):
        app_vec = tf.gather(self.embedding, app_id)
        app_vec_expanded = tf.expand_dims(app_vec, 0)
        usage_time_scaled = tf.expand_dims(usage_time, 0) * 0.1
        
        inputs = tf.concat([app_vec_expanded, usage_time_scaled], axis=1)
        pred = tf.matmul(inputs, self.w1) + self.b1
        return {'timeout': tf.reshape(pred, [1, 1])}

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[10000, 8], dtype=tf.float32, name='emb_in'),
        tf.TensorSpec(shape=[9, 1], dtype=tf.float32, name='w1_in'),
        tf.TensorSpec(shape=[1], dtype=tf.float32, name='b1_in')
    ])
    def import_weights(self, emb_in, w1_in, b1_in):
        self.embedding.assign(emb_in)
        self.w1.assign(w1_in)
        self.b1.assign(b1_in)
        return {'status': tf.reshape(tf.constant(1.0), [1, 1])}

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1], dtype=tf.float32, name='dummy_in')
    ])
    def export_weights(self, dummy_in):
        fake_op = dummy_in * 0.0
        return {
            'emb_out': tf.identity(self.embedding),
            'w1_out': tf.identity(self.w1),
            'b1_out': tf.identity(self.b1) + fake_op
        }

# --- 빌드 로직 ---
model = PowerModelModule()
tf.saved_model.save(model, 'pure_tflite_model', signatures={
    'train': model.train, 'predict': model.predict, 'import': model.import_weights, 'export': model.export_weights
})

converter = tf.lite.TFLiteConverter.from_saved_model('pure_tflite_model')
# Flex 연산을 명시적으로 배제하고 순수 TFLite 연산만 허용
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_model = converter.convert()

with open('power_manager_model.tflite', 'wb') as f:
    f.write(tflite_model)
