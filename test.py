import tensorflow as tf

class PowerModelModule(tf.Module):
    def __init__(self):
        super().__init__()
        bucket_size = 10000
        self.embedding = tf.Variable(
            tf.random.uniform([bucket_size, 8], -0.1, 0.1), name='embedding')
        self.w1 = tf.Variable(
            tf.random.normal([9, 1], stddev=0.1), name='w1')
        self.b1 = tf.Variable(tf.constant([15.0], dtype=tf.float32), name='b1')
        self.learning_rate = tf.constant(0.025)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[], dtype=tf.int32, name='app_id'),
        tf.TensorSpec(shape=[1], dtype=tf.float32, name='usage_time'),
        tf.TensorSpec(shape=[1], dtype=tf.float32, name='label')
    ])
    def train(self, app_id, usage_time, label):
        with tf.GradientTape() as tape:
            # 1. 앱별 임베딩 추출
            app_vec = tf.gather(self.embedding, app_id)
            app_vec_expanded = tf.expand_dims(app_vec, 0)
            
            # 2. 특징 결합
            usage_time_expanded = tf.expand_dims(usage_time, 0)
            inputs = tf.concat([app_vec_expanded, usage_time_expanded], axis=1)

            # 3. 예측값 계산
            pred = tf.matmul(inputs, self.w1) + self.b1
            pred_final = tf.reshape(pred, [1])

            # 4. Huber Loss (안정적인 오차 계산)
            diff = pred_final - label
            loss = tf.where(
                tf.abs(diff) <= 1.0,
                0.5 * tf.square(diff),
                tf.abs(diff) - 0.5
            )

        # 5. 가중치 업데이트 대상
        grads = tape.gradient(loss, [self.w1, self.embedding])
        clipped_grads = [tf.clip_by_value(g, -1.0, 1.0) for g in grads]

        # 6. 업데이트 (Flex 연산 방지를 위해 assign_sub와 scatter_update 우회)
        # W1 업데이트
        self.w1.assign_sub(self.learning_rate * clipped_grads[0])
        
        # [핵심 수정] Embedding 업데이트: 
        # scatter_nd_update 대신, gradient를 이용해 해당 행만 업데이트하는 TFLite 표준 방식 사용
        grad_emb = clipped_grads[1]
        
        # 특정 앱의 gradient만 추출해서 업데이트 값 계산
        specific_grad = tf.gather(grad_emb, app_id)
        new_vec = app_vec - (self.learning_rate * specific_grad)
        
        # scatter_nd_update 대신 tensor_scatter_nd_update를 변수 할당식으로 사용
        # (이 방식이 TFLite 인터프리터에서 더 잘 작동함)
        indices = tf.reshape(app_id, [1, 1])
        updates = tf.reshape(new_vec, [1, 8])
        
        updated_embedding = tf.tensor_scatter_nd_update(self.embedding, indices, updates)
        self.embedding.assign(updated_embedding)

        return {'loss': tf.reduce_mean(loss), 'pred': pred_final}

    # ... (predict, import, export 함수는 이전과 동일) ...
    @tf.function(input_signature=[
        tf.TensorSpec(shape=[], dtype=tf.int32, name='app_id'),
        tf.TensorSpec(shape=[1], dtype=tf.float32, name='usage_time')
    ])
    def predict(self, app_id, usage_time):
        app_vec = tf.gather(self.embedding, app_id)
        app_vec_expanded = tf.expand_dims(app_vec, 0)
        usage_time_expanded = tf.expand_dims(usage_time, 0)
        inputs = tf.concat([app_vec_expanded, usage_time_expanded], axis=1)
        pred = tf.matmul(inputs, self.w1) + self.b1
        return {'timeout': tf.reshape(pred, [1])}

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[10000, 8], dtype=tf.float32, name='emb_in'),
        tf.TensorSpec(shape=[9, 1], dtype=tf.float32, name='w1_in'),
        tf.TensorSpec(shape=[1], dtype=tf.float32, name='b1_in')
    ])
    def import_weights(self, emb_in, w1_in, b1_in):
        self.embedding.assign(emb_in)
        self.w1.assign(w1_in)
        self.b1.assign(b1_in)
        return {'status': tf.constant([1.0], dtype=tf.float32)}

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

# --- 빌드 부분 ---
model = PowerModelModule()
tf.saved_model.save(model, 'temp_model', signatures={
    'train': model.train, 'predict': model.predict, 'import': model.import_weights, 'export': model.export_weights
})

converter = tf.lite.TFLiteConverter.from_saved_model('temp_model')
# Flex 연산 지원 옵션을 제거 (순수 TFLite로만 빌드되도록 강제)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_model = converter.convert()

with open('power_manager_model.tflite', 'wb') as f:
    f.write(tflite_model)
