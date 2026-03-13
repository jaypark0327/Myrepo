import tensorflow as tf

class PowerModelModule(tf.Module):
    def __init__(self):
        super().__init__()
        bucket_size = 10000
        # 초기값 범위를 넓혀서 빠릿함을 회복함
        self.embedding = tf.Variable(
            tf.random.uniform([bucket_size, 8], -0.5, 0.5), name='embedding')
        self.w1 = tf.Variable(
            tf.random.normal([9, 1], stddev=0.1), name='w1')
        self.b1 = tf.Variable(tf.constant([15.0], dtype=tf.float32), name='b1')
        self.learning_rate = tf.constant(0.1) # 빠릿빠릿한 학습률

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[], dtype=tf.int32, name='app_id'),
        tf.TensorSpec(shape=[1], dtype=tf.float32, name='usage_time'),
        tf.TensorSpec(shape=[1], dtype=tf.float32, name='label')
    ])
    def train(self, app_id, usage_time, label):
        # 1. Gradient 계산
        with tf.GradientTape() as tape:
            app_vec = tf.gather(self.embedding, app_id)
            app_vec_expanded = tf.expand_dims(app_vec, 0)
            
            # 입력값 스케일링 (큰 값 대비)
            usage_time_scaled = tf.expand_dims(usage_time, 0) * 0.1
            inputs = tf.concat([app_vec_expanded, usage_time_scaled], axis=1)

            pred = tf.matmul(inputs, self.w1) + self.b1
            pred_final = tf.reshape(pred, [1])

            # MSE Loss로 복귀 (빠릿한 학습)
            loss_mean = tf.reduce_mean(tf.square(pred_final - label))

        trainable_vars = [self.w1, self.embedding]
        grads = tape.gradient(loss_mean, trainable_vars)

        # 2. Gradient Clipping (폭주는 막되 에너지는 유지)
        clipped_grads, _ = tf.clip_by_global_norm(grads, 50.0)

        # 3. [중요] StatefulPartitionedCall 방지를 위한 순수 텐서 업데이트
        # W1 업데이트
        new_w1 = self.w1 - self.learning_rate * clipped_grads[0]
        self.w1.assign(new_w1)
        
        # Embedding 업데이트 (이 방식이 TFLite에서 가장 안전함)
        indices = tf.reshape(app_id, [1, 1])
        specific_grad = tf.gather(clipped_grads[1], app_id)
        new_vec = app_vec - (self.learning_rate * specific_grad)
        
        # 가중치 자체를 클리핑해서 NaN 원천 차단
        new_vec = tf.clip_by_value(new_vec, -50.0, 50.0)
        new_vec_reshaped = tf.reshape(new_vec, [1, 8])

        # 변수를 직접 건드리는 scatter_nd_update 대신 tensor_용 사용 후 assign
        updated_emb = tf.tensor_scatter_nd_update(self.embedding, indices, new_vec_reshaped)
        self.embedding.assign(updated_emb)

        return {
            'loss': tf.reshape(loss_mean, [1, 1]),
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

# --- TFLite 변환 ---
model = PowerModelModule()
tf.saved_model.save(model, 'power_saved_model', signatures={
    'train': model.train, 'predict': model.predict, 'import': model.import_weights, 'export': model.export_weights
})

converter = tf.lite.TFLiteConverter.from_saved_model('power_saved_model')
# 순수 TFLite 연산만 사용하도록 강제 (Select TF Ops 제거)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_model = converter.convert()

with open('power_manager_model.tflite', 'wb') as f:
    f.write(tflite_model)
