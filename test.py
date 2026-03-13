import tensorflow as tf
import os

class PowerModelModule(tf.Module):
    def __init__(self):
        super().__init__()
        bucket_size = 10000
        # 1. 가중치 초기화
        self.embedding = tf.Variable(
            tf.random.uniform([bucket_size, 8], -0.1, 0.1), name='embedding')
        self.w1 = tf.Variable(
            tf.random.normal([9, 1], stddev=0.1), name='w1')
        self.b1 = tf.Variable(tf.constant([15.0], dtype=tf.float32), name='b1')
        
        # 학습률 (안정적인 수치)
        self.learning_rate = tf.constant(0.025)

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[], dtype=tf.int32, name='app_id'),
        tf.TensorSpec(shape=[1], dtype=tf.float32, name='usage_time'),
        tf.TensorSpec(shape=[1], dtype=tf.float32, name='label')
    ])
    def train(self, app_id, usage_time, label):
        with tf.GradientTape() as tape:
            # 1. 앱별 임베딩 추출 [1, 8]
            app_vec = tf.gather(self.embedding, app_id)
            app_vec_expanded = tf.expand_dims(app_vec, 0)
            
            # 2. 특징 결합 [1, 9]
            usage_time_expanded = tf.expand_dims(usage_time, 0)
            inputs = tf.concat([app_vec_expanded, usage_time_expanded], axis=1)

            # 3. 예측값 계산
            pred = tf.matmul(inputs, self.w1) + self.b1
            pred_final = tf.reshape(pred, [1])

            # 4. Huber Loss (안정적인 오차 계산)
            diff = pred_final - label
            loss_val = tf.where(
                tf.abs(diff) <= 1.0,
                0.5 * tf.square(diff),
                tf.abs(diff) - 0.5
            )
            loss_mean = tf.reduce_mean(loss_val)

        # 5. 가중치 업데이트
        grads = tape.gradient(loss_mean, [self.w1, self.embedding])
        clipped_grads = [tf.clip_by_value(g, -1.0, 1.0) for g in grads]

        # W1 업데이트
        self.w1.assign_sub(self.learning_rate * clipped_grads[0])
        
        # Embedding 업데이트 (TFLite 호환 방식으로 우회)
        indices = tf.reshape(app_id, [1, 1])
        specific_grad = tf.gather(clipped_grads[1], app_id)
        new_vec = app_vec - (self.learning_rate * specific_grad)
        new_vec_reshaped = tf.reshape(new_vec, [1, 8])

        updated_emb = tf.tensor_scatter_nd_update(self.embedding, indices, new_vec_reshaped)
        self.embedding.assign(updated_emb)

        # Java의 float[1][1]과 맞추기 위해 [1, 1]로 리셰이프
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
        usage_time_expanded = tf.expand_dims(usage_time, 0)
        
        inputs = tf.concat([app_vec_expanded, usage_time_expanded], axis=1)
        pred = tf.matmul(inputs, self.w1) + self.b1
        
        # Java의 float[1][1]과 맞추기 위해 [1, 1]로 리셰이프
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
export_dir = 'power_manager_temp'
tf.saved_model.save(model, export_dir, signatures={
    'train': model.train,
    'predict': model.predict,
    'import': model.import_weights,
    'export': model.export_weights
})

converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
# 순수 TFLite 연산만 사용 (Flex 연산 에러 방지)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_model = converter.convert()

with open('power_manager_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("TFLite Model Build Success! All outputs are shaped as [1, 1].")
