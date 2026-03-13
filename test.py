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
            app_vec = tf.gather(self.embedding, app_id)
            app_vec_expanded = tf.expand_dims(app_vec, 0)
            
            # 입력값 스케일링 (큰 값 들어올 경우 대비 안전장치)
            usage_time_scaled = tf.expand_dims(usage_time, 0) * 0.1
            inputs = tf.concat([app_vec_expanded, usage_time_scaled], axis=1)

            pred = tf.matmul(inputs, self.w1) + self.b1
            pred_final = tf.reshape(pred, [1])

            # 1. 다시 MSE(Square)로 복귀하여 학습 속도 향상
            loss_mean = tf.reduce_mean(tf.square(pred_final - label))

        trainable_vars = [self.w1, self.embedding]
        grads = tape.gradient(loss_mean, trainable_vars)

        # 2. 범위를 -50.0 ~ 50.0으로 대폭 확장 (기존 1.0은 너무 작았음)
        # 또는 tf.clip_by_global_norm을 써서 방향성을 보존
        clipped_grads, _ = tf.clip_by_global_norm(grads, 50.0)

        # 3. 업데이트 적용 (학습률 0.1 추천)
        self.w1.assign_sub(self.learning_rate * clipped_grads[0])
        
        indices = tf.reshape(app_id, [1, 1])
        specific_grad = tf.gather(clipped_grads[1], app_id)
        new_vec = app_vec - (self.learning_rate * specific_grad)
        
        # 가중치 값 자체가 무한대로 가는 것을 방지 (NaN 최종 수비)
        new_vec = tf.clip_by_value(new_vec, -100.0, 100.0)
        new_vec_reshaped = tf.reshape(new_vec, [1, 8])

        self.embedding.scatter_nd_update(indices, new_vec_reshaped)

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
