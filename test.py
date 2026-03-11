import tensorflow as tf
import os

# 1. 가중치 변수 정의 (전역 변수로 설정)
bucket_size = 10000
embedding = tf.Variable(tf.random.uniform([bucket_size, 8], -0.1, 0.1), name='embedding')
w1 = tf.Variable(tf.random.normal([9, 1], stddev=0.1), name='w1')
b1 = tf.Variable(tf.zeros([1]), name='b1')
learning_rate = tf.constant(0.01)

# 2. 내보낼 모듈 정의
class PowerModelModule(tf.Module):
    def __init__(self):
        super().__init__()
        self.v = [embedding, w1, b1, learning_rate] # 변수 추적용

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1], dtype=tf.int32, name='app_id'),
        tf.TensorSpec(shape=[1], dtype=tf.float32, name='usage_time')
    ])
    def predict(self, app_id, usage_time):
        app_vec = tf.gather(embedding, app_id)
        app_vec = tf.reshape(app_vec, [1, 8])
        usage_time_vec = tf.reshape(usage_time, [1, 1])
        
        x = tf.concat([app_vec, usage_time_vec], axis=1)
        prediction = tf.matmul(x, w1) + b1
        return {'output': prediction}

    @tf.function(input_signature=[
        tf.TensorSpec(shape=[1], dtype=tf.int32, name='app_id'),
        tf.TensorSpec(shape=[1], dtype=tf.float32, name='usage_time'),
        tf.TensorSpec(shape=[1, 1], dtype=tf.float32, name='label')
    ])
    def train(self, app_id, usage_time, label):
        app_vec = tf.gather(embedding, app_id)
        app_vec = tf.reshape(app_vec, [1, 8])
        usage_time_vec = tf.reshape(usage_time, [1, 1])
        
        x = tf.concat([app_vec, usage_time_vec], axis=1)
        pred = tf.matmul(x, w1) + b1
        diff = pred - label
        
        # SGD 업데이트
        grad_w = tf.matmul(tf.transpose(x), diff)
        w1.assign_sub(learning_rate * grad_w)
        b1.assign_sub(learning_rate * tf.reshape(diff, [1]))
        
        grad_emb = tf.reshape(diff * w1[:8, 0], [1, 8])
        updated_emb = app_vec - learning_rate * grad_emb
        
        embedding.assign(tf.tensor_scatter_nd_update(
            embedding, tf.reshape(app_id, [1, 1]), updated_emb))
            
        return {'loss': tf.square(diff)}

# 3. SavedModel로 임시 저장
model_module = PowerModelModule()
SAVED_MODEL_DIR = "temp_power_model"
tf.saved_model.save(
    model_module, 
    SAVED_MODEL_DIR, 
    signatures={
        'predict': model_module.predict,
        'train': model_module.train
    }
)

# 4. TFLite 변환 (SavedModel로부터 로드)
converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_model = converter.convert()

# 5. 최종 tflite 파일 저장
with open('power_manager_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("성공: power_manager_model.tflite 파일이 생성되었습니다.")
