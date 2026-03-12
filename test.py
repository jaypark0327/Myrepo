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
        tf.TensorSpec(shape=[1], dtype=tf.int32, name='app_id'),
        tf.TensorSpec(shape=[1], dtype=tf.float32, name='usage_time'),
        tf.TensorSpec(shape=[1, 1], dtype=tf.float32, name='label')
    ])
    def train(self, app_id, usage_time, label):
        app_vec = tf.gather(self.embedding, app_id)
        app_vec = tf.reshape(app_vec, [1, 8])
        x = tf.concat([app_vec, tf.reshape(usage_time, [1, 1])], axis=1)
        pred = tf.matmul(x, self.w1) + self.b1
        diff = pred - label
        
        grad_w = tf.matmul(tf.transpose(x), diff)
        self.w1.assign_sub(self.learning_rate * grad_w)
        self.b1.assign_sub(self.learning_rate * tf.reshape(diff, [1]))
        
        grad_emb = tf.reshape(diff * self.w1[:8, 0], [1, 8])
        self.embedding.assign(tf.tensor_scatter_nd_update(
            self.embedding, tf.reshape(app_id, [1, 1]), app_vec - self.learning_rate * grad_emb))
            
        return {'loss': tf.square(diff)}

    # --- 추가: 현재 가중치를 추출하는 서명 ---
    @tf.function(input_signature=[])
    def export_weights(self):
        return {
            'emb_out': self.embedding,
            'w1_out': self.w1,
            'b1_out': self.b1
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
        return {'status': tf.constant(1)}

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
