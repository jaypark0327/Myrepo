Welcome to the Myrepo wiki!
import tensorflow as tf

# 1. 가중치 변수 정의
bucket_size = 10000
embedding = tf.Variable(tf.random.uniform([bucket_size, 8], -0.1, 0.1), name='embedding')
w1 = tf.Variable(tf.random.normal([9, 1], stddev=0.1), name='w1')
b1 = tf.Variable(tf.zeros([1]), name='b1')
learning_rate = tf.constant(0.01)

# 2. 추론 함수 (Predict)
@tf.function
def predict(app_id, usage_time):
    app_vec = tf.gather(embedding, app_id)
    app_vec = tf.reshape(app_vec, [1, 8])
    usage_time_vec = tf.reshape(usage_time, [1, 1])
    
    x = tf.concat([app_vec, usage_time_vec], axis=1)
    prediction = tf.matmul(x, w1) + b1
    return {'output': prediction}

# 3. 학습 함수 (Train)
@tf.function
def train(app_id, usage_time, label):
    app_vec = tf.gather(embedding, app_id)
    app_vec = tf.reshape(app_vec, [1, 8])
    usage_time_vec = tf.reshape(usage_time, [1, 1])
    
    x = tf.concat([app_vec, usage_time_vec], axis=1)
    pred = tf.matmul(x, w1) + b1
    diff = pred - label
    
    # SGD Update logic
    grad_w = tf.matmul(tf.transpose(x), diff)
    w1.assign_sub(learning_rate * grad_w)
    b1.assign_sub(learning_rate * tf.reshape(diff, [1]))
    
    grad_emb = tf.reshape(diff * w1[:8, 0], [1, 8])
    updated_emb = app_vec - learning_rate * grad_emb
    
    embedding.assign(tf.tensor_scatter_nd_update(
        embedding, tf.reshape(app_id, [1, 1]), updated_emb))
        
    return {'loss': tf.square(diff)}

# 4. Concrete Functions 생성
predict_fn = predict.get_concrete_function(
    tf.TensorSpec(shape=[1], dtype=tf.int32, name='app_id'),
    tf.TensorSpec(shape=[1], dtype=tf.float32, name='usage_time')
)

train_fn = train.get_concrete_function(
    tf.TensorSpec(shape=[1], dtype=tf.int32, name='app_id'),
    tf.TensorSpec(shape=[1], dtype=tf.float32, name='usage_time'),
    tf.TensorSpec(shape=[1, 1], dtype=tf.float32, name='label')
)

# 5. [핵심!] tf.Module을 사용하여 서명을 그룹화
class ExportModule(tf.Module):
    def __init__(self):
        super().__init__()
        self.predict = predict_fn
        self.train = train_fn

# 6. TFLite 변환 (Concrete Functions 리스트가 아닌 모듈 전체를 넘김)
module = ExportModule()
converter = tf.lite.TFLiteConverter.from_concrete_functions(
    [module.predict, module.train], module
)

# 표준 연산자 설정
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_model = converter.convert()

# 7. 파일 저장
with open('power_manager_model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Final model with multiple signatures generated: power_manager_model.tflite")
