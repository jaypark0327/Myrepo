private float getLearnedTimeout(String packageName) {
    if (mModelInterpreter == null) return 30.0f;

    try {
        String signatureKey = "predict";
        // 모델 내부에서 실제 사용하는 입력/출력 키 목록을 가져옴
        String[] inputNames = mModelInterpreter.getSignatureInputs(signatureKey);
        String[] outputNames = mModelInterpreter.getSignatureOutputs(signatureKey);

        Map<String, Object> inputs = new HashMap<>();
        // 모델이 기대하는 첫 번째, 두 번째 입력 키에 데이터를 매핑
        inputs.put(inputNames[0], new int[]{getAppId(packageName)}); // app_id
        inputs.put(inputNames[1], new float[]{0.0f});               // usage_time

        Map<String, Object> outputs = new HashMap<>();
        float[][] prediction = new float[1][1];
        // 모델이 기대하는 첫 번째 출력 키(예: 'output')에 결과 배열 매핑
        outputs.put(outputNames[0], prediction);

        mModelInterpreter.runSignature(inputs, outputs, signatureKey);
        
        float result = prediction[0][0];
        return (result > 5.0f) ? result : 30.0f; // 최소값 방어 코드

    } catch (Exception e) {
        Slog.e(TAG, "Prediction failed: " + e.getMessage());
        return 30.0f;
    }
}

private static final String SAVE_PATH = "/data/system/power_model_state.bin";

// 1. 현재 학습된 가중치를 파일로 저장
private void saveModelWeights() {
    if (mModelInterpreter == null) return;
    try {
        Map<String, Object> inputs = new HashMap<>();
        Map<String, Object> outputs = new HashMap<>();

        float[][] emb = new float[10000][8];
        float[][] w1 = new float[9][1];
        float[] b1 = new float[1];

        outputs.put("emb_out", emb);
        outputs.put("w1_out", w1);
        outputs.put("b1_out", b1);

        mModelInterpreter.runSignature(inputs, outputs, "export");

        // 파일로 직렬화 (간단히 ByteBuffer 사용)
        try (FileOutputStream fos = new FileOutputStream(SAVE_PATH)) {
            ObjectOutputStream oos = new ObjectOutputStream(fos);
            oos.writeObject(emb);
            oos.writeObject(w1);
            oos.writeObject(b1);
            Slog.i(TAG, "TFLite: Weights saved to disk. Bias: " + b1[0]);
        }
    } catch (Exception e) {
        Slog.e(TAG, "TFLite: Save failed", e);
    }
}

// 2. 저장된 파일이 있으면 모델에 주입
private void loadModelWeights() {
    File file = new File(SAVE_PATH);
    if (!file.exists()) return;

    try (FileInputStream fis = new FileInputStream(SAVE_PATH)) {
        ObjectInputStream ois = new ObjectInputStream(fis);
        float[][] emb = (float[][]) ois.readObject();
        float[][] w1 = (float[][]) ois.readObject();
        float[] b1 = (float[]) ois.readObject();

        Map<String, Object> inputs = new HashMap<>();
        inputs.put("emb_in", emb);
        inputs.put("w1_in", w1);
        inputs.put("b1_in", b1);

        Map<String, Object> outputs = new HashMap<>();
        int[] status = new int[1];
        outputs.put("status", status);

        mModelInterpreter.runSignature(inputs, outputs, "import");
        Slog.i(TAG, "TFLite: Weights loaded from disk. Bias: " + b1[0]);
    } catch (Exception e) {
        Slog.e(TAG, "TFLite: Load failed", e);
    }
}
