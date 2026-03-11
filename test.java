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
