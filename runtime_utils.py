def get_execution_providers():
    try:
        import onnxruntime as ort
    except ImportError:
        return ["CPUExecutionProvider"]

    available = set(ort.get_available_providers())
    preferred = (
        "CoreMLExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    )
    providers = [provider for provider in preferred if provider in available]

    return providers or ["CPUExecutionProvider"]
