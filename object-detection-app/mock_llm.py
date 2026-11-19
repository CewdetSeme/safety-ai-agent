class MockSafetyLLM:
    def __init__(self, unsafe_every_n_calls: int = 3):
        self.n = 0
        self.unsafe_every = unsafe_every_n_calls

    def analyze(self, image_b64: str) -> dict:
        self.n += 1
        unsafe = self.n % self.unsafe_every == 0
        return {
            "unsafe": unsafe,
            "summary": "Worker without helmet detected" if unsafe else "All clear",
            "persons": [
                {"track_id": 1, "helmet": not unsafe, "confidence": 0.9, "notes": ""}
            ],
        }