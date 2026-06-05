from insightface.app import FaceAnalysis

app = FaceAnalysis(name='buffalo_l')

app.prepare(
    ctx_id=-1,
    det_size=(640,640)
)

print("Loaded successfully!")
