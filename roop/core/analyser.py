import insightface
import roop.core.globals

FACE_ANALYSER = None


def get_face_analyser(resolution=None, confidence=None):
    if confidence is None:
        confidence = 0.5

    global FACE_ANALYSER
    if FACE_ANALYSER is None:
        FACE_ANALYSER = insightface.app.FaceAnalysis(name='buffalo_l', providers=roop.core.globals.providers)
        if not resolution:
            resolution = (640, 640)
        FACE_ANALYSER.prepare(ctx_id=0, det_thresh=confidence, det_size=resolution)
    return FACE_ANALYSER


def get_single_face(img_data, resolution=None, face_analyser=None, confidence=None):
    if not face_analyser:
        face = get_face_analyser(resolution, confidence).get(img_data)
    else:
        face = face_analyser.get(img_data)
    try:
        return sorted(face, key=lambda x: x.bbox[0])[0]
    except IndexError:
        return None


def get_all_faces(img_data):
    try:
        return get_face_analyser().get(img_data)
    except IndexError:
        return None