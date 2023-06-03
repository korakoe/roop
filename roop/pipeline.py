from .core.swapper import get_face_swapper
from .core.analyser import get_face_analyser, get_all_faces, get_single_face
from .core.globals import providers
from .core.utils import HiddenPrints
import cv2
from tqdm import tqdm
from moviepy.editor import *
import torch
from codeformer.app import cv2_inference_app

if 'ROCMExecutionProvider' in providers:
    del torch


class RoopPipeline:
    def __init__(self, resolution=None, model_path=None, confidence=None, codeformer_fidelity=0.7):
        """
        The base RoopPipeline class

        :param resolution: The resolution of the analyser | may result in higher quality output
        :param model_path: The path to the model | no path will use the model next to the file using this class
        """
        self.resolution = resolution
        self.model_path = model_path

        self.swapper = get_face_swapper(model_path)
        self.analyser = get_face_analyser(resolution, confidence)
        self.get_face = get_single_face
        self.get_all_faces = get_all_faces
        
        self.fidelity = codeformer_fidelity

    def process_image(self, source_img, target_img, all_faces=False, codeformer=False):
        """
        Processes a cv2 image

        :param source_img: The image you want to deepfake with | cv2 image
        :param target_img: The image you want to deepfake onto | cv2 image
        :param all_faces: Whether to deepfake all faces
        :param codeformer: Whether to post-process with codeformer
        :return: A cv2 image output of the finished deepfake
        """
        if all_faces:
            many_faces = get_all_faces(source_img)
            source_face = self.get_face(source_img, face_analyser=self.analyser)
            if many_faces:
                for face in tqdm(many_faces):
                    target_img = self.swapper.get(target_img, face, source_face, paste_back=True)

            result = target_img
        else:
            face = self.get_face(target_img, face_analyser=self.analyser)
            source_face = self.get_face(source_img, face_analyser=self.analyser)
            result = self.swapper.get(target_img, face, source_face, paste_back=True)

        if codeformer:
            with HiddenPrints():
                result = cv2_inference_app(
                    image=result,
                    background_enhance=False,
                    face_upsample=True,
                    upscale=1,
                    codeformer_fidelity=self.fidelity,
                )

        return result

    def process_image_file(self, source_img, target_img, out_path=None, all_faces=False, codeformer=False):
        """
        Processes an image file, and saves it if supplied an out_path

        :param source_img: The image you want to deepfake with | cv2 image
        :param target_img: The image you want to deepfake onto | cv2 image
        :param out_path: The output of the processed image | No out path will only return the image
        :param all_faces: Whether to deepfake all faces
        :param codeformer: Whether to post-process with codeformer
        :return: A cv2 image output of the finished deepfake
        """
        if all_faces:
            frame = cv2.imread(target_img)
            many_faces = get_all_faces(source_img)
            source_face = self.get_face(source_img, face_analyser=self.analyser)
            if many_faces:
                for face in tqdm(many_faces):
                    frame = self.swapper.get(frame, face, source_face, paste_back=True)

            result = frame
        else:
            frame = cv2.imread(target_img)
            face = self.get_face(frame, face_analyser=self.analyser)
            source_face = self.get_face(cv2.imread(source_img), face_analyser=self.analyser)
            result = self.swapper.get(frame, face, source_face, paste_back=True)

        if codeformer:
            with HiddenPrints():
                result = cv2_inference_app(
                    image=result,
                    background_enhance=False,
                    face_upsample=True,
                    upscale=1,
                    codeformer_fidelity=self.fidelity,
                )

        if out_path:
            cv2.imwrite(out_path, result)
            print("\n\nImage saved as:", out_path, "\n\n")
        return result

    def process_video_file(self, source_img, video_path, out_path=None, all_faces=False, codeformer=False):
        """
        Processes a video file, and saves it if supplied an out_path

        :param source_img: The path to the image you want to deepfake with
        :param video_path: The path to the video you want to deepfake onto
        :param out_path: The output path of the video | No out path will only return the clip
        :param all_faces: Whether to deepfake all faces
        :param codeformer: Whether to post-process with codeformer
        :return: A MoviePy video clip of the finished deepfake
        """
        source_face = self.get_face(cv2.imread(source_img))

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        videoclip = VideoFileClip(video_path)
        audioclip = videoclip.audio

        frames = []

        success, image = cap.read()
        count = 0
        while success:
            frames.append(image)  # save frame as JPEG file
            success, image = cap.read()
            count += 1

        with tqdm(total=len(frames), desc="Processing", unit="frame", dynamic_ncols=True,
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]') as progress:
            for idx, process_frame in enumerate(frames):
                frame = process_frame
                try:
                    if all_faces:
                        many_faces = get_all_faces(frame)
                        source_face = self.get_face(frame, face_analyser=self.analyser)
                        if many_faces:
                            for face in tqdm(many_faces):
                                frame = self.swapper.get(frame, face, source_face, paste_back=True)

                        result = frame

                        if codeformer:
                            with HiddenPrints():
                                result = cv2_inference_app(
                                    image=result,
                                    background_enhance=False,
                                    face_upsample=True,
                                    upscale=1,
                                    codeformer_fidelity=self.fidelity,
                                )

                        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

                        frames[idx] = result
                    else:
                        face = self.get_face(frame)
                        if face:
                            result = self.swapper.get(frame, face, source_face, paste_back=True)

                            if codeformer:
                                with HiddenPrints():
                                    result = cv2_inference_app(
                                        image=result,
                                        background_enhance=False,
                                        face_upsample=True,
                                        upscale=1,
                                        codeformer_fidelity=self.fidelity,
                                    )

                            result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

                            frames[idx] = result
                            progress.set_postfix(status='Face found', refresh=True)
                        else:
                            progress.set_postfix(status='No Faces', refresh=True)
                            frames[idx] = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB)
                except Exception:
                    progress.set_postfix(status='Error', refresh=True)
                    pass
                progress.update(1)

        clip = ImageSequenceClip(frames, fps=fps)
        clip.audio = audioclip
        if out_path:
            print("Saving clip...", end=" ")
            clip.write_videofile(out_path)
        print("Done!")
        return clip

    def live_swap(self, source_img, webcam_number=0, all_faces=False, codeformer=False):
        """
        Starts a live deepfake session, this will continue indefinitely unless you press the "q" key
        :param source_img: The image to deepfake onto you live | cv2 image
        :param webcam_number: The number of the webcam to use
        :param codeformer: Whether to post-process with codeformer
        :param all_faces: Whether to deepfake all faces
        """

        vid = cv2.VideoCapture(webcam_number)

        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            ret, img = vid.read()
            img = cv2.flip(img, 1)
            if all_faces:
                many_faces = get_all_faces(img)
                source_face = self.get_face(source_img, face_analyser=self.analyser)
                if many_faces:
                    for face in tqdm(many_faces):
                        img = self.swapper.get(img, face, source_face, paste_back=True)

                result = img
            else:
                face = self.get_face(img, face_analyser=self.analyser)
                source_face = self.get_face(source_img, face_analyser=self.analyser)
                if face:
                    result = self.swapper.get(img, face, source_face, paste_back=True)
                else:
                    print("No face found")
                    result = img

            if codeformer:
                with HiddenPrints():
                    result = cv2_inference_app(
                        image=result,
                        background_enhance=False,
                        face_upsample=True,
                        upscale=1,
                        codeformer_fidelity=self.fidelity,
                    )

            cv2.imshow("Roop", result)