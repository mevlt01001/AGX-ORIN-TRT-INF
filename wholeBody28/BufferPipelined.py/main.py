from wholeBody28.libs import *
import cv2, time
from collections import deque

if __name__ == "__main__":
    w, h = 1280, 720
    out_w, out_h = 1280, 736
    w_ratio = w / out_w
    h_ratio = h / out_h
    pipeline = (
    "v4l2src device=/dev/video4 ! "
    f"video/x-raw,format=YUY2,width={w},height={h},framerate=30/1 ! "
    "videoconvert ! "
    "video/x-raw,format=BGR ! "  
    "appsink"
    )
    camera = CameraAsync(camera_index=4, queue_size=10)
    camera.cam.set(cv2.CAP_PROP_FPS, 30)
    camera.cam.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    camera.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    # cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    # cap.set(cv2.CAP_PROP_FPS, 30)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)


    model = TRTInfer(engine_path="EngineFolder/yolov9_s_wholebody28_refine_post_0100_1x3x736x1280.engine")
    model.load_engine(output_shape=(300, 7))
    ops = PrePostOP()

    camera.start()

    fpss = deque(maxlen=50)
    preOp_latecies = deque(maxlen=50)
    infer_latecies = deque(maxlen=50)
    postOp_latecies = deque(maxlen=50)

    fps=1
    pre_lat = 1
    infer_lat = 1
    post_lat = 1

    fpss.append(1)
    preOp_latecies.append(1)
    infer_latecies.append(1)
    postOp_latecies.append(1)

    while True:

        frame = camera.get()
        if frame is None:
            print(f"{'[ERROR][CAMERA]':.<40}: frame is None")
            continue
        # success, frame = cap.read()
        # if not success:
        #     print(f"{'[ERROR][CAMERA]':.<40}: frame is None")
        #     break
        
        start = time.perf_counter()
        start_pre = time.perf_counter()
        ops.preprocess(frame, model.input_buffers[0].host)
        end_pre = time.perf_counter()
        
        start_infer = time.perf_counter()
        model.bind(frame)
        end_infer = time.perf_counter()

        start_post = time.perf_counter()
        result = ops.postprocess(model.output_buffers[0].host, w_ratio, h_ratio, score_threshold=0.3)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Pre: {pre_lat:.2f}ms", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Infer: {infer_lat:.2f}ms", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Post: {post_lat:.2f}ms", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        for class_name, id, color, bbox in result:
            if id in [21,22,26]:
                cx = (bbox[0] + bbox[2]) // 2
                cy = (bbox[1] + bbox[3]) // 2
                cv2.circle(frame, (cx, cy), 3, color, 3)
                continue
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(frame, class_name, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        end_post = time.perf_counter()
        end = time.perf_counter()

        fps = 1 / (end - start)
        pre_lat = (end_pre - start_pre)*1000
        infer_lat = (end_infer - start_infer)*1000
        post_lat = (end_post - start_post)*1000
        
        fpss.append(fps)
        preOp_latecies.append(pre_lat)
        infer_latecies.append(infer_lat)
        postOp_latecies.append(post_lat)

        fps = sum(fpss) / len(fpss)
        pre_lat = sum(preOp_latecies) / len(preOp_latecies)
        infer_lat = sum(infer_latecies) / len(infer_latecies)
        post_lat = sum(postOp_latecies) / len(postOp_latecies)

        
    camera.stop()

    