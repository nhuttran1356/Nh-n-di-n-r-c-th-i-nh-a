import cv2
import torch
import time

# define a video capture object
vid = cv2.VideoCapture(0)
# model = torch.hub.load('E:\pcpj\pythonProject\yolov5-master', 'custom', path='best.pt')
model = torch.hub.load('yolov5-master', 'custom', path='best.pt', source='local')

classes = model.names  # name of objects
device = 'cuda' if torch.cuda.is_available() else 'cpu'
a=0

def score_frame(frame):
    """
    Takes a single frame as input, and scores the frame using yolo5 model.
    param frame: input frame in numpy/list/tuple format.
    return: Labels and Coordinates of objects detected by model in the frame.q
    """
    model.to(device)
    frame = [frame]
    results = model(frame)
    labels, cord = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
    return labels, cord


def class_to_label(x):
    """
    For a given label value, return corresponding string label.
    :param x: numeric label
    :return: corresponding string label
    """
    return classes[int(x)]


def plot_boxes(results, frame,a):
    """
    Takes a frame and its results as input, and plots the bounding boxes and label on to the frame.
    param results: contains labels and coordinates predicted by model on the given frame.
    param frame: Frame which has been scored.
    return: Frame with bounding boxes and labels ploted on it.
    """
    labels, cord = results
    print("labels", labels)
    print("cord", cord[:, :-1])
    clas_0 = 0
    clas_1 = 1
    clas_2 = 2
    clas_3 = 3

    if len(labels) != 0:
        print("list is not empty")
        for label in labels:
            if label == clas_0:
                print("send Alucan")
                # time.sleep(10)
            elif label == clas_1:
                print("send Glass")
            elif label == clas_2:
                print("send HDPE")
            elif label == clas_3:
                print("send 123")
                print("send PET")
                print(a)
                # if (a%10==0):
                #     time.sleep(10)
            else:
                print("wrong objects")
    else:
        print("list is empty")
        print("no objects")

    n = len(labels)
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    for i in range(n):
        row = cord[i]
        # print("predict", round(cord[i][4], 2))
        if row[4] >= 0.2:
            x1, y1, x2, y2 = int(row[0] * x_shape), int(row[1] * y_shape), int(row[2] * x_shape), int(
                row[3] * y_shape)
            bgr = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 2)
            cv2.putText(frame, class_to_label(labels[i]) + " " + str(round(row[4], 2)), (x1, y1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, bgr, 2)

    return frame


while True:
    # Capture the video frame
    # by frame
    a=a+1
    ret, frame = vid.read()
    results = score_frame(frame)
    print(results)
    frame = plot_boxes(results, frame,a)
    # Display the resulting frame
    cv2.imshow('frame', frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()