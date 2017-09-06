# coding: utf-8
import cv2
import sys
from PIL import Image

def catchUsbVideo(window_name, camera_idx):
    cv2.namedWindow(window_name)

    # 视频来源，可以来自一段已经存好的视频，也可以直接来自USB摄像头
    cap = cv2.VideoCapture(camera_idx)

    # 告诉OpenCV使用人脸识别分类器
    classifier = cv2.CascadeClassifier("C:/Users/ZhangXuemiao/tools/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml")

    # 识别出人脸后要画框的颜色，RGB格式
    color = (0, 255, 0)

    while cap.isOpened():
        ok, frame = cap.read() # 读取一帧数据
        if not ok:
            break

        # 将当前帧转换成灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
        faceRects = classifier.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors=3, minSize=(32, 32))
        if len(faceRects) > 0:      # 大于0则检测到人脸
            for faceRect in faceRects:  #单独框出每一张人脸
                x, y, w, h = faceRect
                cv2.rectangle(frame, (x-10, y-10), (x+w+10, y+h+10), color, 2)

        # 显示图像并等待10毫秒按键输入，输入'q'退出程序
        cv2.imshow(window_name, frame)
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break

    # 释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()


def CatchPICFromVideoVideo(window_name, camera_idx, catch_pic_num, path_name):
    cv2.namedWindow(window_name)

    # 视频来源，可以来自一段已经存好的视频，也可以直接来自USB摄像头
    cap = cv2.VideoCapture(camera_idx)

    # 告诉OpenCV使用人脸识别分类器
    classifier = cv2.CascadeClassifier("C:/Users/ZhangXuemiao/tools/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml")

    # 识别出人脸后要画框的颜色，RGB格式
    color = (0, 255, 0)

    num = 0
    while cap.isOpened():
        ok, frame = cap.read() # 读取一帧数据
        if not ok:
            break

        # 将当前帧转换成灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
        faceRects = classifier.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors=3, minSize=(32, 32))
        if len(faceRects) > 0:      # 大于0则检测到人脸
            for faceRect in faceRects:  #单独框出每一张人脸
                x, y, w, h = faceRect

                # 将当前帧保存为图片
                image_name = '%s/%d.jpg'%(path_name, num)
                image = frame[y-10:y+h+10, x-10:x+w+10]
                cv2.imwrite(image_name, image)

                num += 1
                if num > catch_pic_num :    # 如果超过指定保存数量，退出循环
                    break

                # 画出矩形框
                cv2.rectangle(frame, (x-10, y-10), (x+w+10, y+h+10), color, 2)

                # 显示当前捕捉到多少人脸图片了， 这样站在那里拍摄是心里有数，不用两眼一抹黑傻等着
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, 'num:%d' % (num), (x+30, y+30), font, 1, (255, 0, 255), 4)

        # 超过指定最大保存数量结束程序
        if num > catch_pic_num: break

        # 显示图像并等待10毫秒按键输入，输入'q'退出程序
        cv2.imshow(window_name, frame)
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break

    # 释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()


def test01():
    if len(sys.argv) != 2:
        print("Usage:%s camera_id\r\n" % (sys.argv[0]))
    else:
        # print(sys.argv[0], sys.argv[1]) # C:/Users/ZhangXuemiao/tensorflow/FaceRecognition/GetObtainStream.py 0
        catchUsbVideo("截取视频流", int(sys.argv[1]))


def test02():
    if len(sys.argv) != 4:
        print("Usage:%s camera_id face_num_max path_name\r\n" % (sys.argv[0]))
    else:
        CatchPICFromVideoVideo("截取人脸", int(sys.argv[1]), int(sys.argv[2]), sys.argv[3])


if __name__ == '__main__':
    test02()