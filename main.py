from PyQt5 import QtCore, QtGui, QtWidgets
from dlcv_hw1_ui import Ui_MainWindow
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import os
import sys
import cv2
import numpy as np


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    
    folder_path = None
    Q1_img_list = []
    Q1_2d_points = []
    Q1_3d_points = []
    Q1_img_shape = None
    
    Q2_img_list = []
    Q2_2d_points = []
    Q2_3d_points = []
    
    Q3_L_img_path = ""
    Q3_R_img_path = ""

    Q4_img_one_path = ""
    Q4_img_two_path = ""

    def __init__(self):
         super().__init__()
         self.setupUi(self)
         self.setControl()
    
    def setControl(self):
        self.LoadFloder_btn.clicked.connect(self.LoadFloder_btn_func)
        self.Load_Image_R_btn.clicked.connect(self.Load_Image_R_btn_func)
        self.Load_Imgae_L_btn.clicked.connect(self.Load_Image_L_btn_func)
        self.FindCorners_btn.clicked.connect(self.FindCorners_btn_func)
        self.FindIntrinsic_btn.clicked.connect(self.FindIntrinsic_btn_func)
        self.FindExrtinsic_btn.clicked.connect(self.FindExrtinsic_btn_func)
        self.FindDistortion_btn.clicked.connect(self.FindDistortion_btn_func)
        self.ShowResult_btn.clicked.connect(self.Q1_ShowResult_btn_func)
        self.Show_words_on_board_btn.clicked.connect(self.Show_words_on_board_btn_func)
        self.Show_words_vertical_btn.clicked.connect(self.Show_words_vertical_btn_func)
        self.Stereo_disparity_map_btn.clicked.connect(self.Stereo_disparity_map_btn_func)
        self.SIFT_LoadImage_1_btn.clicked.connect(self.SIFT_LoadImage_1_btn_func)
        self.SIFT_LoadImage_2_btn.clicked.connect(self.SIFT_LoadImage_2_btn_func)
        self.SIFT_Keypoints_btn.clicked.connect(self.SIFT_Keypoints_btn_func)
        self.SIFT_Matched_Keypoints_btn.clicked.connect(self.SIFT_Matched_Keypoints_btn_func)
    
    def error_message(self, errortext): # Error message box: errortext(Input the error message that you want to show)
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Critical)
        msg.setWindowTitle("Error")
        msg.setText(errortext)
        msg.setInformativeText("Please try again.")
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()
    
    def LoadFloder_btn_func(self):
        self.folder_path = str(QFileDialog.getExistingDirectory(self, "Select Directory"))
    
    def Load_Image_L_btn_func(self): # FileDialog.getOpenFileName return (file path, file type)
        self.Q3_L_img_path = str(QFileDialog.getOpenFileName(self,"Open Image", None, "Image Files (*.png *.jpg *.bmp)")[0])
            
    def Load_Image_R_btn_func(self):
        self.Q3_R_img_path = str(QFileDialog.getOpenFileName(self,"Open Image", None, "Image Files (*.png *.jpg *.bmp)")[0])

    def SIFT_LoadImage_1_btn_func(self):
        self.Q4_img_one_path = str(QFileDialog.getOpenFileName(self,"Open Image", None, "Image Files (*.png *.jpg *.bmp)")[0])
    
    def SIFT_LoadImage_2_btn_func(self):
        self.Q4_img_two_path = str(QFileDialog.getOpenFileName(self,"Open Image", None, "Image Files (*.png *.jpg *.bmp)")[0])

    def FindCorners_btn_func(self):
        if self.folder_path is None:
            self.error_message(errortext="Q1 incorrect data path!")
            return
        for i in range(15):
            img_path = os.path.join(self.folder_path, str(i+1)+".bmp")
            if not os.path.exists(img_path):
                continue
            cur_img = cv2.imread(img_path)
            self.Q1_img_list.append(cur_img)
        
        objp = np.zeros((11 * 8, 3), np.float32)# Define 3d points 11*8 points (88, 3 dimentions)
        objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)# (x,y,z=0)
        for img in self.Q1_img_list:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)# on HW1 ppt
            # https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray_img, (11,8))
            
            # If found, add object points(3d), image points(2d) (after refining them)
            if ret == True:
                corners = cv2.cornerSubPix(gray_img, corners, (5, 5), (-1, -1), criteria)# in order to increase accuracy (refining)
                self.Q1_2d_points.append(corners)
                self.Q1_3d_points.append(objp)
                
                self.Q1_img_shape = gray_img.shape[::-1]
                draw_img = img.copy()
                # Draw and display the corners
                cv2.drawChessboardCorners(draw_img, (11, 8), corners, ret)
                cv2.imshow("Find Corners", cv2.resize(draw_img, (1024, 1024)))
                cv2.waitKey(700)
    
    def FindIntrinsic_btn_func(self):
        if len(self.Q1_3d_points) < 1:
            self.error_message(errortext="1.1 Find corners should execute first!")
            return
        ret, ins, dist, rvec, tvec=cv2.calibrateCamera (self.Q1_3d_points, self.Q1_2d_points, self.Q1_img_shape, None, None)
        print("Q1.2\nIntrinsic:")
        print(ins)
    

    def FindExrtinsic_btn_func(self):
        if len(self.Q1_3d_points) < 1:
            self.error_message(errortext="1.1 Find corners should execute first!")
            return
        img_num = int(self.FindExrtinsic_spinBox.value())# get image num in 1~15
        ret, ins, dist, rvec, tvec=cv2.calibrateCamera (self.Q1_3d_points, self.Q1_2d_points, self.Q1_img_shape, None, None)
        if img_num<1 or img_num>15:# check input img number
            self.error_message(errortext="Error number! (Number range:1~15)")
        else:
            rotation_matrix = cv2.Rodrigues(rvec[img_num-1])[0]#　Transform rotation vector into rotation matrix
            # Concate rotation matrix and translation matrix along the second axis
            extrinsic_matrix = np.hstack((rotation_matrix, tvec[img_num-1]))
            print("Q1.3\nExtrinsic: Image:", img_num, ".bmp")
            print(extrinsic_matrix)
    
    def FindDistortion_btn_func(self):
        if len(self.Q1_3d_points) < 1:
            self.error_message(errortext="1.1 Find corners should execute first!")
            return
        ret, ins, dist, rvec, tvec=cv2.calibrateCamera (self.Q1_3d_points, self.Q1_2d_points, self.Q1_img_shape, None, None)
        print("Q1.4\nDistortion:")
        print(dist)
    
    def Q1_ShowResult_btn_func(self):
        if len(self.Q1_3d_points) < 1:
            self.error_message(errortext="1.1 Find corners should execute first!")
            return
        img_num = int(self.FindExrtinsic_spinBox.value())# get image num in 1~15
        gray_img = cv2.cvtColor(self.Q1_img_list[img_num-1], cv2.COLOR_BGR2GRAY)
        ret, ins, dist, rvec, tvec=cv2.calibrateCamera (self.Q1_3d_points, self.Q1_2d_points, self.Q1_img_shape, None, None)
        result_img = cv2.undistort(gray_img, ins, dist)
        cv2.imshow("Distorted", cv2.resize(gray_img, (512, 512)))
        cv2.imshow("Undistorted", cv2.resize(result_img, (512, 512)))
        cv2.waitKey(700)
    
    def Q2_loadDataset(self, alphabet_db):# load Q2_image and alphabet_db
        if len(self.Q2_img_list)<1:
            for i in range(5):
                img_path = os.path.join(self.folder_path, str(i+1)+".bmp")
                if not os.path.exists(img_path):
                    continue
                cur_img = cv2.imread(img_path)
                self.Q2_img_list.append(cur_img)
        alphabet_db_path = os.path.join(self.folder_path, "Q2_db", alphabet_db)
        if not os.path.exists(alphabet_db_path):
            fs = None
            textboxValue = None
            self.error_message(errortext="Q2 incorrect data path!")
            load = False
        else:
            fs = cv2.FileStorage(alphabet_db_path, cv2.FILE_STORAGE_READ)
            textboxValue = self.AR_Show_textEdit.toPlainText()
            if len(textboxValue) > 6:
                self.error_message(errortext="Q2 Just can input 6 alphabets!")
                load = False
            else:
                load = True

        return load, fs, textboxValue

    def Show_words_on_board_btn_func(self):
        load, fs, textboxValue = self.Q2_loadDataset(alphabet_db="alphabet_db_onboard.txt")
        if load:
            if len(self.Q2_2d_points)<1:
                objp = np.zeros((11 * 8, 3), np.float32)
                objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
                img_shape = None
                for img in self.Q2_img_list:
                    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)# on HW1 ppt
                    ret, corners = cv2.findChessboardCorners(gray_img, (11,8))
                    if ret == True:
                        corners = cv2.cornerSubPix(gray_img, corners, (5, 5), (-1, -1), criteria)# refining
                        self.Q2_2d_points.append(corners)
                        self.Q2_3d_points.append(objp)
                        img_shape = gray_img.shape[::-1]
            else:
                gray_img = cv2.cvtColor(self.Q2_img_list[0], cv2.COLOR_BGR2GRAY)
                img_shape = gray_img.shape[::-1]
            # Calibrate 5 images to get intrinsic, extrinsic, distortion, rotation vector, and translation vector parameters.
            ret, ins, dist, rvec, tvec = cv2.calibrateCamera (self.Q2_3d_points, self.Q2_2d_points, img_shape, None, None)
            start_point = [[7,5,0], [4,5,0], [1,5,0], [7,2,0], [4,2,0], [1,2,0]]# the designated position
            for i, img in enumerate(self.Q2_img_list):
                draw_img = img.copy()
                for text in range(len(textboxValue)):
                    alphabet = str(textboxValue[text])
                    charpoints = fs.getNode(alphabet).mat()# Get the 3D object coordinates of the alphabet
                    charpoints = charpoints.reshape(-1, 3).astype(np.float32)
                    charpoints = charpoints + start_point[text]# Apply translation to 3D object coordinates to move to the designated position
                    newCharPoints_2d, _ = cv2.projectPoints(charpoints, rvec[i], tvec[i], ins, dist)

                    for j in range(0, len(newCharPoints_2d), 2):
                        pointA = tuple(newCharPoints_2d[j][0].astype(int))
                        pointB = tuple(newCharPoints_2d[j+1][0].astype(int))
                        cv2.line(draw_img, pointA, pointB, (0, 0, 255), 8)
                cv2.imshow("Show words on board", cv2.resize(draw_img, (1024, 1024)))
                cv2.waitKey(700)
            cv2.destroyAllWindows()
        
    def Show_words_vertical_btn_func(self):
        load, fs, textboxValue = self.Q2_loadDataset(alphabet_db="alphabet_db_vertical.txt")
        if load:
            if len(self.Q2_2d_points)<1:
                objp = np.zeros((11 * 8, 3), np.float32)
                objp[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2)
                img_shape = None
                for img in self.Q2_img_list:
                    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 30, 0.001)# on HW1 ppt
                    ret, corners = cv2.findChessboardCorners(gray_img, (11,8))
                    if ret == True:
                        corners = cv2.cornerSubPix(gray_img, corners, (5, 5), (-1, -1), criteria)# refining
                        self.Q2_2d_points.append(corners)
                        self.Q2_3d_points.append(objp)
                        img_shape = gray_img.shape[::-1]
            else:
                gray_img = cv2.cvtColor(self.Q2_img_list[0], cv2.COLOR_BGR2GRAY)
                img_shape = gray_img.shape[::-1]
            # Calibrate 5 images to get intrinsic, extrinsic, distortion, rotation vector, and translation vector parameters.
            ret, ins, dist, rvec, tvec = cv2.calibrateCamera (self.Q2_3d_points, self.Q2_2d_points, img_shape, None, None)
            start_point = [[7,5,0], [4,5,0], [1,5,0], [7,2,0], [4,2,0], [1,2,0]]# the designated position
            for i, img in enumerate(self.Q2_img_list):
                draw_img = img.copy()
                for text in range(len(textboxValue)):
                    alphabet = str(textboxValue[text])
                    charpoints = fs.getNode(alphabet).mat()# Get the 3D object coordinates of the alphabet
                    charpoints = charpoints.reshape(-1, 3).astype(np.float32)
                    charpoints = charpoints + start_point[text]# Apply translation to 3D object coordinates to move to the designated position
                    newCharPoints_2d, _ = cv2.projectPoints(charpoints, rvec[i], tvec[i], ins, dist)

                    for j in range(0, len(newCharPoints_2d), 2):
                        pointA = tuple(newCharPoints_2d[j][0].astype(int))
                        pointB = tuple(newCharPoints_2d[j+1][0].astype(int))
                        cv2.line(draw_img, pointA, pointB, (0, 0, 255), 8)
                cv2.imshow("Show words on vertical", cv2.resize(draw_img, (1024, 1024)))
                cv2.waitKey(700)
            cv2.destroyAllWindows()

    def Stereo_disparity_map_btn_func(self):
        stereo = cv2.StereoBM.create(numDisparities=432, blockSize=25)# create a specific stereo
        img_L = None
        img_R = None
        if os.path.exists(self.Q3_L_img_path) and self.Q3_L_img_path.endswith(".png"):
            img_L = cv2.imread(self.Q3_L_img_path)
        if os.path.exists(self.Q3_R_img_path) and self.Q3_R_img_path.endswith(".png"):
            img_R = cv2.imread(self.Q3_R_img_path)
        if img_L is not None and img_R is not None:
            grayimg_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2GRAY)
            grayimg_R = cv2.cvtColor(img_R, cv2.COLOR_BGR2GRAY)
            disparity = stereo.compute(grayimg_L, grayimg_R)# compute disparity for a specific stereo
            disparity_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)# normalized to [0, 255]
            disparity_normalized = np.uint8(disparity_normalized)
            cv2.imshow("Imgae L", cv2.resize(img_L, (704, 479)))
            cv2.imshow("Image R", cv2.resize(img_R, (704, 479)))
            cv2.imshow("Disparity map", cv2.resize(disparity_normalized, (704, 479)))
            cv2.waitKey(700)
        if img_L is None:
            self.error_message(errortext="Left image path is incorrect!")
        if img_R is None:
            self.error_message(errortext="Right image path is incorrect!")
    
    def SIFT_Keypoints_btn_func(self):
        sift = cv2.SIFT_create() #　Create a SIFT detector
        if os.path.exists(self.Q4_img_one_path) and self.Q4_img_one_path.endswith("Left.jpg"):
            img_one = cv2.imread(self.Q4_img_one_path)
            gray_img = cv2.cvtColor(img_one, cv2.COLOR_BGR2GRAY)
            keypoints, descriptors = sift.detectAndCompute(gray_img, None)# Find SIFT keypoints, each keypoint has its descriptor
            draw_img = gray_img.copy()
            draw_img = cv2.drawKeypoints(draw_img, keypoints, None, color=(0,255,0))
            cv2.imshow("Keypoints", cv2.resize(draw_img, (512, 512)))
            cv2.waitKey(700)
        else:
            self.error_message("Q4 Image path one is incorrect!")
    
    def SIFT_Matched_Keypoints_btn_func(self):
        sift = cv2.SIFT_create()
        img_one = None
        img_two = None
        if os.path.exists(self.Q4_img_one_path) and self.Q4_img_one_path.endswith("Left.jpg"):
            img_one = cv2.imread(self.Q4_img_one_path)
            gray_img_one = cv2.cvtColor(img_one, cv2.COLOR_BGR2GRAY)
            one_keypoints, one_descriptors = sift.detectAndCompute(gray_img_one, None)
        if os.path.exists(self.Q4_img_two_path) and self.Q4_img_two_path.endswith("Right.jpg"):
            img_two = cv2.imread(self.Q4_img_two_path)
            gray_img_two = cv2.cvtColor(img_two, cv2.COLOR_BGR2GRAY)
            two_keypoints, two_descriptors = sift.detectAndCompute(gray_img_two, None)
        if img_one is not None and img_two is not None:
            # https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
            matches = cv2.BFMatcher().knnMatch(one_descriptors, two_descriptors, k=2)# get 2 best matches
            good_matches = []
            for i, (m, n) in enumerate(matches):# Apply ratio test
                if m.distance < 0.75*n.distance:
                    good_matches.append(m)
            good_matches = np.expand_dims(good_matches, 1)# cv2.drawMatchesKnn accept good(N, 1)
            draw_img = cv2.drawMatchesKnn(gray_img_one, one_keypoints, gray_img_two, two_keypoints, good_matches, 
                                          None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imshow("Metched Keypoints", cv2.resize(draw_img, (1024, 512)))
            cv2.waitKey(700)
        if img_one is None:
            self.error_message("Q4 Image path one is incorrect!")
        if img_two is None:
            self.error_message("Q4 Image path two is incorrect!")
            
        


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
    
