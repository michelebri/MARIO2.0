import config 
import os
import numpy as np
import cv2
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import threading
import torch
import cv2



class Preparation:

    def __init__(self,mario_instance):
        self.mario_instance = mario_instance
        if config.calibPath is not None:
            self.calib_params = np.load(config.calibPath)
        self.idx = 0
        self.calibration_thread = None
        self.max_idx = 1000
        self.root = tk.Tk()
        self.root.title("Calibration Progress")

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.root, orient="horizontal", length=300, mode="determinate", variable=self.progress_var)
        self.progress_bar.pack(pady=20)
        self.progress_label = tk.Label(self.root, text="")
        self.progress_label.pack()
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()

        x_coordinate = (screen_width - self.root.winfo_reqwidth()) // 2
        y_coordinate = (screen_height - self.root.winfo_reqheight()) // 2

        self.root.geometry(f"+{x_coordinate}+{y_coordinate}")
        print(os.getcwd())
        self.model = torch.hub.load('yolov5','custom', path='data/detectionT.pt',force_reload=True,source='local')


    def calibrateFrame(self, frameToCalibrate):
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            self.calib_params["K"],
            self.calib_params["D"],
            np.eye(3),
            self.calib_params["new_K"],
            (3400, 1912),
            cv2.CV_16SC2
        )

        undistorted_img = cv2.remap(
            frameToCalibrate,
            map1,
            map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT
        )
        undistorted_img = cv2.resize(undistorted_img,[1080,720])
        return undistorted_img

    


    def save_frame_for_background(self):
        video = cv2.VideoCapture(config.videoPath)
        ret,frameVideo = video.read()
        save_frame = True
        while(ret and save_frame):
            
            undistorted_img = frameVideo
            _, r = divmod(self.idx, 30)
            if r == 0 and self.idx < self.max_idx: 
                cv2.imwrite(
                    os.path.join(os.getcwd(), 'imbs-mt/images', str(self.idx)+ ".jpg"),
                    undistorted_img
                )
            elif self.idx > self.max_idx:
                save_frame = False
            ret,frameVideo = video.read()
            self.idx += 1
            
    def start_calibration(self):
        self.calibration_thread = threading.Thread(target=self.save_frame_for_background)
        self.calibration_thread.start()
        self.root.after(100, self.update_progress)
        self.root.mainloop()

    def update_progress(self):
        self.progress_bar.start(10)
        self.progress_label.config(text=f"Calibrating... {self.idx}/{self.max_idx} frames")
        if self.idx > self.max_idx -100:
            self.progress_label.config(text=f"Background substraction ... Please wait!")
        if self.idx >= self.max_idx:
            self.progress_bar.stop()
            self.background_save()
        else:
            self.root.after(100, self.update_progress)

    def background_save(self):
        os.system("./imbs-mt/bin/imbs-mt -img imbs-mt/images/0.jpg")  
        img = cv2.imread("data/background.jpg")
        config.src_homo = img
        results = self.model(config.src_homo)
        table_results = results.pandas().xyxy[0]
        config.detectionT = table_results
        count_of_area = sum(1 for item in list(table_results['name']) if "area" in item)
        if count_of_area > 2:
            config.new_field = True
            config.dst_homo = cv2.imread("data/field_new.png")
            messagebox.showinfo("Preparation Completed", "You are working on a new field")
            self.root.destroy()
            self.calibration_thread.join()

            self.mario_instance.start_tracking_process()
        else:
            config.new_field = False
            config.dst_homo = cv2.imread("data/field_old.png")
            messagebox.showinfo("Preparation Completed", "You are working on an old field")
            self.root.destroy()
            self.calibration_thread.join()
            self.mario_instance.start_tracking_process()
            
        

    def start_preparation(self):   
        self.start_calibration()


