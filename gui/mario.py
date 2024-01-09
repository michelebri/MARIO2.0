import tkinter as tk
from PIL import ImageTk, Image, ImageFont, ImageDraw
from tkinter import ttk, Label, simpledialog, OptionMenu, Toplevel, filedialog, PhotoImage
import config,webbrowser
from functools import partial
import cv2
import config
from Homography import Homography
from Game import Game
from ObjectTracker import ObjectTracker
from FallDetector import FallDetector
from PoseEstimator import PoseEstimator
from matplotlib import colors

#########INTERFACCIA GRAFICA######################


        

class MARIO:

    def start_tracking_process(self):
        #Homography
        h = Homography(config.src_homo,config.dst_homo)
        h._from_detection()
        config.homography_matrix = h.get_H()
        ot = ObjectTracker(
                preparation_instance= self.preparation_instance,
                yolo_weights_path="data/detectionCalciatori.pt", 
                strong_sort_weights="data/osnet_x1_0_msmt17.pth", 
            )
        fd = FallDetector(
            weight_path="data/tsstg-model.pth"
        )
        pe = PoseEstimator(
            "data/pose_estimation.pth",
        )

        game = Game(
            tracker=ot, 
            pose_estimator=pe,
            fall_detector=fd
        )
        game.loop()

    def setPreparationInstance(self, preparation_instance):
        print(preparation_instance)
        self.preparation_instance = preparation_instance

    def __init__(self):
        self.preparation_instance = None
        self.root0 = tk.Tk()

        self.video_button = ttk.Button(
            self.root0,
            text='CHOOSE VIDEO',
            style="Accent.TButton",
            command=lambda:[ self.openVideo(), self.switch_calib.config(state="enabled")])
        self.video_button.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        

        self.gc_button = ttk.Button(
            self.root0,
            text='GAME CONTROLLER',
            style="Accent.TButton",
            command=lambda: [self.open_gc_game()])
        self.gc_button.grid(row=2, column=2, padx=10, pady=10, sticky="nsew")


        self.go_preparation_button = ttk.Button(
            self.root0,
            text='GO TO PREPARATION',
            style="Accent.TButton",
            state="disabled",
            command=lambda: [self.start_preparation_process()])
        self.go_preparation_button.grid(row=5, column=3, padx=20, pady=10, sticky="nsew")

        self.go_tracking_button = ttk.Button(
            self.root0,
            text='GO TO TRACKING',
            style="Accent.TButton",
            state="disabled",
            command=lambda: [self.start_tracking_process()])
        self.go_tracking_button.grid(row=6, column=3, padx=20, pady=10, sticky="nsew")

        self.calib_button = ttk.Button(self.root0, text="CHOOSE CALIB", style="Accent.TButton", command= lambda: [self.openCalib()])
        self.calib_button.grid(row=6, column=1, padx=20, pady=10, sticky="nsew")

        self.switch_calib = ttk.Checkbutton(self.root0, text="Calibrated", state="enabled", style="Switch.TCheckbutton", command= lambda: self.switch())
        self.switch_calib.grid(row=6, column=0, padx=20, pady=20, sticky="nsew")

        self.switch_gpu = ttk.Checkbutton(self.root0, text="CPU", style="Switch.TCheckbutton", command= lambda: self.switch_gpu_fun())
        self.switch_gpu.grid(row=6, column=2, padx=20, pady=20, sticky="nsew")


        self.switch_poses_button = ttk.Checkbutton(self.root0, text="PoseEstimation", style="Switch.TCheckbutton", command= lambda: self.switch_poses())
        self.switch_poses_button.grid(row=2, column=1, padx=10, pady=10, sticky="nsew")

    def gui(self):

            ###FINESTRA 1
            screen_width = self.root0.winfo_screenwidth()
            screen_height = self.root0.winfo_screenheight()
            window_width = 650
            window_height = 450
            center_x = int(screen_width/2 - window_width / 2)
            center_y = int(screen_height/2 - window_height / 2)-40
            self.root0.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
            self.root0.resizable(False, False)
            self.root0.title('MARIO')
            self.root0.tk.call('source', 'gui/media/azure.tcl')
            self.root0.tk.call("set_theme", "light")
            wolves = Image.open("gui/media/unibas_wolves.png")
            spqr = Image.open("gui/media/spqr_team.png")
            uni = Image.open("gui/media/unibas.jpeg")

            titolo0 = ttk.Label(
                self.root0,
                text="MARIO",
                justify="left",
                font=("-size", 25, "-weight", "bold", "-slant", "italic"),
                foreground="#0071c1"
            )
            titolo0.grid(row=0, column=0, pady=10)

            unibas_wolves = ImageTk.PhotoImage(wolves)
            button_wolves = ttk.Button(self.root0, text="wolves", image=unibas_wolves, command=partial(webbrowser.open, "https://sites.google.com/unibas.it/wolves"))
            button_wolves.grid(row=2, column=3, padx=10, pady=10, sticky="nsew")

            
            spqr_team = ImageTk.PhotoImage(spqr)
            button_spqr = ttk.Button(self.root0, text="spqr", image=spqr_team, command=partial(webbrowser.open, "http://spqr.diag.uniroma1.it/"))
            button_spqr.grid(row=4, column=3, padx=10, pady=10, sticky="nsew")

            
            unibas = ImageTk.PhotoImage(uni)
            button_unibas = ttk.Button(self.root0, text="unibas", image=unibas, command=partial(webbrowser.open, "https://portale.unibas.it/site/home.html"))
            button_unibas.grid(row=3, column=3, padx=10, pady=10, sticky="nsew")
    
            self.root0.mainloop()

        #FUNZIONI BOTTONI

    def openVideo(self):
        self.root0.filenameVideos = filedialog.askopenfilename(initialdir="../video", title="Select video", filetypes=(("mp4 files", "*.mp4"), ("avi files", "*.avi"), ("all files", "*.*")))
        config.videoPath = self.root0.filenameVideos
        cap = cv2.VideoCapture(config.videoPath)
        while(True):
            _,frame = cap.read()
            config.imageInitialFrame = frame 
            break
        cap.release()
        uni = Image.fromarray(cv2.cvtColor(config.imageInitialFrame, cv2.COLOR_BGR2RGB))
        size = 350,350
        uni.thumbnail(size)
        test3 = ImageTk.PhotoImage(uni)
        label3 = tk.Label(image=test3)       
        label3.image = test3
        label3.grid(row=3, column=0, columnspan=3, rowspan=3, padx=20, pady=20, sticky="nsew")

        return self.root0.filenameVideos 

    def open_gc_game(self) :
        self.root0.filenameGc_game = filedialog.askopenfilename(initialdir="../", title="Select gc game file", filetypes=(("yaml files", "*.yaml"), ("all files", "*.*")))
        config.gcGameFile = self.root0.filenameGc_game     
        homeNumberTeam = 161
        homeColorTeam = 162
        homeColorGoalkeeper = 163
        awayNumberTeam = 165
        awayColorTeam = 166
        awayColorGoalkeeper = 167

        with open(config.gcGameFile, 'r') as file:
            lines = file.readlines()
            config.homeNumberTeam = int(lines[homeNumberTeam].split(":")[1].replace(" ","").replace("\n",""))
            config.homeColorTeam = colors.to_rgb(lines[homeColorTeam].split(":")[1].replace(" ","").replace("\n",""))
            config.homeColorGoalkeeper = colors.to_rgb(lines[homeColorGoalkeeper].split(":")[1].replace(" ","").replace("\n",""))
            config.awayNumberTeam = int(lines[awayNumberTeam].split(":")[1].replace(" ","").replace("\n",""))
            config.awayColorTeam = colors.to_rgb(lines[awayColorTeam].split(":")[1].replace(" ","").replace("\n",""))
            config.awayColorGoalkeeper = colors.to_rgb(lines[awayColorGoalkeeper].split(":")[1].replace(" ","").replace("\n",""))
            config.awayColorTeam = [round(x * 255) for x in config.awayColorTeam]
            config.homeColorGoalkeeper = [round(x * 255) for x in config.homeColorGoalkeeper]
            config.homeColorTeam = [round(x * 255) for x in config.homeColorTeam]
        associationTeam = {}
        f = open("gui/media/association_teams.txt")
        for line in f.readlines():
            key = line[0:line.find("=")].replace(" ","")
            team = line[line.find("=")+1:line.find(",")]
            associationTeam[int(key)] = team
        config.teamHome = (associationTeam[config.homeNumberTeam])
        config.teamAway = (associationTeam[config.awayNumberTeam])
        return self.root0.filenameGc_game

    def openCalib(self):
        self.root0.fileCalib = filedialog.askopenfilename(initialdir="../calibration_data", title="Select file", filetypes=(("npz files", "*.npz"), ("all files", "*.*")))
        config.calibPath = self.root0.fileCalib
        return self.root0.fileCalib
    
    def switch_gpu_fun(self) :
        if config.gpuSwitch == False:
            config.gpuSwitch = True
            config.device    = "cpu"
            print(config.device)
        else:
            config.gpuSwitch = False   
            config.device    = "cuda"
            print(config.device)
            
    def switch(self) :            
        if config.calibrationSwitch == True:
            config.calibrationSwitch = False
            config.toCalibrate = True
            self.go_preparation_button.config(state="disabled")
            self.calib_button.config(state="enabled")
        else:
            config.calibrationSwitch = True   
            config.toCalibrate = False
            print("Da calibrare : " , config.toCalibrate)
            self.go_preparation_button.config(state="enabled")
            self.calib_button.config(state="disabled")

    def switch_poses(self):
        if config.poseSwitch == False:
            config.poseSwitch = True
        else:
            config.poseSwitch = False

    #BOTTONI


    def start_preparation_process(self):
        state = self.preparation_instance.start_preparation()
        self.go_preparation_button.config(state="disabled")
        self.go_tracking_button.config(state="enabled")
        
    




        ###FINESTRA 2
    def open_vista_analysis(self,vista):
        vista_analysis= Toplevel(vista)
        screen_width = vista_analysis.winfo_screenwidth()
        screen_height = vista_analysis.winfo_screenheight()
        # find the center point
        window_width = 750
        window_height = 570
        center_x = int(screen_width/2 - window_width / 2)
        center_y = int(screen_height/2 - window_height / 2)-40
        vista_analysis.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        vista_analysis.resizable(False, False)
        vista_analysis.title('MARIO') 
        titolo1 = ttk.Label(
        vista_analysis,
        text="MARIO-Analysis",
        justify="center",
        font=("-size", 25, "-weight", "bold", "-slant", "italic"),
        foreground="#0071c1"
        )
        ga = GameAnalyzer("./game_data.csv")
        
        goal0, goal1, shots0, shots1, shots_target0, shots_target1, pass0, pass1 = ga.stats()
        poss0, poss1 = ga.ball_possession()  
        titolo1.grid(row=0, column=0, pady=10)    
        game_data = pd.read_csv("./game_data.csv")
        id_list = list(set(game_data[ga.game_data["team"] != -1]["id"].tolist()))     
        
        def aggiorna_labelsNew(goal0, goal1, shots0, shots1, shots_target0, shots_target1, poss0, poss1, pass0, pass1):
            finestra_stats = Toplevel(vista_analysis)
            screen_width = vista_analysis.winfo_screenwidth()
            screen_height = vista_analysis.winfo_screenheight()
            window_width = 470
            window_height = 400
            center_x = int(screen_width/2 - window_width / 2)
            center_y = int(screen_height/2 - window_height / 2)-40
            finestra_stats.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
            finestra_stats.resizable(False, False)
            finestra_stats.title('ViSta-stats')
            fnt = ImageFont.truetype("gui/Champions-Bold.ttf", 20)
            fntTeam = ImageFont.truetype("gui/Champions-Bold.ttf", 25)
            champions_image = Image.open("gui/champions.jpg")
            edit_image = ImageDraw.Draw(champions_image)
            edit_image.text((20,155), s.getHomeTeam(), ("white"), fntTeam)
            edit_image.text((300,155), s.getAwayTeam() , ("white"), fntTeam)
            edit_image.text((190,155), str(goal0), ("white"), fnt)
            edit_image.text((260,155), str(goal1), ("white"), fnt)
            edit_image.text((20,215), str(poss0)+"\n"+str(shots0)+"\n"+str(shots_target0)+"\n"+str(pass0), ("white"), fnt, spacing=10)
            edit_image.text((350,215), str(poss1)+"\n"+str(shots1)+"\n"+str(shots_target1)+"\n"+str(pass1), ("white"), fnt, spacing=10)
        
            champions_image.save("./stats.png")
            global stats_image
            stats_image = PhotoImage(file="./stats.png")
            label=Label(finestra_stats, image=stats_image)
            label.pack(pady=20)
            label.config(image=stats_image)
        
    
        #POSSESSION
        stats_button = ttk.Button(
            vista_analysis,
            text='CALCULATE STATS',
            style="Accent.TButton",
            command=lambda: [aggiorna_labelsNew(goal0, goal1, shots0, shots1, shots_target0, shots_target1, poss0, poss1, pass0, pass1)])           
        stats_button.grid(row=10, column=3, padx=20, pady=10, sticky="nsew")            
        #HEATMAP           
        menu_heatmap = tk.Menu()
        menu_heatmap.add_command(label="Robot 1", command=lambda: ga.heatmap(1))
        menu_heatmap.add_command(label="Robot 2", command=lambda: ga.heatmap(2))
        menu_heatmap.add_command(label="Robot 3", command=lambda: ga.heatmap(3))
        menu_heatmap.add_command(label="Robot 4", command=lambda: ga.heatmap(4))
        menu_heatmap.add_command(label="Robot 5", command=lambda: ga.heatmap(5))
        menu_heatmap.add_command(label="Robot 6", command=lambda: ga.heatmap(6))
        menu_heatmap.add_command(label="Robot 7", command=lambda: ga.heatmap(7))
        menu_heatmap.add_command(label="Robot 8", command=lambda: ga.heatmap(8))
        menu_heatmap.add_command(label="Robot 9", command=lambda: ga.heatmap(9))
        menu_heatmap.add_command(label="Robot 10", command=lambda: ga.heatmap(10))
        menu_heatmap.add_command(label="ball", command=lambda: ga.heatmap(11))           
        menubutton_heatmap = ttk.Menubutton(
            vista_analysis, text="HEATMAP", menu=menu_heatmap, direction="below"
        )
        menubutton_heatmap.grid(row=2, column=0, padx=20, pady=20, sticky="nsew")     
        #TRACKMAP            
        menu_trackmap = tk.Menu()
        menu_trackmap.add_command(label="Robot 1", command=lambda: ga.trackmap(1))
        menu_trackmap.add_command(label="Robot 2", command=lambda: ga.trackmap(2))
        menu_trackmap.add_command(label="Robot 3", command=lambda: ga.trackmap(3))
        menu_trackmap.add_command(label="Robot 4", command=lambda: ga.trackmap(4))
        menu_trackmap.add_command(label="Robot 5", command=lambda: ga.trackmap(5))
        menu_trackmap.add_command(label="Robot 6", command=lambda: ga.trackmap(6))
        menu_trackmap.add_command(label="Robot 7", command=lambda: ga.trackmap(7))
        menu_trackmap.add_command(label="Robot 8", command=lambda: ga.trackmap(8))
        menu_trackmap.add_command(label="Robot 9", command=lambda: ga.trackmap(9))
        menu_trackmap.add_command(label="Robot 10", command=lambda: ga.trackmap(10))
        menu_trackmap.add_command(label="ball", command=lambda: ga.trackmap(11))          

        menubutton_trackmap = ttk.Menubutton(
            vista_analysis, text="TRACKMAP", menu=menu_trackmap, direction="below"
        )
        menubutton_trackmap.grid(row=2, column=1, padx=20, pady=20, sticky="nsew")        
        #SHOT-PASS MAP
        shotmap_button = ttk.Button(
            vista_analysis,
            text='PASS-SHOT MAP',
            style="Accent.TButton",
            command=lambda: ga.pass_shot_map())
        shotmap_button.grid(row=2, column=2, padx=20, pady=20, sticky="nsew")

        illegal_defender_button = ttk.Button(
            vista_analysis,
            text='ILL.DEF.',
            style="Accent.TButton",
            command=lambda: ga.illegal_defender())
        illegal_defender_button.grid(row=2, column=3, padx=20, pady=20, sticky="nsew")

    def open_vista_tracking(vista):
        open_vista_tracking = Toplevel(vista)
        screen_width = open_vista_tracking.winfo_screenwidth()
        screen_height = open_vista_tracking.winfo_screenheight()
        window_width = 750
        window_height = 570
        center_x = int(screen_width/2 - window_width / 2)
        center_y = int(screen_height/2 - window_height / 2)-40
        open_vista_tracking.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        open_vista_tracking.resizable(False, False)
        open_vista_tracking.title('MARIO')
        titolo_calibration = ttk.Label(open_vista_tracking, text="MARIO-Tracking", justify="center", font=("-size", 25, "-weight", "bold", "-slant", "italic"), foreground="#0071c1")
        titolo_calibration.grid(row=0, column=0, pady=10)
        

        

        go_to_analysis_button = ttk.Button(
            open_vista_tracking,
            text='Go TO ANALYSIS',
            style="Accent.TButton",
            command=lambda: open_vista_analysis(open_vista_tracking))
            
        go_to_analysis_button.grid(row=2, column=2, padx=20, pady=20, sticky="nsew")

    def open_vista_stats(vista):
        open_vista_stats = Toplevel(vista)
        screen_width = open_vista_stats.winfo_screenwidth()
        screen_height = open_vista_stats.winfo_screenheight()
        window_width = 750
        window_height = 570
        center_x = int(screen_width/2 - window_width / 2)
        center_y = int(screen_height/2 - window_height / 2)-40
        open_vista_stats.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
        open_vista_stats.resizable(False, False)
        open_vista_stats.title('MARIO')
        titolo_stats = ttk.Label(open_vista_stats, text="MARIO-Stats", justify="center", font=("-size", 25, "-weight", "bold", "-slant", "italic"), foreground="#0071c1")
        titolo_stats.grid(row=0, column=0, pady=10)    
    
    ####PROGRESS BAR
        
            
        
    

 
