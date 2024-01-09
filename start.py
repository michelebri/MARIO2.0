from gui.mario import MARIO
from Preparation import Preparation
import config

config.background = None
config.videoPath = None
config.new_field = None
config.preparationPhaseEnd = False

config.teamHome = None
config.teamAway = None
config.calibPath = "data/calibration_parameters.npz"

config.filenameVideos = None
config.imageInitialFrame = None

config.gcTeamFile = None
config.gcGameFile = None

config.gpuSwitch = False
config.poseSwitch = False
config.calibrationSwitch = False
config.toCalibrate = True
config.device = "cuda"

config.homeNumberTeam = None
config.homeColorTeam = None
config.homeColorGoalkeeper = None
config.awayNumberTeam = None
config.awayColorTeam = None
config.awayColorGoalkeeper = None

mario_instance = MARIO()
preparation_instance = Preparation(mario_instance=mario_instance)
mario_instance.setPreparationInstance(preparation_instance=preparation_instance)

mario_instance.gui()