'''
This is a sample class that you can use to control the mouse pointer.
It uses the pyautogui library. You can set the precision for mouse movement
(how much the mouse moves) and the speed (how fast it moves) by changing 
precision_dict and speed_dict.
Calling the move function with the x and y output of the gaze estimation model
will move the pointer.
This class is provided to help get you started; you can choose whether you want to use it or create your own from scratch.
'''
import pyautogui

class MouseController:
    def __init__(self, precision, speed):
        precision_dict = {'high':100, 'low':1000, 'medium':500}
        speed_dict = {'fast':1, 'slow':10, 'medium':5}

        self.precision = precision_dict[precision]
        self.speed = speed_dict[speed]
        pyautogui.FAILSAFE = False

    def move(self, x, y):
        current_x, current_y = pyautogui.position()
        dist_x = x*self.precision
        dist_y = -1*y*self.precision
        if(pyautogui.onScreen(current_x+dist_x, current_y+dist_y)):
            pyautogui.moveRel(dist_x, dist_y, duration=self.speed)