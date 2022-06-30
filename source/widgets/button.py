import kivy.uix.button as kv
from kivy.graphics import RoundedRectangle, Color

class Button(kv.Button):
    
    def __init__(self, **kwargs):
        super(Button, self).__init__(**kwargs)
        
        with self.canvas:
            Color(1, 1, 0, 0.5)
            RoundedRectangle(
                pos = self.pos,
                radius = (10, 10, 10, 10)
            )