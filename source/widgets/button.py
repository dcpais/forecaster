import kivy.uix.button as kv

class Button(kv.Button):
    
    def __init__(self, **kwargs):
        super(Button, self).__init__(**kwargs)
        self.color = (1, 1, 0, 0.5)
        with self.canvas:
            RoundedRectangle