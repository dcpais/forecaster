from kivy.uix.anchorlayout import AnchorLayout

class HoltModeller(AnchorLayout):

    def __init__(self, **kwargs):
        super(HoltModeller, self).__init__(**kwargs)
        
        self.plot = GraphWidget()