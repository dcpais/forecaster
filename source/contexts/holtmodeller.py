from kivy.uix.anchorlayout import AnchorLayout

from source.widgets.graphwidget import GraphWidget

class HoltWintersContext(AnchorLayout):

    def __init__(self, **kwargs):
        super(HoltWintersContext, self).__init__(**kwargs)
        

        self.plot = GraphWidget()