from kivy.uix.boxlayout import BoxLayout

class DataWidget(BoxLayout):

    def __init__(self, **kwargs):
        super(DataWidget, self).__init__(**kwargs)
        self.text = "CONTROL THE DATA HERE!"