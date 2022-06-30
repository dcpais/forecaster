from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg

class GraphWidget(FigureCanvasKivyAgg):
    
    def __init__(self, canvas):
        super().__init__(canvas)
        
