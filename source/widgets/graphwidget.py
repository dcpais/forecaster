from kivy.garden.matplotlib.backend_kivyagg import FigureCanvasKivyAgg
from kivy.graphics import RoundedRectangle

class GraphWidget(FigureCanvasKivyAgg):
    
    def __init__(self, canvas):
        super().__init__(canvas)
        with self.canvas:
            RoundedRectangle(
                pos = self.pos,
                radius = (15, 15, 15, 15)
            )
        
