from kivy.uix.anchorlayout import AnchorLayout

from source.widgets.graphwidget import GraphWidget
from source.widgets.datawidget import DataWidget

from source.models.holtwinters import HoltWinters
from source.utility.plots import hwplot
from statsmodels.tsa.api import ExponentialSmoothing

class HoltWintersContext(AnchorLayout):

    def __init__(self, **kwargs):
        """
        Initialize a Holt winters window. 

        Within this window / context, we will be able to actively
        manipulate a HW model that has been fitted to the data
        provided. 

        There are four 'tabs' in this context:
            - The data selector tab
            - The graph preview tab
            - The model adjustment tab
            - The presentation preview tab
        """
        super(HoltWintersContext, self).__init__(**kwargs)
        
        import matplotlib.pyplot as plt
        plt.plot([10, 10], [0, 5])
        self.plot_handle = plt.gcf()

        self.data_manager = DataWidget()
        self.graph_view = GraphWidget(self.plot_handle)
        self.model_adjustment = 




data = pd.read_csv("data.csv")
dates = data["date"]
rev = data["sales"]

start_date = dates[0]
model = ExponentialSmoothing(
    rev, seasonal = "mul",
    trend = "mul", seasonal_periods = 12
).fit( optimized = True)
f_length = len(dates) + 20
forecast = model.predict(0, f_length)