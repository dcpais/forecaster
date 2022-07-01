from kivy.uix.gridlayout import GridLayout

class ModelController(GridLayout):

    def __init__(self, layout: dict, **kwargs):
        """
        Initialize a ModelController instance.

        A Model controller is a panel that will have 
        """
        super(ModelController, self).__init__(**kwargs)

        # Set up our controller layout following the 
        # rules below:
        # 
        #   - Every 'key' in the layout dict is the 
        #     start of a new section
        #   - 