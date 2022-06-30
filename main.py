from source.contexts.mainmenu import MainMenu
from source.widgets
from kivy.app import App

class MainApp(App):
    
    def build(self):
        return HoltWintersPane()
        

if __name__ == "__main__":
    app = MainApp()
    app.run()