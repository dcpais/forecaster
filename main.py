from source.contexts.mainmenu import MainMenu
from source.contexts.hwcontext import HoltWintersContext
from kivy.app import App

class MainApp(App):
    
    def build(self):
        return HoltWintersContext()
        

if __name__ == "__main__":
    app = MainApp()
    app.run()