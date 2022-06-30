from source.panes.mainmenu import MainMenu
from kivy.app import App

class MainApp(App):
    
    def build(self):
        return MainMenu()
        

if __name__ == "__main__":
    app = MainApp()
    app.run()