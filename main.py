from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label

class MainApp(App):
    
    def build(self):
        box = BoxLayout()
        box.add_widget(Label(
            text = "Hello",
            color = (1, 1, 1, 1)
        ))
        box.add_widget(Button(
            text = "Click me!"
        ))
        return box
        

if __name__ == "__main__":
    app = MainApp()
    app.run()